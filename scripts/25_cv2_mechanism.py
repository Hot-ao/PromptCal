"""
cv2 기전 규명: cv2-FP가 유사도(cv4)를 바꾸는 지점을 특정한다.
(cv2는 box branch라 이론상 cv4 경로 밖 → 그런데 유사도를 57% 바꿈. 어디서 새는가?)

절차:
  0) head 실제 구조 출력 — 이 checkpoint에서 cv2/cv3/cv4가 무엇인지 직접 확인.
  1) naive vs cv2-FP 두 빌드에서 cv3 conv의 calibration scale이 동일한가?
     · 다르면 → cv2-FP가 cv3의 calibration을 흔든 것(간접 confound).
  2) 같은 이미지에서 계단식 비교(max|Δ|):
     · cv3 입력 x[i]  : 다르면 → 상류(neck)가 다름(공유 경로/스킵 오염).
     · cv3 출력       : 입력·scale 같은데 다르면 → cv3 양자화 비결정성/버그.
     · cv4 출력(sim)  : 위가 다 같은데 다르면 → cv4 내부/harness 쪽(text? 측정?).

판정 트리:
  - cv3 입력/scale/출력 모두 동일한데 cv4만 다름 → cv2는 cv4를 물리적으로 못 바꿈.
    = 앞서 본 sim 차이는 '측정 방식' 문제일 가능성(빌드 간 비결정/상태 누수). 재현 재점검.
  - cv3 scale이 다름 → calibration coupling. (cv2-FP forward가 cv3 관측을 바꿈? 이론상 아님 → 재현)
  - cv3 입력이 다름 → 상류 오염. skip_names={'cv2'}가 공유/다른 모듈을 건드렸는지 확인.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/25_cv2_mechanism.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.fake_quant import QuantConv2d
from src.quant.pdquant import _find_head


def load_coco_names():
    import ultralytics, yaml
    from pathlib import Path
    d = yaml.safe_load(open(Path(ultralytics.__file__).parent/"cfg"/"datasets"/"coco.yaml"))
    return [d["names"][i] for i in range(len(d["names"]))]

def letterbox(im, new=640, color=(114,114,114)):
    h,w=im.shape[:2]; r=min(new/h,new/w); nh,nw=int(round(h*r)),int(round(w*r))
    im_r=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
    top,left=(new-nh)//2,(new-nw)//2
    return cv2.copyMakeBorder(im_r,top,new-nh-top,left,new-nw-left,cv2.BORDER_CONSTANT,value=color)

def preprocess(path, imgsz, device):
    im=letterbox(cv2.imread(path),imgsz)
    im=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))
    return torch.from_numpy(im).float().unsqueeze(0).to(device)/255.0

def branch_convs(det, branch):
    head=_find_head(det); br=getattr(head,branch,None)
    return [] if br is None else [m for m in br.modules() if isinstance(m,QuantConv2d)]

def build(model_cls, w, names, device, calib, skip=None):
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8, skip_names=skip or set())
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    return m


class HeadTap:
    """head.cv3 입력/출력, cv4 출력을 레벨별로 캡처."""
    def __init__(self, head):
        self.i3={}; self.o3={}; self.o4={}; self.h=[]
        for i,sub in enumerate(head.cv3):
            def mk(i):
                def hook(m,inp,out): self.i3[i]=inp[0].detach().float(); self.o3[i]=out.detach().float()
                return hook
            self.h.append(sub.register_forward_hook(mk(i)))
        for i,sub in enumerate(head.cv4):
            def mk4(i):
                def hook(m,inp,out): self.o4[i]=out.detach().float()
                return hook
            self.h.append(sub.register_forward_hook(mk4(i)))
    def close(self):
        for h in self.h: h.remove()

def maxdiff(a, b):
    ks=sorted(set(a)&set(b)); 
    if not ks: return None
    return max(float((a[k]-b[k]).abs().max()) for k in ks)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe0=preprocess(imgs[args.calib], args.imgsz, device)   # 비교용 이미지 1장

    # ---- 0) head 구조 ----
    print("[0] head 구조 (이 checkpoint에서 cv2/cv3/cv4가 무엇인가)")
    fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    head=_find_head(fp.model)
    print(f"    head type: {type(head).__name__}")
    for nm,ch in head.named_children():
        extra=""
        if hasattr(ch,"__len__"):
            try: extra=f" (len={len(ch)}, elem0={type(ch[0]).__name__})"
            except Exception: pass
        print(f"    - {nm}: {type(ch).__name__}{extra}")

    # ---- 빌드 2종 ----
    nv=build(YOLOWorld,args.model,names,device,calib)                # naive
    c2=build(YOLOWorld,args.model,names,device,calib,skip={"cv2"})   # cv2-FP
    print(f"\n    quantConv: naive={sum(1 for m in nv.model.modules() if isinstance(m,QuantConv2d))}, "
          f"cv2-FP={sum(1 for m in c2.model.modules() if isinstance(m,QuantConv2d))}")
    print(f"    cv3 quantConv 유지: naive={len(branch_convs(nv.model,'cv3'))}, cv2-FP={len(branch_convs(c2.model,'cv3'))}")

    # ---- 1) cv3 calibration scale 동일성 ----
    s_nv=[float(c.a_obs.scale) for c in branch_convs(nv.model,"cv3")]
    s_c2=[float(c.a_obs.scale) for c in branch_convs(c2.model,"cv3")]
    if len(s_nv)==len(s_c2) and len(s_nv)>0:
        sd=max(abs(a-b) for a,b in zip(s_nv,s_c2))
        print(f"\n[1] cv3 activation scale 최대차(naive vs cv2-FP): {sd:.3e}  "
              f"→ {'동일' if sd<1e-9 else '다름(!)'} ")
    else:
        print("\n[1] cv3 conv 수 불일치 — 구조 확인 필요")

    # ---- 2) 같은 이미지에서 계단식 비교 ----
    tap_nv=HeadTap(_find_head(nv.model)); tap_c2=HeadTap(_find_head(c2.model))
    with torch.no_grad():
        nv.model(probe0); c2.model(probe0)
    d_in=maxdiff(tap_nv.i3, tap_c2.i3)
    d_o3=maxdiff(tap_nv.o3, tap_c2.o3)
    d_o4=maxdiff(tap_nv.o4, tap_c2.o4)
    tap_nv.close(); tap_c2.close()
    print("\n[2] 같은 이미지 계단식 max|Δ| (naive vs cv2-FP):")
    print(f"    cv3 입력 x[i] : {d_in:.4e}")
    print(f"    cv3 출력      : {d_o3:.4e}")
    print(f"    cv4 출력(sim) : {d_o4:.4e}")

    print("\n[판정]")
    tol=1e-6
    if d_in is not None and d_in>tol:
        print(" cv3 '입력'이 다름 → 상류(neck)가 바뀜. cv2 skip이 공유/다른 모듈을 건드렸는지 확인.")
    elif d_o3 is not None and d_o3>tol:
        print(" 입력 같은데 cv3 '출력'이 다름 → cv3 양자화가 빌드 간 다름(scale[1] 결과와 대조).")
    elif d_o4 is not None and d_o4>tol:
        print(" cv3까지 동일한데 cv4만 다름 → cv2는 cv4를 물리적으로 못 바꿈.")
        print("   = 앞선 sim 차이는 '측정/빌드 비결정' 쪽. 23의 cv2-FP -12%는 confound로 재검토.")
    else:
        print(" 전부 동일 → cv2-FP는 사실상 아무것도 안 바꿈. 앞선 sim_delta 측정 자체를 재점검.")
    print("\n(주의: 이 스크립트가 cv3 전부 동일을 보이면, 23의 cv2-FP 개선은 method 근거가 못 됨 → cv3-only 확정.)")

if __name__=="__main__":
    main()
