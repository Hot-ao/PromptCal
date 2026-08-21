"""
결정 실험: cv3@A11(성공한 mixed-precision) 위에 v2 decision loss 학습을 얹으면
held-out이 '유지/개선'되나 '붕괴'하나?

동기: 유저 직감 = cv3 정밀도 보호 + decision loss 결합이 더 method답다.
      Claude 우려 = v2에서 decision loss 학습은 anti-transfer로 held-out을 악화시켰다.
      → 말싸움 대신 데이터로 결정. cv3@A11 기반 위에 v2를 실제로 돌려 held-out 비교.

비교(같은 seed·같은 split):
  naive            : 전부 W8A8
  Ours (cv3@A11)   : cv3 activation만 A11, 학습 없음 (현재 method)
  Ours+decision    : cv3@A11 위에 convert_to_learnable_scale + optimize_promptcal_v2
                     (removed=H_cal, keep=S). cv3의 A11 정밀도는 유지된 채 scale만 학습.

판정:
  - Ours+decision ≤ Ours 이고 seed 일관 → 결합이 안전/유익 (유저 직감 옳음 → 2번으로).
  - Ours+decision > Ours (naive 쪽으로 되돌아감) → 학습이 anti-transfer 재유입 (결합 해로움 → mixed-precision 단독).

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/27_combine_test.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --iters 150 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.fake_quant import QuantConv2d
from src.quant.pdquant import _find_head
from src.quant.promptcal_v2 import convert_to_learnable_scale, optimize_promptcal_v2

CV3_ABITS = 11


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

def cv3_convs(det):
    head=_find_head(det); br=getattr(head,"cv3",None)
    return [] if br is None else [m for m in br.modules() if isinstance(m,QuantConv2d)]

def build_base(model_cls, w, names, device, calib, ours_cv3=False):
    """naive 또는 Ours(cv3@A11) 정적 모델."""
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8)
    if ours_cv3:
        for c in cv3_convs(m.model): c.a_obs.bits=CV3_ABITS
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    return m

def build_combine(model_cls, w, names, device, calib, fp, removed, keep, iters):
    """Ours(cv3@A11) 위에 v2 decision-loss 학습(scale)을 얹은 모델."""
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8)
    for c in cv3_convs(m.model): c.a_obs.bits=CV3_ABITS      # cv3 A11 정밀도 먼저
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    convert_to_learnable_scale(m.model)                       # 그 위에 학습 scale
    optimize_promptcal_v2(m.model, fp.model, calib, device, removed, keep,
                          iters=iters, lr=3e-3, temp=1.0, verbose=False)
    return m

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    Hm=torch.zeros(80,dtype=torch.bool); Hm[H_eval]=True; tot=fl=0
    for i,t in enumerate(probe):
        sf=h_fp.run_image(t,i).sim; sq=h_q.run_image(t,i).sim
        prob=sf.sigmoid(); mp,c_fp=prob.max(-1); target=(mp>conf)&Hm[c_fp]
        if target.sum()==0: continue
        idx=target.nonzero(as_tuple=True)[0]
        fpK=sf.clone(); fpK[:,Hm]=-1e9; qK=sq.clone(); qK[:,Hm]=-1e9
        fl+=int((fpK.argmax(-1)[idx]!=qK.argmax(-1)[idx]).sum()); tot+=len(idx)
    return fl/max(tot,1)*100

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=500)
    ap.add_argument("--iters",type=int,default=150)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe=[preprocess(p,args.imgsz,device) for p in imgs[args.calib:args.calib+args.eval]]

    print("[build] FP"); fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    print("[build] naive"); nv=build_base(YOLOWorld,args.model,names,device,calib,ours_cv3=False)
    print("[build] Ours (cv3@A11)"); ou=build_base(YOLOWorld,args.model,names,device,calib,ours_cv3=True)
    h_fp=SimilarityHarness(fp.model,device=device)
    h_nv=SimilarityHarness(nv.model,device=device)
    h_ou=SimilarityHarness(ou.model,device=device)

    print("\n"+"="*76)
    print(" 결합 검증: cv3@A11 위에 decision loss를 얹으면 held-out이 유지되나 붕괴하나")
    print("="*76)
    print(f"{'seed':>5} | {'naive':>7} | {'Ours(cv3@A11)':>13} | {'Ours+decision':>13} | {'combine효과':>10}")
    print("-"*76)
    nl,ol,cl=[],[],[]
    for s in args.seeds:
        rng=np.random.default_rng(s); perm=rng.permutation(80)
        S=perm[:40].tolist(); H_cal=perm[40:60].tolist(); H_eval=perm[60:80].tolist()
        cm=build_combine(YOLOWorld,args.model,names,device,calib,fp,H_cal,S,args.iters)
        h_cm=SimilarityHarness(cm.model,device=device)
        fn=heldout_flip(h_fp,h_nv,probe,H_eval)
        fo=heldout_flip(h_fp,h_ou,probe,H_eval)
        fc=heldout_flip(h_fp,h_cm,probe,H_eval)
        h_cm.close()
        nl.append(fn); ol.append(fo); cl.append(fc)
        eff=(fc-fo)/max(fo,1e-9)*100    # +면 결합이 나쁨(Ours보다 flip↑)
        print(f"{s:>5} | {fn:>6.2f}% | {fo:>12.2f}% | {fc:>12.2f}% | {eff:>+9.1f}%")
    h_fp.close(); h_nv.close(); h_ou.close()
    print("-"*76)
    eff=(np.mean(cl)-np.mean(ol))/max(np.mean(ol),1e-9)*100
    print(f"{'mean':>5} | {np.mean(nl):>6.2f}% | {np.mean(ol):>12.2f}% | {np.mean(cl):>12.2f}% | {eff:>+9.1f}%")
    print("="*76)
    print("\n판정:")
    print(" - Ours+decision ≤ Ours (combine효과 ≤ 0) 이고 seed 일관 → 결합 안전/유익. 유저 직감 옳음.")
    print(" - Ours+decision > Ours (combine효과 > 0, naive 쪽 회귀) → 학습이 anti-transfer 재유입. 결합 해로움.")

if __name__=="__main__":
    main()
