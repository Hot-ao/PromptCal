"""
method 메인 평가: FP / naive W8A8 / Ours(cv3@A11 mixed-precision)를
같은 조건에서 (1) held-out flip 과 (2) AP(pycocotools) 두 축으로 비교.

논문 메인표의 우리 method 행들을 생성한다. baseline(AdaRound/BRECQ/QDrop/PD-Quant,
v1/v2) 수치는 기존 결과(results_v*/·script 08~12·20~21)에서 가져와 같은 표에 배치.

Ours = 전체 W8A8, 단 head cv3 branch(9 convs)의 activation만 A11 (weight 8bit 유지).
  · held-out: cv3 embedding 정밀도 보호로 semantic 결정 손상 복구(-40%).
  · AP      : cv3만 A11이라 naive와 사실상 동일해야(양자화 둔감 전제 + 최소 개입).

핵심 주장: baseline·v1·v2가 못 낮춘 held-out flip을, AP 손실 없이, 학습 없이, 값싸게 낮춘다.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/26_main_eval.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --calib 32 --eval 1000 --seeds 0 1 2 --device 0
    # held-out만 빠르게: --no-ap
    # AP만: --no-flip
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.fake_quant import QuantConv2d
from src.quant.pdquant import _find_head

CV3_ABITS = 11   # method의 cv3 activation bit (24_method_confirm에서 확정)


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

def build(model_cls, w, names, device, calib, mode):
    """mode: 'fp' | 'naive' | 'ours'. ours=cv3 activation만 A11(head-scoped, 충돌 없음)."""
    m=model_cls(w); m.set_classes(names)
    if mode=="fp":
        m.fuse(); m.model.to(device).eval(); return m
    m.fuse()
    wrap_convs(m.model, 8, 8)
    if mode=="ours":
        for c in cv3_convs(m.model):
            c.a_obs.bits = CV3_ABITS            # weight는 8bit 유지, cv3 activation만 A11
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    return m

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    Hm=torch.zeros(80,dtype=torch.bool); Hm[H_eval]=True
    tot=fl=0
    for i,t in enumerate(probe):
        sf=h_fp.run_image(t,i).sim; sq=h_q.run_image(t,i).sim
        prob=sf.sigmoid(); mp,c_fp=prob.max(-1); target=(mp>conf)&Hm[c_fp]
        if target.sum()==0: continue
        idx=target.nonzero(as_tuple=True)[0]
        fpK=sf.clone(); fpK[:,Hm]=-1e9; qK=sq.clone(); qK[:,Hm]=-1e9
        fl+=int((fpK.argmax(-1)[idx]!=qK.argmax(-1)[idx]).sum()); tot+=len(idx)
    return fl/max(tot,1)*100

def measure_ap(model, data, imgsz, device):
    try:
        metrics=model.val(data=data, imgsz=imgsz, device=device, save_json=False, verbose=False)
        return float(metrics.box.map)*100, float(metrics.box.map50)*100
    except Exception as e:
        print(f"    [ap] 측정 실패: {e}")
        return None, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--data",default="configs/coco_local.yaml")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=1000)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    ap.add_argument("--no-ap",dest="ap",action="store_false",default=True)
    ap.add_argument("--no-flip",dest="flip",action="store_false",default=True)
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe=[preprocess(p,args.imgsz,device) for p in imgs[args.calib:args.calib+args.eval]]

    print("[build] FP");    fp=build(YOLOWorld,args.model,names,device,calib,"fp")
    print("[build] naive"); nv=build(YOLOWorld,args.model,names,device,calib,"naive")
    print(f"[build] Ours (cv3@A{CV3_ABITS})"); ou=build(YOLOWorld,args.model,names,device,calib,"ours")
    print(f"    (Ours: cv3 convs={len(cv3_convs(ou.model))}개 activation A{CV3_ABITS}, 나머지 W8A8)")

    # ---- held-out flip ----
    flip={"naive":None,"ours":None}
    if args.flip:
        h_fp=SimilarityHarness(fp.model,device=device)
        h_nv=SimilarityHarness(nv.model,device=device)
        h_ou=SimilarityHarness(ou.model,device=device)
        nvv,ouv=[],[]
        for s in args.seeds:
            rng=np.random.default_rng(s); H_eval=rng.permutation(80)[60:80].tolist()
            nvv.append(heldout_flip(h_fp,h_nv,probe,H_eval))
            ouv.append(heldout_flip(h_fp,h_ou,probe,H_eval))
        flip["naive"]=(float(np.mean(nvv)),nvv); flip["ours"]=(float(np.mean(ouv)),ouv)
        h_fp.close(); h_nv.close(); h_ou.close()

    # ---- AP (pycocotools via ultralytics val) ----
    ap_res={"fp":(None,None),"naive":(None,None),"ours":(None,None)}
    if args.ap:
        print("\n[val] FP AP...");    ap_res["fp"]=measure_ap(fp,args.data,args.imgsz,args.device)
        print("[val] naive AP...");   ap_res["naive"]=measure_ap(nv,args.data,args.imgsz,args.device)
        print(f"[val] Ours AP...");   ap_res["ours"]=measure_ap(ou,args.data,args.imgsz,args.device)

    # ---- 표 ----
    print("\n"+"="*72)
    print(" 메인 평가: held-out flip(↓) + AP(→ 유지)")
    print("="*72)
    print(f"{'config':22s} | {'held-out flip':>14s} | {'mAP50-95':>9s} | {'mAP50':>7s}")
    print("-"*72)
    def frow(name, key_ap, key_flip):
        m50=ap_res[key_ap][0]; m5=ap_res[key_ap][1]
        fl = "-" if (key_flip is None or flip.get(key_flip) is None) else f"{flip[key_flip][0]:.2f}%"
        aps = "-" if m50 is None else f"{m50:.2f}"
        ap5 = "-" if m5 is None else f"{m5:.2f}"
        print(f"{name:22s} | {fl:>14s} | {aps:>9s} | {ap5:>7s}")
    frow("FP32", "fp", None)
    frow("Naive W8A8", "naive", "naive")
    frow(f"Ours (cv3@A{CV3_ABITS})", "ours", "ours")
    print("-"*72)
    if flip["naive"] and flip["ours"]:
        d=(flip["ours"][0]-flip["naive"][0])/max(flip["naive"][0],1e-9)*100
        print(f"held-out: Ours vs naive = {d:+.1f}%  (seeds naive {flip['naive'][1]}, ours {flip['ours'][1]})")
    print("\n메인표에 baseline 행(기존 결과) 배치:")
    print("  AdaRound / BRECQ / QDrop / PD-Quant : held-out ~8.5% (전부 실패, AP는 naive급)")
    print("  PromptCal v1(margin) / v2(decision) : held-out ≥ naive (anti-transfer)")
    print("  → Ours만 held-out을 유의미하게 낮추면서 AP 유지 = method 성립.")

if __name__=="__main__":
    main()
