"""
PromptCal 디버그 전용 (빠름 ~30초). alpha·margin·유사도 변화를 관찰.

무거운 것 다 제거: calib 6장, eval 없음(선택), iters 150, PromptCal만 빌드.
매 N iter마다 alpha 통계·margin_loss·h 수렴률·유사도 변화를 찍음.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/dbg_promptcal.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 6 --iters 150 --device 0
"""
import argparse, glob, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, list_adaround_convs, h_alpha
from src.quant.pdquant import _find_head, _CV4Capture
from src.quant.promptcal import margin_loss


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

def sim_from_cap(cap):
    parts=[]
    for i in sorted(cap.buf):
        B,P,H,W=cap.buf[i].shape
        parts.append(cap.buf[i].reshape(B,P,H*W))
    return torch.cat(parts,dim=2)[0].transpose(0,1)   # [A,P]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=6)
    ap.add_argument("--iters",type=int,default=150)
    ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--reg-weight",type=float,default=0.1)
    ap.add_argument("--warmup",type=float,default=0.4)
    ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--watch",type=int,nargs="+",default=[0,35,70],help="관찰할 alpha layer 인덱스")
    ap.add_argument("--mode", default="alpha", choices=["alpha","scale"],
                    help="alpha=rounding 최적화(기존), scale=activation scale 최적화(방향 C)")
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]

    t0=time.time()
    fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    q=YOLOWorld(args.model); q.set_classes(names); q.fuse()
    wrap_convs(q.model,8,8); q.model.to(device).eval()
    calibrate(q.model, calib, device=device)
    convert_to_adaround(q.model)
    ada=list_adaround_convs(q.model)
    if args.mode=="scale":
        # 방향 C: rounding 고정, activation scale(s_mult) 최적화
        from src.quant.promptcal import optimize_promptcal_scale
        print(f"[dbg] mode=SCALE (방향 C), alpha layers={len(ada)}")
        optimize_promptcal_scale(q.model, fp.model, calib, device,
                                 list(range(60)), iters=args.iters, lr=args.lr,
                                 k=args.k, verbose=True)
        # scale 최적화 후 dbg 종료 (아래 alpha 루프는 스킵)
        print(f"\n[dbg] 총 {time.time()-t0:.1f}s")
        print("해석: margin_loss가 매끄럽게 줄고 s_mult가 움직이면 방향 C 정상.")
        return
    for ac in ada: ac.soft=True; ac.ste=True
    print(f"[dbg] setup {time.time()-t0:.1f}s, alpha layers={len(ada)}, watch={args.watch}")

    # FP 타깃 캐시
    fp_head=_find_head(fp.model); fp_cap=_CV4Capture(fp_head)
    fp_sims=[]
    with torch.no_grad():
        for t in calib:
            fp_cap.clear(); fp.model(t); fp_sims.append(sim_from_cap(fp_cap).detach())
    fp_cap.close()

    q_head=_find_head(q.model); q_cap=_CV4Capture(q_head)
    alphas=[ac.alpha for ac in ada]
    a0=[a.detach().clone() for a in alphas]
    opt=torch.optim.Adam(alphas, lr=args.lr)
    pidx=torch.arange(60,device=device)  # 디버그: 앞 60개 프롬프트

    warmup=int(args.warmup*args.iters)
    print(f"\n{'it':>4} | {'margin':>7} | {'h→01':>5} | "
          + " | ".join(f"a[{w}]Δ" for w in args.watch) + " | grad")
    print("-"*70)
    n=len(calib)
    for it in range(args.iters):
        j=it%n; t=calib[j]; sim_fp=fp_sims[j]
        prob=sim_fp.sigmoid(); mp,_=prob.max(-1); conf=mp>0.25
        aidx=conf.nonzero(as_tuple=True)[0]
        q_cap.clear(); opt.zero_grad()
        q.model(t)
        sim_q=sim_from_cap(q_cap)
        ml=margin_loss(sim_q[aidx][:,pidx], sim_fp[aidx][:,pidx], k=args.k)
        if it<warmup: beta=max(2.0,20.0*(1-it/warmup)); rw=10.0
        else: beta=2.0; rw=args.reg_weight
        reg=sum(ac.reg_loss(beta,reduction="mean") for ac in ada)/len(ada)
        loss=ml+rw*reg
        loss.backward()
        gnorm=torch.nn.utils.clip_grad_norm_(alphas,max_norm=1.0)
        opt.step()

        if it%max(1,args.iters//15)==0 or it==args.iters-1:
            hc=sum(float(((h_alpha(a.alpha)<0.05)|(h_alpha(a.alpha)>0.95)).float().mean())
                   for a in ada)/len(ada)*100
            deltas=[]
            for w in args.watch:
                d=float((alphas[w].detach()-a0[w]).abs().mean())
                deltas.append(f"{d:>6.3f}")
            print(f"{it:>4} | {float(ml.detach()):>7.4f} | {hc:>4.0f}% | "
                  + " | ".join(deltas) + f" | {float(gnorm):.3f}")
    q_cap.close()

    print("-"*70)
    # 최종 alpha 분포 (watch 레이어)
    for w in args.watch:
        a=h_alpha(alphas[w].detach())
        print(f"[layer {w}] h 분포: <0.05={float((a<0.05).float().mean())*100:.0f}% "
              f">0.95={float((a>0.95).float().mean())*100:.0f}% "
              f"중간={float(((a>=0.05)&(a<=0.95)).float().mean())*100:.0f}%")
    print(f"\n[dbg] 총 {time.time()-t0:.1f}s")
    print("해석: h→01이 안 오르면 rounding 정체(의심 D). margin 주는데 h 정체면 목적함수 문제.")

if __name__=="__main__":
    main()
