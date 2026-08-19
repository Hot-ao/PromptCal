"""
D-1 결정 실험: held-out 조건에서 naive vs AdaRound 비교.

질문: AdaRound(강한 reconstruction)가 seen에선 flip을 줄였는데(0.73→0.40),
held-out(안 본 프롬프트)에서도 줄이는가?
  - AdaRound held-out flip ≈ naive → 논지 강화("reconstruction은 held-out 못 지킴")
  - AdaRound held-out flip << naive → 논지 재검토

FP + naive-quant + AdaRound-quant 세 모델을 띄우고, 같은 held-out 분할(FP pseudo-GT,
H열 마스킹)에서 두 양자화의 seen/held-out flip을 나란히 측정.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/09_heldout_adaround.py \
        --coco-root /data/taeho/coco_datasets --calib 32 --iters 2000 \
        --eval 1000 --seeds 0 1 2 --device 0
"""

import argparse, glob, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround


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


def build_quant(model_cls, weights, names, device, calib_tensors, adaround=False,
                fp_module=None, iters=2000):
    m = model_cls(weights); m.set_classes(names); m.fuse()
    wrap_convs(m.model, w_bits=8, a_bits=8)
    m.model.to(device).eval()
    calibrate(m.model, calib_tensors, device=device)
    if adaround:
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp_module, calib_tensors, device, iters=iters, verbose=True)
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args=ap.parse_args()

    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names(); nc=len(names)

    from ultralytics import YOLOWorld
    all_imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib_tensors=[preprocess(p,args.imgsz,device) for p in all_imgs[:args.calib]]
    eval_imgs=all_imgs[args.calib:args.calib+args.eval]

    print("[load] FP model")
    model_fp=YOLOWorld(args.model); model_fp.set_classes(names); model_fp.fuse()
    model_fp.model.to(device).eval()

    print("[build] naive-quant")
    model_naive=build_quant(YOLOWorld,args.model,names,device,calib_tensors,adaround=False)

    print("[build] AdaRound-quant (최적화 ~12min)")
    t0=time.time()
    model_ada=build_quant(YOLOWorld,args.model,names,device,calib_tensors,adaround=True,
                          fp_module=model_fp.model, iters=args.iters)
    print(f"[build] AdaRound 완료 {time.time()-t0:.0f}s")

    # held-out 마스크
    H_masks={}
    for s in args.seeds:
        H=np.random.default_rng(s).permutation(nc)[:args.heldout]
        m=torch.zeros(nc,dtype=torch.bool); m[H.tolist()]=True; H_masks[s]=m

    h_fp=SimilarityHarness(model_fp.model,device=device)
    h_nv=SimilarityHarness(model_naive.model,device=device)
    h_ad=SimilarityHarness(model_ada.model,device=device)

    acc={s:dict(n=0,sn=0,sa=0,hn=0,ha=0) for s in args.seeds}
    print(f"[eval] {len(eval_imgs)}장 x {len(args.seeds)} seeds...")
    for i,p in enumerate(eval_imgs):
        t=preprocess(p,args.imgsz,device)
        sim_fp=h_fp.run_image(t,i).sim
        sim_nv=h_nv.run_image(t,i).sim
        sim_ad=h_ad.run_image(t,i).sim
        prob=sim_fp.sigmoid(); maxp,c_fp=prob.max(-1); conf=maxp>args.conf_thres
        nv_full=sim_nv.argmax(-1); ad_full=sim_ad.argmax(-1)
        for s in args.seeds:
            Hm=H_masks[s]; target=conf&Hm[c_fp]
            if target.sum()==0: continue
            idx=target.nonzero(as_tuple=True)[0]
            a=acc[s]; a["n"]+=len(idx)
            a["sn"]+=int((nv_full[idx]!=c_fp[idx]).sum())
            a["sa"]+=int((ad_full[idx]!=c_fp[idx]).sum())
            fpK=sim_fp.clone(); fpK[:,Hm]=-1e9
            nvK=sim_nv.clone(); nvK[:,Hm]=-1e9
            adK=sim_ad.clone(); adK[:,Hm]=-1e9
            a["hn"]+=int((fpK.argmax(-1)[idx]!=nvK.argmax(-1)[idx]).sum())
            a["ha"]+=int((fpK.argmax(-1)[idx]!=adK.argmax(-1)[idx]).sum())
        if (i+1)%200==0: print(f"   [{i+1}/{len(eval_imgs)}]")
    h_fp.close(); h_nv.close(); h_ad.close()

    print("\n"+"="*72)
    print(" HELD-OUT: naive vs AdaRound")
    print("="*72)
    print(f"{'seed':>5} | {'target':>7} | {'seen_nv':>7} | {'seen_ad':>7} | "
          f"{'ho_nv':>6} | {'ho_ad':>6} | {'ho 감소':>7}")
    print("-"*72)
    hn_l,ha_l=[],[]
    for s in args.seeds:
        a=acc[s]; n=max(a["n"],1)
        sn,sa=a["sn"]/n*100,a["sa"]/n*100
        hn,ha=a["hn"]/n*100,a["ha"]/n*100
        hn_l.append(hn); ha_l.append(ha)
        red=(hn-ha)/max(hn,1e-9)*100
        print(f"{s:>5} | {a['n']:>7} | {sn:>6.2f}% | {sa:>6.2f}% | "
              f"{hn:>5.2f}% | {ha:>5.2f}% | {red:>6.1f}%")
    print("-"*72)
    print(f"{'mean':>5} | {'':>7} | {'':>7} | {'':>7} | "
          f"{np.mean(hn_l):>5.2f}% | {np.mean(ha_l):>5.2f}% | "
          f"{(np.mean(hn_l)-np.mean(ha_l))/max(np.mean(hn_l),1e-9)*100:>6.1f}%")
    print("="*72)
    print("\n해석:")
    print(" - 'ho 감소'가 작으면(AdaRound held-out flip ≈ naive) → reconstruction이")
    print("   held-out decision을 못 지킴 = 논지 강화.")
    print(" - seen에선 AdaRound가 flip을 줄여도 held-out에선 못 줄이는 대비가 핵심.")


if __name__ == "__main__":
    main()
