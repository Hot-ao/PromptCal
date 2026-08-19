"""
D-1 드라이버: AdaRound 학습 후 AP와 semantic 지표를 측정, naive와 비교.

흐름: fuse → QuantConv 래핑 → activation 보정 → AdaRound 변환 → layer-wise 최적화
      → (a) confident flip + substitution 측정  (b) 원하면 AP val.

핵심 확인: AdaRound가 naive 대비
  - AP는 개선되는가 (reconstruction 성공)
  - flip/substitution은 여전히 높은가 (decision은 못 지킴)  ← 논문 예측

naive 기준선(기억용): AP 37.3, confident flip ~0.7%, substitution 이웃 69%.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/08_adaround.py \
        --coco-root /data/taeho/coco_datasets --calib 32 --iters 2000 \
        --eval 500 --with-ap --data configs/coco_local.yaml --device 0
"""

import argparse, glob, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness, paired_run
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


def neighbors(txt, k):
    txt=txt/txt.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    sim=txt@txt.T
    return [set([o for o in torch.argsort(sim[c],descending=True).tolist() if o!=c][:k])
            for c in range(sim.shape[0])]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--calib", type=int, default=32, help="AdaRound calib 이미지(보수적으로)")
    ap.add_argument("--iters", type=int, default=2000, help="layer당 최적화 스텝")
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--with-ap", action="store_true", help="val로 AP도 측정")
    ap.add_argument("--device", default="0")
    args=ap.parse_args()

    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names(); nc=len(names)

    from ultralytics import YOLOWorld
    print("[load] FP32 + quant models")
    model_fp=YOLOWorld(args.model); model_fp.set_classes(names)
    model_q=YOLOWorld(args.model); model_q.set_classes(names)
    model_fp.fuse(); model_q.fuse()
    wrap_convs(model_q.model, w_bits=8, a_bits=8)
    model_fp.model.to(device).eval(); model_q.model.to(device).eval()

    all_imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib_imgs=all_imgs[:args.calib]
    eval_imgs=all_imgs[args.calib:args.calib+args.eval]

    print(f"[calib] activation 보정 {len(calib_imgs)}장...")
    calib_tensors=[preprocess(p,args.imgsz,device) for p in calib_imgs]
    calibrate(model_q.model, calib_tensors, device=device)

    print(f"[adaround] 변환 + 최적화 (iters={args.iters}/layer)...")
    n=convert_to_adaround(model_q.model)
    print(f"[adaround] {n} conv 변환됨")
    t0=time.time()
    optimize_adaround(model_q.model, model_fp.model, calib_tensors, device,
                      iters=args.iters, lr=1e-2, reg_weight=1e-3, verbose=True)
    print(f"[adaround] 최적화 {time.time()-t0:.0f}s 소요")

    # ---- semantic 측정 (flip + substitution) ----
    txt=model_fp.model.txt_feats.detach().float()
    if txt.dim()==3: txt=txt[0]
    neigh=neighbors(txt,args.k)

    h_fp=SimilarityHarness(model_fp.model,device=device)
    h_q=SimilarityHarness(model_q.model,device=device)
    n_conf=n_flip=to_neigh=0
    print(f"[measure] semantic {len(eval_imgs)}장...")
    for i,p in enumerate(eval_imgs):
        t=preprocess(p,args.imgsz,device)
        rec_fp,rec_q=paired_run(h_fp,h_q,t,image_id=i)
        prob=rec_fp.sim.sigmoid(); maxp,c_fp=prob.max(-1); conf=maxp>args.conf_thres
        c_q=rec_q.sim.argmax(-1)
        idx=conf.nonzero(as_tuple=True)[0]
        n_conf+=int(conf.sum())
        for j in idx.tolist():
            a=int(c_fp[j]); b=int(c_q[j])
            if b!=a:
                n_flip+=1
                if b in neigh[a]: to_neigh+=1
        if (i+1)%100==0: print(f"   [{i+1}/{len(eval_imgs)}] flips={n_flip}")
    h_fp.close(); h_q.close()

    flip_rate=n_flip/max(n_conf,1)*100
    neigh_rate=to_neigh/max(n_flip,1)*100

    ap_str="(생략)"
    if args.with_ap:
        print("[val] AdaRound AP 측정...")
        m=model_q.val(data=args.data, imgsz=args.imgsz, device=args.device,
                      save_json=True, verbose=False)
        ap_str=f"{float(m.box.map)*100:.2f} (ultralytics), pycocotools는 로그 참고"

    print("\n"+"="*58)
    print(" ADAROUND (D-1) vs NAIVE")
    print("="*58)
    print(f" {'지표':>22} | {'naive':>10} | {'AdaRound':>10}")
    print("-"*58)
    print(f" {'AP (mAP50-95)':>22} | {'37.3':>10} | {ap_str if args.with_ap else 'N/A':>10}")
    print(f" {'confident flip':>22} | {'~0.7%':>10} | {flip_rate:>9.2f}%")
    print(f" {'flip→이웃(top-'+str(args.k)+')':>22} | {'69%':>10} | {neigh_rate:>9.1f}%")
    print("="*58)
    print("\n예측 확인:")
    print(" - AP가 naive(37.3)보다 오르면 → reconstruction 개선 성공.")
    print(" - flip/이웃이 naive와 비슷하게 높으면 → decision은 여전히 못 지킴.")
    print("   = 'reconstruction ≠ decision preservation' 강한 baseline에서도 성립.")


if __name__ == "__main__":
    main()
