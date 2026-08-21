"""
E 두 번째 관문(v2): decision 보존 + activation scale 최적화가 baseline을 넘는가.

v1(20_*)과의 차이:
  - 최적화 대상: alpha(weight rounding) → activation scale(m). '안 굳는' 문제 없음.
  - 목적함수: margin(값 MSE) → decision CE(held-out fallback argmax 직접 보존).

prompt 3분할은 20_*와 동일: S(seen) + H_cal(calib held-out) + H_eval(eval held-out).
  학습:  removed=H_cal, keep=S  (H_eval은 loss에 절대 안 들어감)
  측정:  H_eval에서만 heldout_flip  (20_*와 동일 로직 재사용)

성공 판정: PromptCal-v2의 H_eval flip < naive & < AdaRound, 여러 seed 일관.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/21_promptcal_decision.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 1000 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.promptcal_v2 import convert_to_learnable_scale, optimize_promptcal_v2


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

def build(model_cls, w, names, device, calib, mode, fp=None, iters=1000,
          removed=None, keep=None, lr=1e-2, temp=1.0, gate=0.0):
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model,8,8); m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode=="adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=iters, verbose=False)
    elif mode=="promptcal_v2":
        # AdaRound 없이 calibration 직후 바로 scale 최적화. weight는 round-to-nearest 고정.
        convert_to_learnable_scale(m.model)
        optimize_promptcal_v2(m.model, fp.model, calib, device, removed, keep,
                              iters=iters, lr=lr, temp=temp, margin_gate=gate, verbose=True)
    return m

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    """20_*와 동일. H_eval을 마스킹하고, FP full top-1이 H_eval인 confident anchor에서
    K-only(H_eval 제거) argmax의 FP vs quant 불일치율."""
    Hm = torch.zeros(80, dtype=torch.bool); Hm[H_eval] = True
    tot = fl = 0
    for i, t in enumerate(probe):
        sf = h_fp.run_image(t, i).sim; sq = h_q.run_image(t, i).sim
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1)
        conf_m = mp > conf
        target = conf_m & Hm[c_fp]
        if target.sum() == 0:
            continue
        idx = target.nonzero(as_tuple=True)[0]
        fp_K = sf.clone(); fp_K[:, Hm] = -1e9
        q_K = sq.clone();  q_K[:, Hm] = -1e9
        fl += int((fp_K.argmax(-1)[idx] != q_K.argmax(-1)[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=1000)
    ap.add_argument("--iters",type=int,default=150, help="full-batch 최적화 스텝(각 스텝=calib 1회 순회)")
    ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--temp",type=float,default=1.0)
    ap.add_argument("--gate",type=float,default=0.0, help="FP fallback margin gate(0=off)")
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--no-adaround",action="store_true", help="AdaRound baseline 건너뛰기")
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--device",default="0")
    args=ap.parse_args()
    device=f"cuda:{args.device}" if args.device!="cpu" else "cpu"
    names=load_coco_names()

    from ultralytics import YOLOWorld
    imgs=sorted(glob.glob(os.path.join(args.coco_root,"val2017","*.jpg")))
    calib=[preprocess(p,args.imgsz,device) for p in imgs[:args.calib]]
    probe=[preprocess(p,args.imgsz,device) for p in imgs[args.calib:args.calib+args.eval]]

    print("[load] FP"); fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    print("[build] naive"); nv=build(YOLOWorld,args.model,names,device,calib,"naive")
    h_fp=SimilarityHarness(fp.model,device=device)
    h_nv=SimilarityHarness(nv.model,device=device)

    ad=None
    if not args.no_adaround:
        print("[build] AdaRound (baseline)")
        ad=build(YOLOWorld,args.model,names,device,calib,"adaround",fp=fp,iters=args.iters)
        h_ad=SimilarityHarness(ad.model,device=device)

    print("\n"+"="*74)
    print(" E v2: decision 보존 + scale 최적화 vs baseline")
    print("="*74)
    hdr = f"{'seed':>5} | {'naive':>7} |"
    if ad is not None: hdr += f" {'AdaRound':>8} |"
    hdr += f" {'PC-v2':>7} | {'vs naive':>8}"
    print(hdr); print("-"*74)

    nv_l, ad_l, pc_l = [], [], []
    for s in args.seeds:
        rng=np.random.default_rng(s); perm=rng.permutation(80)
        S=perm[:40].tolist(); H_cal=perm[40:60].tolist(); H_eval=perm[60:80].tolist()
        # 학습: removed=H_cal(제거할 held-out), keep=S(겨루는 남은 집합). H_eval 미사용.
        pc=build(YOLOWorld,args.model,names,device,calib,"promptcal_v2",fp=fp,
                 iters=args.iters,removed=H_cal,keep=S,lr=args.lr,temp=args.temp,gate=args.gate)
        h_pc=SimilarityHarness(pc.model,device=device)
        fnv=heldout_flip(h_fp,h_nv,probe,H_eval)
        fpc=heldout_flip(h_fp,h_pc,probe,H_eval)
        row=f"{s:>5} | {fnv:>6.2f}% |"
        if ad is not None:
            fad=heldout_flip(h_fp,h_ad,probe,H_eval); ad_l.append(fad)
            row+=f" {fad:>7.2f}% |"
        vs=(fnv-fpc)/max(fnv,1e-9)*100
        row+=f" {fpc:>6.2f}% | {vs:>7.1f}%"
        print(row)
        nv_l.append(fnv); pc_l.append(fpc)
        h_pc.close()

    h_fp.close(); h_nv.close()
    if ad is not None: h_ad.close()
    print("-"*74)
    mrow=f"{'mean':>5} | {np.mean(nv_l):>6.2f}% |"
    if ad_l: mrow+=f" {np.mean(ad_l):>7.2f}% |"
    mrow+=f" {np.mean(pc_l):>6.2f}% | {(np.mean(nv_l)-np.mean(pc_l))/max(np.mean(nv_l),1e-9)*100:>7.1f}%"
    print(mrow); print("="*74)
    print("\n판정:")
    print(" - PC-v2 H_eval flip < naive & < AdaRound 이고 seed 일관 → v2 방향 성공.")
    print(" - naive와 비슷/높음 → scale+decision으로도 전이 실패. 설계 문서의 대안(§9)로.")

if __name__=="__main__":
    main()
