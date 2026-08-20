"""
E 첫 관문: held-out margin 보존이 baseline을 넘는가 (최소 실험).

prompt 3분할: S(seen) + H_cal(calib held-out) + H_eval(eval held-out).
비교: naive / AdaRound(baseline) / PromptCal(held-out margin 보존).
측정: H_eval에서만 held-out flip.

성공 판정: PromptCal의 H_eval flip < AdaRound flip, 여러 seed 일관.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/20_promptcal_minimal.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 1000 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.promptcal import optimize_promptcal


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

def build(model_cls, w, names, device, calib, mode, fp=None, iters=1000, pidx=None):
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model,8,8); m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode=="adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=iters, verbose=False)
    elif mode=="promptcal":
        # AdaRound 초기화 없이 calibration 직후 바로 margin 최적화(독립적 접근).
        # AdaRound refine으로 짜면 AdaRound에 갇혀 항상 동일해짐 → 초기화 제거.
        convert_to_adaround(m.model)
        optimize_promptcal(m.model, fp.model, calib, device, pidx, iters=1500,
                           lr=3e-3, reg_weight=0.1, k=5, verbose=True)
    return m

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    """05와 동일한 held-out 측정. H_eval을 마스킹하고, target(=FP full top-1이 H_eval에
    속하는 confident anchor)에서 K-only(H_eval 제거) argmax의 FP vs quant 불일치."""
    Hm = torch.zeros(80, dtype=torch.bool); Hm[H_eval] = True
    tot = fl = 0
    for i, t in enumerate(probe):
        sf = h_fp.run_image(t, i).sim; sq = h_q.run_image(t, i).sim
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1)
        conf_m = mp > conf
        target = conf_m & Hm[c_fp]          # 원래 H_eval 클래스로 판정된 confident region
        if target.sum() == 0:
            continue
        idx = target.nonzero(as_tuple=True)[0]
        fp_K = sf.clone(); fp_K[:, Hm] = -1e9   # H_eval 제거 후 남는 것 중 top-1
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
    ap.add_argument("--iters",type=int,default=2000)
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

    print("[load] FP"); fp=YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    print("[build] naive"); nv=build(YOLOWorld,args.model,names,device,calib,"naive")
    print("[build] AdaRound (baseline)"); ad=build(YOLOWorld,args.model,names,device,calib,"adaround",fp=fp,iters=args.iters)

    h_fp=SimilarityHarness(fp.model,device=device)
    h_nv=SimilarityHarness(nv.model,device=device)
    h_ad=SimilarityHarness(ad.model,device=device)

    print("\n"+"="*70)
    print(" E 최소 실험: held-out margin 보존 vs baseline")
    print("="*70)
    print(f"{'seed':>5} | {'naive':>7} | {'AdaRound':>8} | {'PromptCal':>9} | {'vs Ada':>7}")
    print("-"*70)
    nv_l,ad_l,pc_l=[],[],[]
    for s in args.seeds:
        rng=np.random.default_rng(s); perm=rng.permutation(80)
        S=perm[:40].tolist(); H_cal=perm[40:60].tolist(); H_eval=perm[60:80].tolist()
        pidx=S+H_cal   # calibration에 쓰는 프롬프트(S+H_cal), H_eval 제외
        # PromptCal 빌드 (seed마다 prompt 분할이 달라 재빌드)
        pc=build(YOLOWorld,args.model,names,device,calib,"promptcal",fp=fp,iters=args.iters,pidx=pidx)
        h_pc=SimilarityHarness(pc.model,device=device)
        fnv=heldout_flip(h_fp,h_nv,probe,H_eval)
        fad=heldout_flip(h_fp,h_ad,probe,H_eval)
        fpc=heldout_flip(h_fp,h_pc,probe,H_eval)
        h_pc.close()
        nv_l.append(fnv); ad_l.append(fad); pc_l.append(fpc)
        diff=(fad-fpc)/max(fad,1e-9)*100
        print(f"{s:>5} | {fnv:>6.2f}% | {fad:>7.2f}% | {fpc:>8.2f}% | {diff:>6.1f}%")
    h_fp.close(); h_nv.close(); h_ad.close()
    print("-"*70)
    print(f"{'mean':>5} | {np.mean(nv_l):>6.2f}% | {np.mean(ad_l):>7.2f}% | "
          f"{np.mean(pc_l):>8.2f}% | {(np.mean(ad_l)-np.mean(pc_l))/max(np.mean(ad_l),1e-9)*100:>6.1f}%")
    print("="*70)
    print("\n판정:")
    print(" - PromptCal H_eval flip < AdaRound이고 seed 일관 → 방향 성공(baseline 넘음).")
    print(" - AdaRound와 비슷/높음 → 09_phase1처럼 held-out 전이 실패. 재설계 필요.")

if __name__=="__main__":
    main()
