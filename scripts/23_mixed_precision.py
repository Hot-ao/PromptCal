"""
E method 확정: 손상 lever(cv3)를 격리하고, FP가 아닌 '높은 bit'로 두는 실제 mixed-precision.

22_precision_probe에서 head-FP가 held-out을 -55% 낮췄고(cv4는 conv 없음, neck 무효),
남은 후보는 cv3(contrastive embedding conv, cv4에 embedding을 먹이는 branch)다.
이 스크립트는:
  (A) 격리: cv3-only FP vs cv2-only FP → 정말 cv3가 lever인지, cv2는 무효인지 확정.
  (B) method: cv3를 FP가 아니라 A16/A12/A10로 두고 held-out 회복이 유지되는지(효율↔정밀도).
       Reg-PTQ의 'head-FP는 비용 큼' 우려를, 최소 branch(cv3)만 + 낮은 여유 bit로 대응.

측정: heldout_flip(H_eval), 3 seed. 학습 없음(빠름).

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/23_mixed_precision.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 300 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
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

def n_quant_convs(mod):
    return sum(1 for m in mod.modules() if isinstance(m, QuantConv2d))

def branch_convs(det_model, branch):
    """det_model(=m.model)의 head.<branch> 아래 QuantConv2d 목록."""
    head = _find_head(det_model)
    br = getattr(head, branch, None)
    if br is None:
        return []
    return [m for m in br.modules() if isinstance(m, QuantConv2d)]

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
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

def build(model_cls, w, names, device, calib, skip=None, bit_override=None):
    """
    skip         : wrap 단계에서 FP로 둘 서브트리 이름 집합.
    bit_override : {branch: (w_bits, a_bits)} — wrap(8/8) 후 해당 branch conv의 bit를 상향.
                   calibrate 전에 적용하므로 freeze가 그 bit로 scale을 잡는다.
    """
    m=model_cls(w); m.set_classes(names); m.fuse()
    wrap_convs(m.model, 8, 8, skip_names=skip or set())
    if bit_override:
        for br, (wb, ab) in bit_override.items():
            for c in branch_convs(m.model, br):
                if wb is not None: c.w_bits = wb
                if ab is not None: c.a_obs.bits = ab
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=300)
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
    h_fp=SimilarityHarness(fp.model,device=device)

    # config: (라벨, skip, bit_override, 설명)
    configs=[
        ("naive (all W8A8)",   None, None),
        # (A) 격리 — cv3가 lever인가, cv2는 무효인가
        ("cv3-FP",             {"cv3"}, None),
        ("cv2-FP",             {"cv2"}, None),
        # (B) method — cv3를 FP 아닌 높은 bit로
        ("cv3 @ W8A16",        None, {"cv3": (8, 16)}),
        ("cv3 @ W8A12",        None, {"cv3": (8, 12)}),
        ("cv3 @ W8A10",        None, {"cv3": (8, 10)}),
        ("cv3 @ W16A16",       None, {"cv3": (16, 16)}),
    ]

    built=[]
    for label, skip, bo in configs:
        m=build(YOLOWorld,args.model,names,device,calib,skip=skip,bit_override=bo)
        h=SimilarityHarness(m.model,device=device)
        nq=n_quant_convs(m.model)
        ncv3=len(branch_convs(m.model,"cv3"))
        built.append((label,h,nq,ncv3))
        print(f"[build] {label:16s} quantConv={nq} (cv3 convs={ncv3})")

    print("\n"+"="*80)
    print(" cv3 격리 + bit sweep vs held-out flip (낮을수록 손상 적음)")
    print("="*80)
    hdr=f"{'config':16s} |"
    for s in args.seeds: hdr+=f" seed{s:>2d} |"
    hdr+=f" {'mean':>6s} | {'Δ vs naive':>10s}"
    print(hdr); print("-"*80)

    naive_mean=None
    for label,h,nq,ncv3 in built:
        vals=[]
        for s in args.seeds:
            rng=np.random.default_rng(s); perm=rng.permutation(80)
            H_eval=perm[60:80].tolist()
            vals.append(heldout_flip(h_fp,h,probe,H_eval))
        mean=float(np.mean(vals))
        if naive_mean is None: naive_mean=mean
        delta=(mean-naive_mean)/max(naive_mean,1e-9)*100
        row=f"{label:16s} |"
        for v in vals: row+=f" {v:>5.2f}% |"
        row+=f" {mean:>5.2f}% | {delta:>+9.1f}%"
        print(row)
    for _,h,_,_ in built: h.close()
    h_fp.close()
    print("-"*80)
    print("\n판독:")
    print(" (A) 격리: cv3-FP ≈ head-FP(-55%)이고 cv2-FP ≈ naive면 → lever는 cv3 단독으로 확정.")
    print(" (B) method: cv3 @ W8A16/A12/A10에서 회복이 어디까지 유지되나 → 최소 여유 bit 결정.")
    print("     예) cv3 @ A10에서도 held-out이 크게 낮으면, 효율 손해 거의 없이 손상 복구.")
    print(" 다음: 이 표 + baseline(AdaRound/BRECQ/QDrop/PD-Quant/v1/v2) 표로 method 대비 완성.")

if __name__=="__main__":
    main()
