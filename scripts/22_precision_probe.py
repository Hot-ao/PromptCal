"""
E 방향 전환 진단(v2 anti-transfer 이후): 손상이 '어디서' 결정을 뒤집는가를 특정.

가설: calibration-vocabulary fitting(v1/v2)은 held-out으로 anti-transfer한다. 그렇다면
방법은 vocabulary를 fit하지 말고, 결정이 내려지는 지점의 '정밀도'를 prompt-agnostic하게
지켜야 한다. 이 스크립트는 그 지점을 특정한다.

방법: 양자화에서 특정 서브트리를 '빼서'(FP 유지) held-out flip이 얼마나 회복되는지 사다리로 측정.
  P0 naive            : 전부 W8A8 (기준 손상)
  P1 cv4-FP           : ContrastiveHead(cv4)만 FP → head 정렬 projection 보호
  P2 head-FP          : detect head 전체(cv2/cv3/cv4) FP
  P3 head+last1       : head + 직전 블록(neck 말단) FP
  P4 head+last2       : head + 직전 2블록 FP
  (--extra K로 head+lastK까지 확장)

읽는 법:
  - P1/P2에서 held-out flip이 크게 떨어지면 → 손상은 head(결정 지점)에 몰림.
    = cv4/head를 높은 bit로 두는 mixed-precision이 곧 방법. (FP는 상한, 실제론 A16 등으로)
  - P1/P2로 안 떨어지고 P3/P4에서 떨어지면 → 손상은 상류(neck) region feature에.
    = 00d 민감도로 상류 타깃 정밀도 상향.
  - 어디서도 안 떨어지면 → 손상이 광범위 분산. mixed-precision 단독으론 부족(재검토).

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/22_precision_probe.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 300 --seeds 0 1 2 --extra 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.fake_quant import QuantConv2d


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

def n_quant_convs(model_module):
    return sum(1 for m in model_module.modules() if isinstance(m, QuantConv2d))

def build_skip(model_cls, w, names, device, calib, skip_names):
    """skip_names 서브트리는 FP로 두고 나머지 W8A8. 교체/스킵 개수 리포트."""
    m=model_cls(w); m.set_classes(names); m.fuse()
    n_wrapped=wrap_convs(m.model, 8, 8, skip_names=skip_names)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    return m, n_wrapped

def heldout_flip(h_fp, h_q, probe, H_eval, conf=0.25):
    """20_*/21_*와 동일 로직."""
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

def head_index(model):
    """m.model.model(Sequential)에서 detect head의 인덱스(=마지막) 반환."""
    seq = getattr(model.model, "model", None)
    if seq is None or not hasattr(seq, "__len__"):
        return None
    return len(seq) - 1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="yolov8s-world.pt")
    ap.add_argument("--coco-root",default="/data/taeho/coco_datasets")
    ap.add_argument("--calib",type=int,default=32)
    ap.add_argument("--eval",type=int,default=300)
    ap.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--extra",type=int,default=2, help="head+lastK 사다리 확장 K")
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

    hidx=head_index(fp)
    if hidx is None:
        print("[warn] head 인덱스 자동탐지 실패 — cv-name 기반만 사용")

    # 사다리 config: (라벨, skip_names)
    configs=[("P0 naive", set()),
             ("P1 cv4-FP", {"cv4"}),
             ("P2 head-FP", {"cv2","cv3","cv4"})]
    if hidx is not None:
        for k in range(1, args.extra+1):
            skip={"cv2","cv3","cv4"} | {str(hidx-j) for j in range(0, k+1)}
            configs.append((f"P{2+k} head+last{k}", skip))

    # 각 config 모델을 1회 빌드(프롬프트 무관) → 여러 seed의 H_eval로 측정
    built=[]
    for label, skip in configs:
        m, nw = build_skip(YOLOWorld,args.model,names,device,calib,skip)
        h = SimilarityHarness(m.model,device=device)
        nq = n_quant_convs(m.model)
        built.append((label, skip, h, nw, nq))
        print(f"[build] {label:16s} wrapped(quant)={nq}  skip={sorted(skip)}")

    print("\n"+"="*78)
    print(" 손상 위치 진단: skip 사다리 vs held-out flip (낮을수록 손상 적음)")
    print("="*78)
    # 헤더
    hdr=f"{'config':16s} | {'quantConv':>9s} |"
    for s in args.seeds: hdr+=f" seed{s:>2d} |"
    hdr+=f" {'mean':>6s} | {'Δ vs naive':>10s}"
    print(hdr); print("-"*78)

    naive_mean=None
    for label, skip, h, nw, nq in built:
        vals=[]
        for s in args.seeds:
            rng=np.random.default_rng(s); perm=rng.permutation(80)
            H_eval=perm[60:80].tolist()
            vals.append(heldout_flip(h_fp,h,probe,H_eval))
        mean=float(np.mean(vals))
        if naive_mean is None: naive_mean=mean
        delta=(mean-naive_mean)/max(naive_mean,1e-9)*100
        row=f"{label:16s} | {nq:>9d} |"
        for v in vals: row+=f" {v:>5.2f}% |"
        row+=f" {mean:>5.2f}% | {delta:>+9.1f}%"
        print(row)
    for _,_,h,_,_ in built: h.close()
    h_fp.close()
    print("-"*78)
    print("\n판독:")
    print(" - P1/P2에서 held-out이 크게 ↓ → 손상은 head(결정 지점). cv4/head mixed-precision이 방법.")
    print(" - P1/P2 미미, P3/P4에서 ↓ → 손상은 상류(neck). 00d 민감도로 타깃.")
    print(" - 어디서도 안 ↓ → 손상 광범위 분산. mixed-precision 단독 부족 → 재설계.")
    print(" (주의: skip=FP는 '상한'. 실제 방법은 해당 지점을 A16 등 높은 bit로 두는 것.)")

if __name__=="__main__":
    main()
