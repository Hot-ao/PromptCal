"""
국면 C-3: COCO 80 vs LVIS 1203 — 촘촘한 vocabulary에서 손상이 증폭되는가?

controlled 비교: 같은 이미지·같은 region·같은 양자화 모델(calibration은 vision-only라
프롬프트 무관 → 한 번만 calibrate 후 재사용). 프롬프트 vocabulary만 COCO↔LVIS로 교체.

측정(각 vocabulary):
  - flip rate (confident region)                : 손상 빈도
  - flip → 의미적 이웃(top-k) 비율 / 우연 대비 배수 : semantic substitution (C-1 재현)
  - mean margin, small-margin(<0.5) 비율         : 경쟁 심화의 메커니즘(국면 B와 연결)

기대: LVIS에서 margin↓(경쟁↑) → flip↑, 이웃 편향 유지 → "규모가 커지면 손상 증폭".

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/07_lvis_compare.py \
        --coco-root /data/taeho/coco_datasets --calib 64 --eval 500 --k 5 --device 0
"""

import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness, paired_run
from src.quant.quant_model import wrap_convs, calibrate


def load_names(which):
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / f"{which}.yaml"
    d = yaml.safe_load(open(p))
    n = d["names"]
    return [n[i] for i in range(len(n))]


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    return cv2.copyMakeBorder(im_r, top, new - nh - top, left, new - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(path, imgsz, device):
    im = letterbox(cv2.imread(path), imgsz)
    im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(im).float().unsqueeze(0).to(device) / 255.0


def neighbors_from_txt(txt, k):
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = txt @ txt.T
    P = sim.shape[0]
    idx = sim.argsort(dim=-1, descending=True)  # [P, P]
    neigh = []
    for c in range(P):
        row = idx[c].tolist()
        neigh.append(set([o for o in row if o != c][:k]))
    return neigh


def switch_vocab(model, names, device):
    """vocabulary 교체 시 device 불일치 회피: CPU로 내려 CLIP을 새로 빌드해
    txt_feats를 CPU에서 계산한 뒤 GPU로 복귀. 양자화 buffer는 이동에도 보존됨."""
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.to(device).eval()


def measure_vocab(tag, names, model_fp, model_q, eval_imgs, args, device):
    """한 vocabulary에 대해 flip/substitution/margin 측정."""
    switch_vocab(model_fp, names, device)
    switch_vocab(model_q, names, device)
    P = len(names)
    txt = model_fp.model.txt_feats.detach().float()
    if txt.dim() == 3:
        txt = txt[0]
    neigh = neighbors_from_txt(txt, args.k)

    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    n_conf = n_flip = flip_to_neigh = 0
    margin_sum = 0.0
    small_margin = 0

    for i, p in enumerate(eval_imgs):
        t = preprocess(p, args.imgsz, device)
        rec_fp, rec_q = paired_run(h_fp, h_q, t, image_id=i)
        sim_fp, sim_q = rec_fp.sim, rec_q.sim
        prob = sim_fp.sigmoid()
        maxp, c_fp = prob.max(-1)
        conf = maxp > args.conf_thres
        if conf.sum() == 0:
            continue
        c_q = sim_q.argmax(-1)
        top2 = sim_fp.topk(2, -1).values
        margin = top2[:, 0] - top2[:, 1]

        idxc = conf.nonzero(as_tuple=True)[0]
        n_conf += int(conf.sum())
        margin_sum += float(margin[idxc].sum())
        small_margin += int((margin[idxc] < 0.5).sum())
        for j in idxc.tolist():
            a = int(c_fp[j]); b = int(c_q[j])
            if b != a:
                n_flip += 1
                if b in neigh[a]:
                    flip_to_neigh += 1

        if (i + 1) % 100 == 0:
            print(f"   [{tag}] [{i+1}/{len(eval_imgs)}] flips={n_flip}")

    h_fp.close(); h_q.close()

    chance = args.k / (P - 1)
    flip_rate = n_flip / max(n_conf, 1) * 100
    to_neigh = flip_to_neigh / max(n_flip, 1) * 100
    bias = (flip_to_neigh / max(n_flip, 1)) / chance
    mean_margin = margin_sum / max(n_conf, 1)
    small_pct = small_margin / max(n_conf, 1) * 100
    return dict(P=P, n_conf=n_conf, n_flip=n_flip, flip_rate=flip_rate,
                to_neigh=to_neigh, chance=chance*100, bias=bias,
                mean_margin=mean_margin, small_pct=small_pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    coco = load_names("coco")
    lvis = load_names("lvis")

    from ultralytics import YOLOWorld
    print("[load] FP32 + quant models")
    model_fp = YOLOWorld(args.model); model_fp.set_classes(coco)
    model_q = YOLOWorld(args.model); model_q.set_classes(coco)
    model_fp.fuse(); model_q.fuse()
    wrap_convs(model_q.model, w_bits=8, a_bits=8)
    model_fp.model.to(device).eval(); model_q.model.to(device).eval()

    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_imgs = all_imgs[:args.calib]
    eval_imgs = all_imgs[args.calib:args.calib + args.eval]
    print(f"[calib] {len(calib_imgs)}장 (vision-only, 프롬프트 무관)...")
    calibrate(model_q.model, (preprocess(p, args.imgsz, device) for p in calib_imgs), device=device)

    print(f"[measure] COCO-80 ...")
    r_coco = measure_vocab("COCO", coco, model_fp, model_q, eval_imgs, args, device)
    print(f"[measure] LVIS-1203 ...")
    r_lvis = measure_vocab("LVIS", lvis, model_fp, model_q, eval_imgs, args, device)

    def row(tag, r):
        return (f" {tag:>9} | {r['P']:>5} | {r['n_conf']:>7} | {r['flip_rate']:>6.2f}% | "
                f"{r['to_neigh']:>6.1f}% | {r['bias']:>6.1f}x | {r['mean_margin']:>7.2f} | "
                f"{r['small_pct']:>6.1f}%")

    print("\n" + "=" * 78)
    print(" COCO vs LVIS — vocabulary 규모별 손상 (같은 이미지·region·양자화)")
    print("=" * 78)
    print(f" {'vocab':>9} | {'P':>5} | {'conf':>7} | {'flip':>6} | {'→이웃':>6} | "
          f"{'편향':>6} | {'margin':>7} | {'small':>6}")
    print("-" * 78)
    print(row("COCO-80", r_coco))
    print(row("LVIS-1203", r_lvis))
    print("=" * 78)
    print("\n해석:")
    print(f" - flip: {r_coco['flip_rate']:.2f}% → {r_lvis['flip_rate']:.2f}% "
          f"(LVIS에서 {'증가' if r_lvis['flip_rate']>r_coco['flip_rate'] else '감소'})")
    print(f" - mean margin: {r_coco['mean_margin']:.2f} → {r_lvis['mean_margin']:.2f} "
          f"(작을수록 경쟁↑)")
    print(f" - small-margin 비율: {r_coco['small_pct']:.1f}% → {r_lvis['small_pct']:.1f}%")
    print(" LVIS에서 margin↓·small↑와 함께 flip↑이면, 촘촘한 vocabulary가 경쟁을 심화시켜")
    print(" 손상을 증폭 = OVOD 규모에서 우리 주장이 성립. 이웃 편향 유지 시 substitution도 견고.")


if __name__ == "__main__":
    main()
