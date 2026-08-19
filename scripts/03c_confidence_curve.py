"""
3-c: confidence / margin 구간별 flip 곡선 — 손상이 경계에 집중됨을 정량 확정.

두 질문에 답한다:
  Q1. all-anchor flip 30%는 신호(경계 손상)인가 노이즈(배경 앵커)인가?
      -> 낮은 maxprob 구간의 앵커 수와 flip을 보면 판별됨.
  Q2. "손상이 경계일수록 심하다"가 단조적으로 성립하는가?
      -> maxprob 낮을수록 / top1-top2 margin 작을수록 flip이 높아지는 곡선으로 확인.

두 축으로 flip을 구간화:
  (A) FP maxprob (confidence): 이 region이 얼마나 확신하는가
  (B) FP top1-top2 margin (logit): 1·2위가 얼마나 근소한가 = 경계다움(직접 지표)

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/03c_confidence_curve.py \
        --coco-root /data/taeho/coco_datasets --calib 64 --eval 200 --device 0
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness, paired_run
from src.quant.quant_model import wrap_convs, calibrate


def load_coco_names():
    import ultralytics
    import yaml
    from pathlib import Path
    cfg = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    with open(cfg) as f:
        names = yaml.safe_load(f)["names"]
    return [names[i] for i in range(len(names))]


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    bottom, right = new - nh - top, new - nw - left
    return cv2.copyMakeBorder(im_r, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(path, imgsz, device):
    im0 = cv2.imread(path)
    im = letterbox(im0, imgsz)
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    t = torch.from_numpy(im).float() / 255.0
    return t.unsqueeze(0).to(device)


def bar(frac, width=24):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def print_curve(title, edges, cnt, flip_sum, unit=""):
    total = cnt.sum()
    print(f"\n{title}")
    print(f"{'구간':>16} | {'앵커수':>9} | {'비율':>6} | {'flip':>6} |")
    print("-" * 62)
    for b in range(len(edges) - 1):
        n = int(cnt[b])
        share = n / total * 100 if total > 0 else 0
        rate = (flip_sum[b] / cnt[b]).item() if cnt[b] > 0 else 0.0
        lo, hi = edges[b], edges[b + 1]
        rng = f"[{lo:.2f},{hi:.2f}){unit}"
        print(f"{rng:>16} | {n:9d} | {share:5.1f}% | {rate*100:5.1f}% {bar(rate)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()

    from ultralytics import YOLOWorld
    print("[load] FP32 + quant models")
    model_fp = YOLOWorld(args.model); model_fp.set_classes(names)
    model_q = YOLOWorld(args.model); model_q.set_classes(names)
    model_fp.fuse(); model_q.fuse()
    n = wrap_convs(model_q.model, w_bits=8, a_bits=8)
    model_fp.model.to(device).eval(); model_q.model.to(device).eval()
    print(f"[quant] wrapped {n} Conv2d (W8A8)")

    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_imgs = all_imgs[:args.calib]
    eval_imgs = all_imgs[args.calib:args.calib + args.eval]

    print(f"[calib] {len(calib_imgs)} images...")
    calibrate(model_q.model, (preprocess(p, args.imgsz, device) for p in calib_imgs), device=device)

    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    # 구간 정의
    conf_edges = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001])
    marg_edges = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9])
    conf_t = torch.tensor(conf_edges)
    marg_t = torch.tensor(marg_edges)
    nc, nm = len(conf_edges) - 1, len(marg_edges) - 1
    conf_cnt, conf_flip = torch.zeros(nc), torch.zeros(nc)
    marg_cnt, marg_flip = torch.zeros(nm), torch.zeros(nm)

    print(f"[eval] {len(eval_imgs)} images...")
    for i, p in enumerate(eval_imgs):
        t = preprocess(p, args.imgsz, device)
        rec_fp, rec_q = paired_run(h_fp, h_q, t, image_id=i)
        sim_fp, sim_q = rec_fp.sim, rec_q.sim

        prob = sim_fp.sigmoid()
        maxprob, amax_fp = prob.max(-1)
        amax_q = sim_q.argmax(-1)
        flip = (amax_fp != amax_q).float()
        top2 = sim_fp.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        cb = torch.bucketize(maxprob, conf_t, right=False) - 1
        cb = cb.clamp(0, nc - 1)
        mb = torch.bucketize(margin, marg_t, right=False) - 1
        mb = mb.clamp(0, nm - 1)
        for b in range(nc):
            sel = cb == b
            conf_cnt[b] += sel.sum(); conf_flip[b] += flip[sel].sum()
        for b in range(nm):
            sel = mb == b
            marg_cnt[b] += sel.sum(); marg_flip[b] += flip[sel].sum()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(eval_imgs)}]")

    h_fp.close(); h_q.close()

    print("\n" + "=" * 62)
    print(" W8A8 FLIP — 경계 집중도 분석 (200 images)")
    print("=" * 62)
    print_curve("(A) FP maxprob(confidence) 구간별 top-1 flip", conf_edges, conf_cnt, conf_flip)
    print_curve("(B) FP top1-top2 margin(logit) 구간별 top-1 flip  ← 경계 직접 지표",
                marg_edges, marg_cnt, marg_flip, unit="")

    print("\n[읽는 법]")
    print(" - (A)에서 낮은 maxprob 구간이 앵커 대부분을 차지하고 flip이 높으면,")
    print("   all-anchor 30%의 상당수는 '배경/저확신 앵커'의 흔들림(노이즈).")
    print(" - (B)에서 margin 작을수록 flip↑, margin 클수록 flip≈0의 단조 감소가 보이면,")
    print("   '양자화 손상이 semantic decision boundary에 집중된다'가 정량 확정됨.")


if __name__ == "__main__":
    main()
