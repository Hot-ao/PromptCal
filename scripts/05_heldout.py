"""
국면 C-2: Held-out Vocabulary — calibration/vocab에 없던 프롬프트에서 손상이 악화되는가?
(multi-seed 버전: 분할 운이 아님을 확인)

설계(방식 1, 클래스 분할 + FP pseudo-GT):
  - COCO 80을 K(유지)/H(held-out)로 분할. FP full-80 top-1이 H인 confident region = target.
  - 트릭1: 모델 재실행 불필요 — 유사도 행렬에서 H열 -inf 마스킹 후 argmax만.
  - 트릭2: 분할(seed)은 유사도 행렬과 무관 — forward 1회로 여러 seed를 동시 평가.
  - 조건 비교: seen(full-80) vs held-out(K only)의 FP vs quant flip.
  - 기대: held-out flip > seen flip, 여러 seed에서 일관.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/05_heldout.py \
        --coco-root /data/taeho/coco_datasets --calib 64 --eval 2000 \
        --heldout 40 --seeds 0 1 2 3 4 --device 0
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
    return torch.from_numpy(im).float().unsqueeze(0).to(device) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()
    nc = len(names)

    # 각 seed의 held-out 마스크 미리 구성
    H_masks = {}
    for s in args.seeds:
        rng = np.random.default_rng(s)
        H = rng.permutation(nc)[:args.heldout]
        m = torch.zeros(nc, dtype=torch.bool); m[H.tolist()] = True
        H_masks[s] = m
    print(f"[split] {len(args.seeds)} seeds, held-out {args.heldout}/{nc} classes each")

    from ultralytics import YOLOWorld
    print("[load] FP32 + quant models")
    model_fp = YOLOWorld(args.model); model_fp.set_classes(names)
    model_q = YOLOWorld(args.model); model_q.set_classes(names)
    model_fp.fuse(); model_q.fuse()
    wrap_convs(model_q.model, w_bits=8, a_bits=8)
    model_fp.model.to(device).eval(); model_q.model.to(device).eval()

    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_imgs = all_imgs[:args.calib]
    eval_imgs = all_imgs[args.calib:args.calib + args.eval]
    print(f"[calib] {len(calib_imgs)}장...")
    calibrate(model_q.model, (preprocess(p, args.imgsz, device) for p in calib_imgs), device=device)

    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    # seed별 누적
    acc = {s: dict(n=0, seen=0, ho=0, m_seen=0.0, m_ho=0.0) for s in args.seeds}

    print(f"[eval] {len(eval_imgs)}장 x {len(args.seeds)} seeds...")
    for i, p in enumerate(eval_imgs):
        t = preprocess(p, args.imgsz, device)
        rec_fp, rec_q = paired_run(h_fp, h_q, t, image_id=i)
        sim_fp, sim_q = rec_fp.sim, rec_q.sim

        prob = sim_fp.sigmoid()
        maxprob, c_fp = prob.max(-1)
        conf = maxprob > args.conf_thres
        q_full = sim_q.argmax(-1)

        for s in args.seeds:
            Hm = H_masks[s]
            target = conf & Hm[c_fp]
            if target.sum() == 0:
                continue
            idx = target.nonzero(as_tuple=True)[0]
            a = acc[s]
            a["seen"] += int((q_full[idx] != c_fp[idx]).sum())
            fp_K = sim_fp.clone(); fp_K[:, Hm] = -1e9
            q_K = sim_q.clone();   q_K[:, Hm] = -1e9
            a["ho"] += int((fp_K.argmax(-1)[idx] != q_K.argmax(-1)[idx]).sum())
            t2s = sim_fp[idx].topk(2, -1).values
            a["m_seen"] += float((t2s[:, 0] - t2s[:, 1]).sum())
            t2h = fp_K[idx].topk(2, -1).values
            a["m_ho"] += float((t2h[:, 0] - t2h[:, 1]).sum())
            a["n"] += int(target.sum())

        if (i + 1) % 400 == 0:
            print(f"  [{i+1}/{len(eval_imgs)}]")

    h_fp.close(); h_q.close()

    print("\n" + "=" * 66)
    print(" HELD-OUT VOCABULARY (multi-seed)")
    print("=" * 66)
    print(f"{'seed':>5} | {'targets':>8} | {'seen%':>6} | {'held-out%':>9} | {'악화배수':>7} | {'m_seen':>6} | {'m_ho':>5}")
    print("-" * 66)
    sf_list, hf_list, amp_list = [], [], []
    for s in args.seeds:
        a = acc[s]; n = max(a["n"], 1)
        sf = a["seen"] / n * 100
        hf = a["ho"] / n * 100
        amp = hf / max(sf, 1e-9)
        sf_list.append(sf); hf_list.append(hf); amp_list.append(amp)
        print(f"{s:>5} | {a['n']:>8} | {sf:>5.2f}% | {hf:>8.2f}% | {amp:>6.1f}x | "
              f"{a['m_seen']/n:>6.2f} | {a['m_ho']/n:>5.2f}")
    print("-" * 66)
    print(f"{'mean':>5} | {'':>8} | {np.mean(sf_list):>5.2f}% | {np.mean(hf_list):>8.2f}% | "
          f"{np.mean(amp_list):>6.1f}x |")
    print(f"{'std':>5} | {'':>8} | {np.std(sf_list):>5.2f}  | {np.std(hf_list):>8.2f}  | "
          f"{np.std(amp_list):>6.1f}  |")
    print("=" * 66)
    print("\n해석: 여러 seed에서 악화배수가 일관되게 >1이면, held-out 손상 악화가")
    print("특정 분할 운이 아니라 견고한 현상임이 확정. (std가 작을수록 견고)")


if __name__ == "__main__":
    main()
