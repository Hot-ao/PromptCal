"""
3단계: Naive W8A8 양자화의 region-prompt 의미 손상 측정 (논문 Sec 3 핵심 그림).

절차:
  1. FP32 YOLO-World와, 그 사본을 W8A8로 양자화한 모델을 각각 준비.
  2. calibration 이미지로 양자화 모델 activation scale 보정.
  3. eval 이미지들에 대해 FP vs 양자화 유사도 행렬을 SimilarityHarness로 쌍(pair) 캡처.
  4. Top-1 flip rate / boundary inversion rate 계산 (둘 다 GT 불필요, 순위 기반).
  헤드라인 주장: "AP는 소폭 하락하나 region-prompt 의사결정은 크게 손상된다."

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/03_motivation_damage.py \
        --coco-root /data/taeho/coco_datasets \
        --calib 64 --eval 200 --device 0
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

# repo 루트를 path에 추가 (src 임포트용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness, paired_run
from src.metrics.semantic_metrics import top1_flip_rate, boundary_inversion_rate
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
    im0 = cv2.imread(path)                       # BGR HWC
    im = letterbox(im0, imgsz)
    im = im[:, :, ::-1].transpose(2, 0, 1)       # BGR->RGB, HWC->CHW
    im = np.ascontiguousarray(im)
    t = torch.from_numpy(im).float() / 255.0
    return t.unsqueeze(0).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64, help="calibration 이미지 수")
    ap.add_argument("--eval", type=int, default=200, help="손상 측정 이미지 수")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf-thres", type=float, default=0.25,
                    help="confident region 판정 임계 (FP maxprob 기준)")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="boundary inversion margin (logit 공간)")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()
    print(f"[info] prompts={len(names)}  device={device}")

    from ultralytics import YOLOWorld
    print("[load] FP32 model")
    model_fp = YOLOWorld(args.model)
    model_fp.set_classes(names)
    print("[load] quant model (copy)")
    model_q = YOLOWorld(args.model)
    model_q.set_classes(names)

    # 양자화 전에 두 모델 모두 conv+BN fuse: 실제 배포 weight를 양자화하고,
    # BN이 양자화 오차를 씻어내는 artifact를 제거(FP/Q 동일 상태로 공정 비교).
    # fuse는 conv+BN 병합만 하며 cv4(유사도 캡처 지점)는 건드리지 않는다.
    model_fp.fuse()
    model_q.fuse()

    n_wrapped = wrap_convs(model_q.model, w_bits=8, a_bits=8)
    print(f"[quant] wrapped {n_wrapped} Conv2d as QuantConv2d (W8A8)")

    # 래핑 후 두 모델을 명시적으로 device로 올린다(래핑된 conv weight까지 함께 이동).
    model_fp.model.to(device).eval()
    model_q.model.to(device).eval()

    # 이미지 목록
    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_imgs = all_imgs[:args.calib]
    eval_imgs = all_imgs[args.calib:args.calib + args.eval]   # calib과 분리
    print(f"[data] calib={len(calib_imgs)}  eval={len(eval_imgs)}")

    # 캘리브레이션
    print("[calib] running...")
    calib_tensors = (preprocess(p, args.imgsz, device) for p in calib_imgs)
    calibrate(model_q.model, calib_tensors, device=device)
    print("[calib] done, quant mode ON")

    # 하네스
    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    # 손상 측정 누적
    flip_all, flip_conf, binv = [], [], []
    conf_region_total = 0
    for i, p in enumerate(eval_imgs):
        t = preprocess(p, args.imgsz, device)
        rec_fp, rec_q = paired_run(h_fp, h_q, t, image_id=i)
        sim_fp, sim_q = rec_fp.sim, rec_q.sim

        # confident region mask: FP에서 어떤 프롬프트든 maxprob > thres
        fp_maxprob = sim_fp.sigmoid().max(dim=-1).values
        mask = fp_maxprob > args.conf_thres
        conf_region_total += int(mask.sum())

        flip_all.append(top1_flip_rate(sim_fp, sim_q))
        flip_conf.append(top1_flip_rate(sim_fp, sim_q, region_mask=mask))
        binv.append(boundary_inversion_rate(sim_fp, sim_q, margin=args.margin))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(eval_imgs)}] "
                  f"flip_conf={np.mean(flip_conf)*100:.2f}%")

    h_fp.close()
    h_q.close()

    def pct(xs):
        return float(np.mean(xs)) * 100 if xs else 0.0

    print("\n" + "=" * 52)
    print(" NAIVE W8A8 — REGION-PROMPT SEMANTIC DAMAGE")
    print("=" * 52)
    print(f" eval images            : {len(eval_imgs)}")
    print(f" confident regions/img  : {conf_region_total/max(len(eval_imgs),1):.1f} "
          f"(FP maxprob > {args.conf_thres})")
    print(f" Top-1 flip (all anchors): {pct(flip_all):.2f}%")
    print(f" Top-1 flip (confident)  : {pct(flip_conf):.2f}%   <- 핵심 지표")
    print(f" Boundary inversion      : {pct(binv):.2f}%  (margin={args.margin} logit)")
    print("=" * 52)
    print("\n해석: confident region에서의 Top-1 flip이 높을수록, 양자화가 실제로 탐지에")
    print("영향을 주는 지점들의 region-prompt 의사결정을 뒤집었다는 뜻. AP 하락(별도 측정)과")
    print("대비하면 'AP는 조금 떨어지는데 의미적 의사결정은 크게 손상' 주장을 입증.")


if __name__ == "__main__":
    main()
