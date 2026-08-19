"""
3-b: Naive W8A8 양자화 모델의 AP 측정 (양자화가 실제로 걸렸는지 검증).

03의 손상 지표(confident flip 0.77%)가 "손상이 작다"인지 "지표가 둔하다"인지
판별하려면, 먼저 양자화가 실제로 AP를 떨어뜨리는지 확인해야 한다.
  - FP 37.8 그대로면 -> 양자화 무력화(구현 재점검)
  - 유의미하게 하락(예: 35~36)이면 -> 양자화는 제대로, 지표가 둔한 것

절차: 03과 동일하게 wrap+calibrate+quantized 모드로 만든 모델을 그대로
ultralytics val(pycocotools)에 태워 size별 AP까지 뽑는다.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/03b_quant_ap.py \
        --coco-root /data/taeho/coco_datasets --data configs/coco_local.yaml \
        --calib 64 --device 0
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", action="store_true", default=True,
                    help="양자화 적용(기본). --no-quantize로 FP 대조군 가능")
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()

    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    model.set_classes(names)

    if args.quantize:
        # 양자화 전에 conv+BN을 fuse: (1) 실제 배포 weight를 양자화(올바른 PTQ),
        # (2) 이미 fused라 val이 다시 fuse 시도 안 함(크래시 방지). cv4는 안 건드림.
        model.fuse()
        n = wrap_convs(model.model, w_bits=8, a_bits=8)
        model.model.to(device).eval()
        print(f"[quant] wrapped {n} Conv2d, calibrating on {args.calib} images...")
        calib_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))[:args.calib]
        calibrate(model.model, (preprocess(p, args.imgsz, device) for p in calib_imgs),
                  device=device)
        print("[quant] W8A8 mode ON")

        # 양자화 sanity: 실제로 int8로 동작 중인지 첫 QuantConv 상태 출력
        from src.quant.fake_quant import QuantConv2d
        for m in model.model.modules():
            if isinstance(m, QuantConv2d):
                print(f"[check] sample QuantConv: quantized={m.quantized}, "
                      f"a_scale={m.a_obs.scale.item():.4f}, ready={m.a_obs.ready}")
                break
    else:
        model.model.to(device).eval()
        print("[fp] no quantization (control)")

    tag = "W8A8" if args.quantize else "FP32"
    print(f"\n[val] evaluating {tag} model on {args.data} ...")
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device,
                        save_json=True, verbose=False)

    got = float(metrics.box.map) * 100
    print("\n" + "=" * 44)
    print(f" {tag} AP (Ultralytics metric)")
    print("=" * 44)
    print(f" mAP50-95 : {got:.2f}")
    print(f" mAP50    : {float(metrics.box.map50)*100:.2f}")
    print(f" (pycocotools 값은 위 로그의 AP@[.50:.95|all] 참고)")
    print(f" 대조: FP32 baseline = 37.8 (pycocotools)")
    print("=" * 44)


if __name__ == "__main__":
    main()
