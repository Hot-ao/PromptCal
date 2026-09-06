"""
Utility-Constrained Refinement ablation (09-06, 43번 다음 단계).

논문 §4.3 Utility Constraints(threshold-crossing + box consistency,
semantic_calib.py의 utility_refinement_terms)를 43번에서 검증한 asymmetric
neighbor preservation 위에 재통합해서, AP가 더 좋아지는지 확인한다. §6.4
Ablation Study("Semantic Calibration / Utility Refinement on/off")가 요구하는
비교이기도 하다.

비교:
  AdaRound              baseline
  Combined              AdaRound weight + asymmetric neighbor scale (43번, utility 없음)
  Combined+Utility      위에 stage2(threshold-crossing + box consistency) 추가

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/44_utility_ap_check.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --calib 32 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.promptcal import optimize_promptcal_scale_neighbor, optimize_promptcal_scale_neighbor_utility


def load_coco_names():
    import ultralytics, yaml
    from pathlib import Path
    d = yaml.safe_load(open(Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"))
    return [d["names"][i] for i in range(len(d["names"]))]


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


def measure_ap(model, data, imgsz, device):
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False, verbose=False)
    overall = float(metrics.box.map) * 100, float(metrics.box.map50) * 100
    per_class = dict(zip(metrics.box.ap_class_index.tolist(),
                         (metrics.box.maps if hasattr(metrics.box, "maps") else metrics.box.all_ap[:, 0]).tolist()))
    return overall, per_class


def subset_map(per_class, idx, scale=100.0):
    vals = [per_class[i] for i in idx if i in per_class]
    return float(np.mean(vals)) * scale if vals else float("nan")


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0,
          stage2_frac=0.3, thresh_w=1.0, box_w=0.5):
    m = model_cls(w)
    m.set_classes(names)
    if mode == "fp":
        m.fuse(); m.model.to(device).eval()
        return m
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "combined":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=True, verbose=True)
    elif mode == "combined_utility":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_promptcal_scale_neighbor_utility(m.model, fp.model, calib, device, pidx,
                                                  iters=iters, lr=lr, k=k, neighbor_k=neighbor_k,
                                                  neighbor_weight=neighbor_weight,
                                                  stage2_frac=stage2_frac, thresh_w=thresh_w,
                                                  box_w=box_w, verbose=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--stage2-frac", type=float, default=0.3)
    ap.add_argument("--thresh-w", type=float, default=1.0)
    ap.add_argument("--box-w", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[determinism] torch.manual_seed={args.torch_seed}, cudnn.deterministic=True, cudnn.benchmark=False")

    names = load_coco_names()
    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib = [preprocess(p, args.imgsz, device) for p in imgs[:args.calib]]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
    pidx = S

    print("[build] FP");              fp = build(YOLOWorld, args.model, names, device, calib, "fp")
    print("[build] AdaRound");        ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)
    print("[build] Combined (neighbor only)")
    cb = build(YOLOWorld, args.model, names, device, calib, "combined", fp=fp,
              iters=args.iters, pidx=pidx, lr=args.lr, k=args.k,
              neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight)
    print("[build] Combined+Utility")
    cu = build(YOLOWorld, args.model, names, device, calib, "combined_utility", fp=fp,
              iters=args.iters, pidx=pidx, lr=args.lr, k=args.k,
              neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
              stage2_frac=args.stage2_frac, thresh_w=args.thresh_w, box_w=args.box_w)

    print("\n[ap] AdaRound ..."); (ad_map, ad_map50), ad_pc = measure_ap(ad, args.data, args.imgsz, args.device)
    print("[ap] Combined ..."); (cb_map, cb_map50), cb_pc = measure_ap(cb, args.data, args.imgsz, args.device)
    print("[ap] Combined+Utility ..."); (cu_map, cu_map50), cu_pc = measure_ap(cu, args.data, args.imgsz, args.device)

    print("\n" + "=" * 80)
    print(f" 전체 80-class AP (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>18} | {'mAP50-95':>9} | {'mAP50':>9}")
    print(f"{'AdaRound':>18} | {ad_map:>9.2f} | {ad_map50:>9.2f}")
    print(f"{'Combined':>18} | {cb_map:>9.2f} | {cb_map50:>9.2f}")
    print(f"{'Combined+Utility':>18} | {cu_map:>9.2f} | {cu_map50:>9.2f}")

    print("\n" + "=" * 80)
    print(f" S(계산에 씀, 40) vs H_eval(전혀 안 씀, 20) subset mAP50-95 (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>18} | {'S mAP':>8} | {'H_eval mAP':>10}")
    print(f"{'AdaRound':>18} | {subset_map(ad_pc, S):>8.2f} | {subset_map(ad_pc, H_eval):>10.2f}")
    print(f"{'Combined':>18} | {subset_map(cb_pc, S):>8.2f} | {subset_map(cb_pc, H_eval):>10.2f}")
    print(f"{'Combined+Utility':>18} | {subset_map(cu_pc, S):>8.2f} | {subset_map(cu_pc, H_eval):>10.2f}")

    print("\n판정:")
    print("  Combined+Utility > Combined (특히 H_eval subset) -> utility constraint가")
    print("     추가 이득을 준다. 논문 §4.3 포함이 정당화됨.")
    print("  Combined+Utility ≈ Combined -> semantic calibration만으로 이미 충분,")
    print("     utility constraint는 없어도 되거나 이 설정에서는 중립적.")
    print("  Combined+Utility < Combined -> utility 항(thresh_w/box_w)이 오히려")
    print("     방해됨 -> 가중치 재조정 필요.")


if __name__ == "__main__":
    main()
