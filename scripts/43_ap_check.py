"""
AP 검증 (09-06, flip 진단 이후 첫 AP 확인).

29~42에서 H_eval flip만 봤다. 논문 §3.4 "Reconstruction-Utility Misalignment"가
스스로 경고하듯, rank/flip 보존이 AP를 보장하지 않는다. 이 스크립트는 41번에서
확정한 조합(AdaRound weight + 방향 C 연속 scale + asymmetric neighbor preservation,
neighbor_weight=1.0)이 실제 pycocotools AP까지 유지/개선하는지 확인한다.

대표 seed 하나(기본 2 -- person 등 이상치 없는 정상적인 승리 케이스, 09-06 §42)로:
  FP32 / naive W8A8 / AdaRound / Combined(AdaRound+scale+asym-neighbor)
네 조건의 전체 80-class mAP50-95/mAP50을 먼저 비교하고,
AdaRound vs Combined는 S(계산에 쓴 40개)/H_eval(전혀 안 쓴 20개) subset별
mAP도 따로 뽑아서 flip 지표와 같은 축(seen vs held-out)으로 AP까지 본다.

실행:
    CUDA_VISIBLE_DEVICES=7 python scripts/43_ap_check.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --calib 32 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.promptcal import optimize_promptcal_scale_neighbor


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
    """전체 mAP + per-class mAP50-95(ap_class_index 순서) 반환."""
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False, verbose=False)
    overall = float(metrics.box.map) * 100, float(metrics.box.map50) * 100
    per_class = dict(zip(metrics.box.ap_class_index.tolist(),
                         (metrics.box.maps if hasattr(metrics.box, "maps") else metrics.box.all_ap[:, 0]).tolist()))
    return overall, per_class


def subset_map(per_class, idx, scale=100.0):
    vals = [per_class[i] for i in idx if i in per_class]
    return float(np.mean(vals)) * scale if vals else float("nan")


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0):
    m = model_cls(w)
    m.set_classes(names)
    if mode == "fp":
        m.fuse(); m.model.to(device).eval()
        return m
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "naive":
        pass
    elif mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "combined":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=True, verbose=True)
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

    print("[build] FP");       fp = build(YOLOWorld, args.model, names, device, calib, "fp")
    print("[build] naive");    nv = build(YOLOWorld, args.model, names, device, calib, "naive")
    print("[build] AdaRound"); ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)
    print("[build] Combined (AdaRound+scale+asym-neighbor)")
    cb = build(YOLOWorld, args.model, names, device, calib, "combined", fp=fp,
              iters=args.iters, pidx=pidx, lr=args.lr, k=args.k,
              neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight)

    print("\n[ap] FP32 ..."); (fp_map, fp_map50), fp_pc = measure_ap(fp, args.data, args.imgsz, args.device)
    print("[ap] naive ..."); (nv_map, nv_map50), nv_pc = measure_ap(nv, args.data, args.imgsz, args.device)
    print("[ap] AdaRound ..."); (ad_map, ad_map50), ad_pc = measure_ap(ad, args.data, args.imgsz, args.device)
    print("[ap] Combined ..."); (cb_map, cb_map50), cb_pc = measure_ap(cb, args.data, args.imgsz, args.device)

    print("\n" + "=" * 80)
    print(f" 전체 80-class AP (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>10} | {'mAP50-95':>9} | {'mAP50':>9}")
    print(f"{'FP32':>10} | {fp_map:>9.2f} | {fp_map50:>9.2f}")
    print(f"{'naive':>10} | {nv_map:>9.2f} | {nv_map50:>9.2f}")
    print(f"{'AdaRound':>10} | {ad_map:>9.2f} | {ad_map50:>9.2f}")
    print(f"{'Combined':>10} | {cb_map:>9.2f} | {cb_map50:>9.2f}")

    print("\n" + "=" * 80)
    print(f" S(계산에 씀, 40) vs H_eval(전혀 안 씀, 20) subset mAP50-95 (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>10} | {'S mAP':>8} | {'H_eval mAP':>10}")
    print(f"{'AdaRound':>10} | {subset_map(ad_pc, S):>8.2f} | {subset_map(ad_pc, H_eval):>10.2f}")
    print(f"{'Combined':>10} | {subset_map(cb_pc, S):>8.2f} | {subset_map(cb_pc, H_eval):>10.2f}")

    print("\n판정:")
    print("  전체 mAP: Combined가 AdaRound 대비 크게 떨어지면(예: -1.0 이상) -> flip 개선이")
    print("     AP 희생으로 얻어진 것. 논문 §4.3 utility constraint 없이는 위험.")
    print("  H_eval subset mAP: Combined > AdaRound면 -> flip 개선이 실제 held-out 탐지")
    print("     성능 향상으로도 이어진다는 직접 증거(논문 핵심 주장 지지).")


if __name__ == "__main__":
    main()
