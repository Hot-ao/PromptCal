"""
per-channel s_mult 재설계 1차 검증 (09-07, §7.6 옵션 1). src/quant/adaround.py의
s_mult를 conv당 스칼라 -> [in_channels] 벡터로 바꾼 뒤(scale_reg_weight=0,
정규화 없음), COCO-80/LVIS 양쪽에서 원래 스칼라 버전(36.46/0.216 근방) 대비
어떻게 달라지는지 확인.

확인 포인트:
  1. 채널 간 실제로 다른 값으로 갈라지는가(분산이 0에 가까우면 per-channel이
     스칼라와 다를 게 없다는 뜻 -- 최적화가 그냥 전부 같은 방향으로 움직였다면
     구조 확장이 무의미).
  2. LVIS AP가 스칼라 버전(naive보다 낮았음, ~0.21)보다 나아지는가 -- 특히
     AdaRound(0.2232) 수준에 가까워지는가.
  3. COCO-80 AP가 원래 스칼라 버전의 이득(36.46, +1.44)을 유지하는가 -- 옵션 1은
     "이득은 유지하면서 부작용만 없애는" 게 목표였다는 점에서 가장 중요한 확인.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/56_perchannel_smult_check.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --lvis-ann /data/taeho/lvis_datasets/annotations/lvis_v1_val.json \
        --calib 32 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround, list_adaround_convs
from src.quant.promptcal import optimize_promptcal_scale_neighbor


def load_names(which):
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / f"{which}.yaml"
    d = yaml.safe_load(open(p))
    n = d["names"]
    return [n[i] for i in range(len(n))]


def preprocess(path, imgsz, device):
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    im_p = cv2.copyMakeBorder(im_r, top, imgsz - nh - top, left, imgsz - nw - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    im_p = np.ascontiguousarray(im_p[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(im_p).float().unsqueeze(0).to(device) / 255.0


def switch_vocab(model, names, device):
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.to(device).eval()


def measure_ap(model, data, imgsz, device):
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False, verbose=False)
    return float(metrics.box.map) * 100, float(metrics.box.map50) * 100


def predict_lvis_results(model, img_paths, img_ids, imgsz, device, conf=0.001, max_det=300):
    results = []
    for path, img_id in zip(img_paths, img_ids):
        r = model.predict(source=path, imgsz=imgsz, device=device, conf=conf,
                          max_det=max_det, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
            results.append({"image_id": int(img_id), "category_id": int(c) + 1,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(sc)})
    return results


def run_lvis_eval(lvis_gt, results, img_ids):
    from lvis import LVISEval, LVISResults
    if not results:
        return dict(AP=0.0, AP50=0.0)
    lvis_dt = LVISResults(lvis_gt, results, max_dets=300)
    ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
    ev.params.img_ids = img_ids
    ev.run()
    return ev.get_results()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--lvis-ann", default="/data/taeho/lvis_datasets/annotations/lvis_v1_val.json")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--scale-reg-weight", type=float, default=0.0)
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

    from lvis import LVIS
    print(f"[lvis] loading GT {args.lvis_ann}")
    lvis_gt = LVIS(args.lvis_ann)
    lvis_img_ids = set(lvis_gt.get_img_ids())

    coco = load_names("coco")
    lvis_names = load_names("lvis")
    from ultralytics import YOLOWorld
    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_paths = all_imgs[:args.calib]
    probe_paths_all = all_imgs[args.calib:args.calib + args.eval]
    probe_paths, probe_ids = [], []
    for p in probe_paths_all:
        iid = int(os.path.basename(p).split(".")[0])
        if iid in lvis_img_ids:
            probe_paths.append(p); probe_ids.append(iid)
    print(f"[filter] probe {len(probe_paths_all)}장 중 LVIS val에 있는 {len(probe_paths)}장만 사용")
    calib = [preprocess(p, args.imgsz, device) for p in calib_paths]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist()

    print("[build] FP (COCO-80)")
    fp = YOLOWorld(args.model); fp.set_classes(coco); fp.fuse(); fp.model.to(device).eval()

    print("[build] adaround (참조)")
    ada = YOLOWorld(args.model); ada.set_classes(coco); ada.fuse()
    wrap_convs(ada.model, 8, 8); ada.model.to(device).eval()
    calibrate(ada.model, calib, device=device)
    convert_to_adaround(ada.model)
    optimize_adaround(ada.model, fp.model, calib, device, iters=args.recon_iters_ada, verbose=False)

    print("[build] combined (per-channel s_mult)")
    cb = YOLOWorld(args.model); cb.set_classes(coco); cb.fuse()
    wrap_convs(cb.model, 8, 8); cb.model.to(device).eval()
    calibrate(cb.model, calib, device=device)
    convert_to_adaround(cb.model)
    optimize_adaround(cb.model, fp.model, calib, device, iters=args.recon_iters_ada, verbose=False)
    optimize_promptcal_scale_neighbor(cb.model, fp.model, calib, device, S, iters=args.iters,
                                      lr=args.lr, k=args.k, neighbor_k=args.neighbor_k,
                                      neighbor_weight=args.neighbor_weight,
                                      asymmetric=True, scale_reg_weight=args.scale_reg_weight,
                                      verbose=True)

    ada_convs = list_adaround_convs(cb.model)
    all_vals = torch.cat([c.s_mult.detach().flatten() for c in ada_convs])
    per_conv_std = np.mean([float(c.s_mult.detach().std()) for c in ada_convs if c.s_mult.numel() > 1])
    print(f"\n[s_mult] 전체 평균={float(all_vals.mean()):.4f} 전체 std={float(all_vals.std()):.4f} "
          f"conv-내부 평균 std={per_conv_std:.4f} (0에 가까우면 채널들이 다 같이 움직인 것)")
    print(f"[s_mult] min={float(all_vals.min()):.4f} max={float(all_vals.max()):.4f}")

    # COCO-80 AP
    coco_ap_ada, _ = measure_ap(ada, args.data, args.imgsz, args.device)
    coco_ap_cb, _ = measure_ap(cb, args.data, args.imgsz, args.device)

    # LVIS AP (이식)
    switch_vocab(ada, lvis_names, device)
    switch_vocab(cb, lvis_names, device)
    preds_ada = predict_lvis_results(ada, probe_paths, probe_ids, args.imgsz, device)
    lvis_ada = run_lvis_eval(lvis_gt, preds_ada, probe_ids)
    preds_cb = predict_lvis_results(cb, probe_paths, probe_ids, args.imgsz, device)
    lvis_cb = run_lvis_eval(lvis_gt, preds_cb, probe_ids)

    print("\n" + "=" * 80)
    print(f" per-channel s_mult 1차 검증 (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>10} | {'COCO-80 AP':>10} | {'LVIS AP':>8}")
    print(f"{'adaround':>10} | {coco_ap_ada:>10.2f} | {lvis_ada.get('AP',0):>8.4f}")
    print(f"{'combined':>10} | {coco_ap_cb:>10.2f} | {lvis_cb.get('AP',0):>8.4f}")
    print("\n참고(스칼라 버전, 09-07 이전 결과): COCO-80 AP=36.46±0.06, LVIS AP≈0.21(naive=0.2175보다 낮음)")
    print("판정: COCO-80 AP가 36 근처를 유지하면서 LVIS AP가 naive(0.2175)를 넘고")
    print("  AdaRound(0.2232)에 가까워지면 -> per-channel이 실제로 트레이드오프를 개선한 것.")


if __name__ == "__main__":
    main()
