"""
W4A4 스모크 테스트 (09-07): bit-width를 8->4로 내리면 방법 간 차이가 아직
보이는 정도인지, 아니면 다 같이 랜덤 근처로 붕괴해서 비교 자체가 무의미해지는지
5분짜리로 먼저 확인. 의미 있으면 본격적으로 5-way x multi-seed로 확장.

fake_quant.py의 QuantConv2d/ActObserver는 이미 bits를 파라미터로 받으므로
wrap_convs(model, w_bits, a_bits) 호출만 바꾸면 됨 -- quant 스킴 자체(weight
per-channel symmetric, activation per-tensor asymmetric)는 그대로.

측정: naive와 Combined만, W8A8(sanity, 기존 6-seed 결과와 비교용)과 W4A4
둘 다에서 AP + 표준 Top1_flip.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python scripts/48_w4a4_probe.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --calib 32 --eval 200 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
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
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False, verbose=False)
    return float(metrics.box.map) * 100, float(metrics.box.map50) * 100


def standard_flip(fp_sims, q_sims, conf=0.25):
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
        prob = sf.sigmoid()
        mp, c_fp = prob.max(-1)
        conf_m = mp > conf
        if conf_m.sum() == 0:
            continue
        idx = conf_m.nonzero(as_tuple=True)[0]
        c_q = sq.argmax(-1)
        fl += int((c_fp[idx] != c_q[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def build(model_cls, w, names, device, calib, mode, w_bits, a_bits, fp=None,
          iters=1500, pidx=None, lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0,
          recon_iters_ada=1000):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, w_bits, a_bits)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "naive":
        pass
    elif mode == "combined":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=True, verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
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

    names = load_coco_names()
    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib = [preprocess(p, args.imgsz, device) for p in imgs[:args.calib]]
    probe_paths = imgs[args.calib:args.calib + args.eval]
    probe = [preprocess(p, args.imgsz, device) for p in probe_paths]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist()

    print("[load] FP32 (never quantized, shared reference)")
    fp = YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()

    results = {}
    for (w_bits, a_bits) in [(8, 8), (4, 4)]:
        tag = f"W{w_bits}A{a_bits}"
        print(f"\n===== {tag} =====")
        print(f"[build] naive {tag}")
        nv = build(YOLOWorld, args.model, names, device, calib, "naive", w_bits, a_bits)
        print(f"[build] combined {tag}")
        cb = build(YOLOWorld, args.model, names, device, calib, "combined", w_bits, a_bits,
                  fp=fp, iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                  neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                  recon_iters_ada=args.recon_iters_ada)

        h_fp = SimilarityHarness(fp.model, device=device)
        fp_sims = [h_fp.run_image(t, i).sim for i, t in enumerate(probe)]
        h_fp.close()

        flips = {}
        for label, m in [("naive", nv), ("combined", cb)]:
            h_q = SimilarityHarness(m.model, device=device)
            q_sims = [h_q.run_image(t, i).sim for i, t in enumerate(probe)]
            h_q.close()
            flips[label], n = standard_flip(fp_sims, q_sims)
            print(f"  {label}: Top1_flip={flips[label]:.2f}%(n={n})")

        print(f"[ap] {tag} naive ...")
        nv_ap, nv_ap50 = measure_ap(nv, args.data, args.imgsz, args.device)
        print(f"[ap] {tag} combined ...")
        cb_ap, cb_ap50 = measure_ap(cb, args.data, args.imgsz, args.device)
        results[tag] = dict(nv_ap=nv_ap, nv_ap50=nv_ap50, cb_ap=cb_ap, cb_ap50=cb_ap50,
                            nv_flip=flips["naive"], cb_flip=flips["combined"])

    fp_ap, fp_ap50 = measure_ap(fp, args.data, args.imgsz, args.device)

    print("\n" + "=" * 80)
    print(f" W8A8 vs W4A4 스모크 테스트 (seed {args.seed}, probe {len(probe_paths)}장)")
    print("=" * 80)
    print(f"{'':>12} | {'mAP50-95':>9} | {'mAP50':>9} | {'Top1_flip':>10}")
    print(f"{'FP32':>12} | {fp_ap:>9.2f} | {fp_ap50:>9.2f} | {'-':>10}")
    for tag in ["W8A8", "W4A4"]:
        r = results[tag]
        print(f"{tag+' naive':>12} | {r['nv_ap']:>9.2f} | {r['nv_ap50']:>9.2f} | {r['nv_flip']:>9.2f}%")
        print(f"{tag+' combined':>12} | {r['cb_ap']:>9.2f} | {r['cb_ap50']:>9.2f} | {r['cb_flip']:>9.2f}%")
    print("=" * 80)

    print("\n판정:")
    print("  W4A4 naive AP가 FP32 대비 크게 무너지지 않고(랜덤 근처 아님) Combined가")
    print("     naive보다 여전히 낫다면 -> 본격적 5-way x multi-seed W4A4 검증 가치 있음.")
    print("  W4A4에서 naive/Combined 둘 다 AP가 붕괴(랜덤 근처)하면 -> 지금 스킴")
    print("     (per-tensor activation, per-channel weight)으로는 4bit 비교가 무의미 ->")
    print("     스킴 개선(per-channel activation 등) 없이는 W8A8 범위로 스코프 유지.")


if __name__ == "__main__":
    main()
