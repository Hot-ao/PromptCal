"""
Baseline 비교: AdaRound / QDrop / BRECQ vs Combined (09-06, 44번 다음 단계).

지금까지는 AdaRound 하나만 baseline으로 썼다. 논문 Related Work(§2.2)에 언급된
QDrop(activation drop으로 강건한 rounding), BRECQ(block 단위 joint reconstruction)
는 AdaRound보다 강한 reconstruction-based PTQ로 알려져 있다 -- 이들과 비교해도
Combined(AdaRound weight + asymmetric neighbor scale)가 여전히 이기는지 확인한다.
src/quant/brecq.py, src/quant/adaround.py(qdrop_prob)에 이미 구현돼 있음
(scripts/10_qdrop.py, 12_brecq.py에서 이미 검증된 코드 재사용).

비교:
  FP32 / naive / AdaRound / QDrop / BRECQ / Combined(AdaRound weight+asym neighbor scale)
  각각 AP(전체+S/H_eval subset)와 H_eval flip을 함께 측정(논문 §5.2 Main Results:
  "AP + representative semantic metrics"). flip은 AP 측정용 model.val()과 무관하게
  SimilarityHarness로 cv4 유사도 행렬을 직접 뽑아 계산한다(--eval 개수만큼 probe
  이미지 필요, calib 이후 이미지 사용).

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/45_baseline_compare.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --calib 32 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.brecq import optimize_brecq
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
    overall = float(metrics.box.map) * 100, float(metrics.box.map50) * 100
    per_class = dict(zip(metrics.box.ap_class_index.tolist(),
                         (metrics.box.maps if hasattr(metrics.box, "maps") else metrics.box.all_ap[:, 0]).tolist()))
    return overall, per_class


def subset_map(per_class, idx, scale=100.0):
    vals = [per_class[i] for i in idx if i in per_class]
    return float(np.mean(vals)) * scale if vals else float("nan")


def group_flip(h_fp, h_q, imgs, group_idx, conf=0.25):
    """H_eval flip: group_idx(H_eval) 컬럼을 가린 뒤, FP full-top1이 group_idx에
    속하는 confident anchor에서 K-only(가림) argmax의 FP vs quant 불일치율."""
    Gm = torch.zeros(80, dtype=torch.bool)
    Gm[group_idx] = True
    tot = fl = 0
    for i, t in enumerate(imgs):
        sf = h_fp.run_image(t, i).sim
        sq = h_q.run_image(t, i).sim
        prob = sf.sigmoid()
        mp, c_fp = prob.max(-1)
        conf_m = mp > conf
        target = conf_m & Gm[c_fp]
        if target.sum() == 0:
            continue
        idx = target.nonzero(as_tuple=True)[0]
        fp_K = sf.clone(); fp_K[:, Gm] = -1e9
        q_K = sq.clone();  q_K[:, Gm] = -1e9
        fl += int((fp_K.argmax(-1)[idx] != q_K.argmax(-1)[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0,
          recon_iters_ada=1000, recon_iters_strong=2000, qdrop_prob=0.5):
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
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
    elif mode == "qdrop":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_strong,
                          qdrop_prob=qdrop_prob, verbose=False)
    elif mode == "brecq":
        convert_to_adaround(m.model)
        optimize_brecq(m.model, fp.model, calib, device, iters=recon_iters_strong, verbose=False)
    elif mode == "combined":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
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
    ap.add_argument("--eval", type=int, default=500, help="H_eval flip 측정용 probe 이미지 수")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--recon-iters-strong", type=int, default=2000)
    ap.add_argument("--qdrop-prob", type=float, default=0.5)
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
    probe = [preprocess(p, args.imgsz, device) for p in imgs[args.calib:args.calib + args.eval]]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_eval = perm[60:80].tolist()
    pidx = S

    print("[load] FP"); fp = build(YOLOWorld, args.model, names, device, calib, "fp")
    conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]
    models = {}
    for mode in conditions:
        print(f"[build] {mode}")
        models[mode] = build(YOLOWorld, args.model, names, device, calib, mode, fp=fp,
                             iters=args.iters, pidx=pidx, lr=args.lr, k=args.k,
                             neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                             recon_iters_ada=args.recon_iters_ada,
                             recon_iters_strong=args.recon_iters_strong,
                             qdrop_prob=args.qdrop_prob)

    print(f"\n[flip] H_eval flip 측정 (probe {len(probe)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    flip_results = {}
    for mode in conditions:
        h_q = SimilarityHarness(models[mode].model, device=device)
        flip_results[mode], n = group_flip(h_fp, h_q, probe, H_eval)
        h_q.close()
        print(f"  {mode:>10}: H_eval flip={flip_results[mode]:.2f}% (n={n})")
    h_fp.close()

    results = {}
    print(f"\n[ap] FP32 ..."); results["FP32"] = measure_ap(fp, args.data, args.imgsz, args.device)
    for mode in conditions:
        print(f"[ap] {mode} ...")
        results[mode] = measure_ap(models[mode], args.data, args.imgsz, args.device)

    print("\n" + "=" * 80)
    print(f" 전체 80-class AP (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>10} | {'mAP50-95':>9} | {'mAP50':>9}")
    for name in ["FP32"] + conditions:
        (m, m50), _ = results[name]
        print(f"{name:>10} | {m:>9.2f} | {m50:>9.2f}")

    print("\n" + "=" * 80)
    print(f" S(계산에 씀, 40) vs H_eval(전혀 안 씀, 20) subset mAP50-95 + H_eval flip (seed {args.seed})")
    print("=" * 80)
    print(f"{'':>10} | {'S mAP':>8} | {'H_eval mAP':>10} | {'H_eval flip':>11}")
    for name in conditions:
        _, pc = results[name]
        print(f"{name:>10} | {subset_map(pc, S):>8.2f} | {subset_map(pc, H_eval):>10.2f} | "
              f"{flip_results[name]:>10.2f}%")

    print("\n판정:")
    print("  Combined H_eval mAP > QDrop/BRECQ H_eval mAP -> AdaRound뿐 아니라 더 강한")
    print("     reconstruction-based baseline까지 이긴다(논문 주장 훨씬 강해짐).")
    print("  Combined H_eval mAP < QDrop/BRECQ H_eval mAP -> 더 강한 reconstruction")
    print("     baseline에는 못 미침 -> 그 baseline의 weight rounding 위에 우리 scale")
    print("     tuning을 얹는 조합을 다음에 시도해야 함.")


if __name__ == "__main__":
    main()
