"""
Baseline 종합 비교: AdaRound / QDrop / BRECQ vs Combined.

pipeline/ 디렉토리의 유일한 실행 스크립트. scripts/45_baseline_compare.py와
로직은 동일하고(그 스크립트로 낸 6-seed 결과가 논문/진행 문서의 근거), import
경로만 이 디렉토리 안의 harness.py/quant/*로 바뀌었다 -- pipeline/README.md
참고.

09_phase1_final_report_ko_v2.pdf(§1.3 평가 지표)의 표준 정의를 따라 확장:
  - AP(전체 + S/H_eval subset)
  - H_eval flip(우리가 만든 masked 버전 -- "H_eval을 안 쓰는 다른 사용자" 시나리오 진단용)
  - Top-1 flip(Phase 1 표준: masking 없이 FP confident anchor의 raw top-1 일치율.
    AP와 같은 배포 조건(전체 vocabulary)에서 결정 일치도를 직접 잼)
  - GT MRR / GT R@1(실제 COCO ground truth 기준. FP32를 pseudo-GT로 안 쓰고 진짜 정답 사용)
  - GT top-1 loss / gained(FP에서 정답이 1위였다가 양자화 후 잃은/얻은 수)
  - UPIR(calibration에서 안 본 prompt가 정답 위로 침입한 비율, GT가 S/H_cal일 때만 집계)
  - calibration 시간(wall-clock), 이론적 모델 크기(양자화 weight 8bit 기준)

GT 앵커 매칭: COCO 원본 GT 박스를 letterbox 변환으로 640 공간에 옮긴 뒤, 3개 FPN
level(80x80/40x40/20x20, stride 8/16/32)에서 그리드 셀 중심이 박스 안에 드는
후보를 모으고, 그중 FP32가 정답 class에 가장 confident한 anchor 하나를 GT의
대표 anchor로 선택한다(공식 TaskAlignedAssigner와 완전히 동일하지 않은 근사).

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python pipeline/run_comparison.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --calib 32 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import SimilarityHarness
from quant.quant_model import wrap_convs, calibrate
from quant.adaround import convert_to_adaround, optimize_adaround, AdaRoundQuantConv2d
from quant.fake_quant import QuantConv2d
from quant.brecq import optimize_brecq
from quant.promptcal import optimize_promptcal_scale_neighbor


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


def group_flip(fp_sims, q_sims, group_idx, conf=0.25):
    """우리가 만든 masked H_eval flip. group_idx(H_eval) 컬럼을 가린 뒤, FP
    full-top1이 group_idx에 속하는 confident anchor에서 K-only(가림) argmax의
    FP vs quant 불일치율 -- "H_eval을 안 쓰는 다른 사용자" 시나리오 진단용."""
    Gm = torch.zeros(80, dtype=torch.bool)
    Gm[group_idx] = True
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
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


def standard_flip(fp_sims, q_sims, conf=0.25):
    """Phase 1 표준 Top-1 flip: masking 없이, FP confident anchor에서 raw
    top-1(전체 80class)이 양자화 모델과 같은지. AP와 동일한 배포 조건(전체
    vocabulary)에서 결정 일치도를 직접 잰다."""
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


# ---------------------------------------------------------------------------
# GT 기반 지표: 실제 COCO annotation과 anchor 매칭
# ---------------------------------------------------------------------------

def load_gt_by_path(ann_path, img_paths):
    """COCO GT를 로드해서 img_path -> [(cls80, cx, cy, w, h)] (원본 이미지 px)로 반환."""
    from pycocotools.coco import COCO
    from ultralytics.data.converter import coco91_to_coco80_class
    coco = COCO(ann_path)
    map91to80 = coco91_to_coco80_class()
    fname_to_id = {info["file_name"]: img_id for img_id, info in coco.imgs.items()}
    out = {}
    for p in img_paths:
        fname = os.path.basename(p)
        img_id = fname_to_id.get(fname)
        boxes = []
        if img_id is not None:
            for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False)):
                cls80 = map91to80[a["category_id"] - 1]
                if cls80 is None:
                    continue
                x, y, w, h = a["bbox"]
                boxes.append((cls80, x + w / 2, y + h / 2, w, h))
        out[p] = boxes
    return out


def get_grid_specs(h_fp, sample_img):
    """SimilarityHarness의 level별 (H, W, stride)를 실제 forward로 확인."""
    h_fp.run_image(sample_img, -1)
    specs = []
    for i in sorted(h_fp._level_buf):
        B, P, H, W = h_fp._level_buf[i].shape
        specs.append((H, W))
    return specs


def build_gt_targets(fp_sims, img_paths, gt_by_path, grid_specs, imgsz):
    """각 이미지에 대해 [(anchor_idx, cls80), ...] 리스트를 만든다. GT 박스 중심이
    드는 grid cell 후보 중 FP32가 정답 class에 가장 confident한 anchor를 선택.
    fp_sims: probe 이미지별로 미리 계산해둔 FP sim 리스트."""
    offsets = []
    off = 0
    for (H, W) in grid_specs:
        stride = imgsz // W
        offsets.append((off, H, W, stride))
        off += H * W

    targets = []
    for i, p in enumerate(img_paths):
        sf = fp_sims[i]
        im0 = cv2.imread(p)
        h0, w0 = im0.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
        img_targets = []
        for (cls80, cx, cy, bw, bh) in gt_by_path.get(p, []):
            x0 = (cx - bw / 2) * r + left; x1 = (cx + bw / 2) * r + left
            y0 = (cy - bh / 2) * r + top;  y1 = (cy + bh / 2) * r + top
            candidates = []
            for (off0, H, W, s) in offsets:
                col0 = max(0, int(x0 / s - 0.5)); col1 = min(W - 1, int(x1 / s - 0.5) + 1)
                row0 = max(0, int(y0 / s - 0.5)); row1 = min(H - 1, int(y1 / s - 0.5) + 1)
                for row in range(row0, row1 + 1):
                    for col in range(col0, col1 + 1):
                        ccx, ccy = (col + 0.5) * s, (row + 0.5) * s
                        if x0 <= ccx <= x1 and y0 <= ccy <= y1:
                            candidates.append(off0 + row * W + col)
            if not candidates:
                continue
            best = max(candidates, key=lambda a: float(sf[a, cls80]))
            img_targets.append((best, cls80))
        targets.append(img_targets)
    return targets


def gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set):
    """GT MRR / GT R@1 / GT top-1 lost / gained / UPIR 계산."""
    recip_ranks = []
    lost = gained = 0
    upir_num = upir_den = 0
    for i in range(len(fp_sims)):
        sf = fp_sims[i]
        sq = q_sims[i]
        for (aidx, cls80) in gt_targets[i]:
            fp_s = sf[aidx]; q_s = sq[aidx]
            fp_rank = int((fp_s > fp_s[cls80]).sum()) + 1
            q_rank = int((q_s > q_s[cls80]).sum()) + 1
            recip_ranks.append(1.0 / q_rank)
            if fp_rank == 1 and q_rank != 1:
                lost += 1
            if fp_rank != 1 and q_rank == 1:
                gained += 1
            if fp_rank == 1 and cls80 not in H_eval_set:
                upir_den += 1
                if int(q_s.argmax()) in H_eval_set:
                    upir_num += 1
    n = len(recip_ranks)
    mrr = sum(recip_ranks) / max(n, 1)
    r1 = sum(1 for r in recip_ranks if r == 1.0) / max(n, 1)
    upir = upir_num / max(upir_den, 1) * 100
    return dict(mrr=mrr, r1=r1, lost=lost, gained=gained, upir=upir, n=n, upir_n=upir_den)


def quantized_weight_mib(model_module):
    """양자화 대상 conv weight 총량을 8bit(1 byte/element) 기준 이론 크기(MiB)로."""
    total = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            total += m.conv.weight.numel()
    return total / (1024 * 1024)


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
    ap.add_argument("--gt-ann", default=None,
                    help="기본값: {coco-root}/annotations/instances_val2017.json")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=500, help="flip/GT 측정용 probe 이미지 수")
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
    gt_ann = args.gt_ann or os.path.join(args.coco_root, "annotations", "instances_val2017.json")

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[determinism] torch.manual_seed={args.torch_seed}, cudnn.deterministic=True, cudnn.benchmark=False")

    names = load_coco_names()
    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_paths = imgs[:args.calib]
    probe_paths = imgs[args.calib:args.calib + args.eval]
    calib = [preprocess(p, args.imgsz, device) for p in calib_paths]
    probe = [preprocess(p, args.imgsz, device) for p in probe_paths]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_eval = perm[60:80].tolist()
    H_eval_set = set(H_eval)
    pidx = S

    print("[load] FP"); fp = build(YOLOWorld, args.model, names, device, calib, "fp")
    conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]
    models = {}
    calib_time = {}
    for mode in conditions:
        print(f"[build] {mode}")
        t0 = time.perf_counter()
        models[mode] = build(YOLOWorld, args.model, names, device, calib, mode, fp=fp,
                             iters=args.iters, pidx=pidx, lr=args.lr, k=args.k,
                             neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                             recon_iters_ada=args.recon_iters_ada,
                             recon_iters_strong=args.recon_iters_strong,
                             qdrop_prob=args.qdrop_prob)
        calib_time[mode] = time.perf_counter() - t0

    model_mib = quantized_weight_mib(models["adaround"].model)

    print(f"\n[gt] FP sim 1회 계산(baseline 무관, 재사용) + {gt_ann} anchor 매칭 (probe {len(probe_paths)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    grid_specs = get_grid_specs(h_fp, probe[0])
    fp_sims = [h_fp.run_image(t, i).sim for i, t in enumerate(probe)]
    h_fp.close()
    gt_by_path = load_gt_by_path(gt_ann, probe_paths)
    gt_targets = build_gt_targets(fp_sims, probe_paths, gt_by_path, grid_specs, args.imgsz)
    n_gt = sum(len(t) for t in gt_targets)
    print(f"  grid_specs={grid_specs}, 매칭된 GT anchor 수={n_gt}")

    print(f"\n[flip] H_eval flip(masked) + Top-1 flip(표준, unmasked) 측정")
    flip_results = {}
    std_flip_results = {}
    gt_results = {}
    for mode in conditions:
        h_q = SimilarityHarness(models[mode].model, device=device)
        q_sims = [h_q.run_image(t, i).sim for i, t in enumerate(probe)]
        h_q.close()
        flip_results[mode], n1 = group_flip(fp_sims, q_sims, H_eval)
        std_flip_results[mode], n2 = standard_flip(fp_sims, q_sims)
        gt_results[mode] = gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set)
        print(f"  {mode:>10}: H_eval_flip={flip_results[mode]:.2f}%(n={n1})  "
              f"Top1_flip={std_flip_results[mode]:.2f}%(n={n2})  "
              f"GT_MRR={gt_results[mode]['mrr']:.4f}  GT_R@1={gt_results[mode]['r1']:.4f}  "
              f"lost={gt_results[mode]['lost']}  gained={gt_results[mode]['gained']}  "
              f"UPIR={gt_results[mode]['upir']:.2f}%(n={gt_results[mode]['upir_n']})")

    results = {}
    print(f"\n[ap] FP32 ..."); results["FP32"] = measure_ap(fp, args.data, args.imgsz, args.device)
    for mode in conditions:
        print(f"[ap] {mode} ...")
        results[mode] = measure_ap(models[mode], args.data, args.imgsz, args.device)

    print("\n" + "=" * 100)
    print(f" 전체 80-class AP (seed {args.seed})")
    print("=" * 100)
    print(f"{'':>10} | {'mAP50-95':>9} | {'mAP50':>9}")
    for name in ["FP32"] + conditions:
        (m, m50), _ = results[name]
        print(f"{name:>10} | {m:>9.2f} | {m50:>9.2f}")

    print("\n" + "=" * 100)
    print(f" S vs H_eval subset mAP + flip 지표 (seed {args.seed})")
    print("=" * 100)
    print(f"{'':>10} | {'S mAP':>8} | {'H_eval mAP':>10} | {'Heval_flip':>10} | {'Top1_flip':>9}")
    for name in conditions:
        _, pc = results[name]
        print(f"{name:>10} | {subset_map(pc, S):>8.2f} | {subset_map(pc, H_eval):>10.2f} | "
              f"{flip_results[name]:>9.2f}% | {std_flip_results[name]:>8.2f}%")

    print("\n" + "=" * 100)
    print(f" GT 기반 지표 (seed {args.seed}, GT anchor n={n_gt})")
    print("=" * 100)
    print(f"{'':>10} | {'GT_MRR':>7} | {'GT_R@1':>7} | {'lost':>6} | {'gained':>6} | {'UPIR':>7}")
    for name in conditions:
        g = gt_results[name]
        print(f"{name:>10} | {g['mrr']:>7.4f} | {g['r1']:>7.4f} | {g['lost']:>6} | "
              f"{g['gained']:>6} | {g['upir']:>6.2f}%")

    print("\n" + "=" * 100)
    print(f" 비용/크기")
    print("=" * 100)
    print(f"  양자화 weight 이론 크기: {model_mib:.2f} MiB (모든 W8A8 방법 공통, naive 제외 동일 구조)")
    print(f"  {'':>10} | {'calib 시간(s)':>13}")
    for name in conditions:
        print(f"  {name:>10} | {calib_time[name]:>13.1f}")

    print("\n판정:")
    print("  Combined H_eval mAP > QDrop/BRECQ H_eval mAP -> AdaRound뿐 아니라 더 강한")
    print("     reconstruction-based baseline까지 이긴다.")
    print("  Top1_flip(표준, AP와 같은 배포 조건)이 Combined에서도 낮으면 -> H_eval flip에서")
    print("     본 반전이 masking 정의 때문이었다는 게 확인됨(진짜 결정 일치도는 개선).")
    print("  GT_MRR/R@1이 Combined에서 가장 높고 UPIR이 가장 낮으면 -> 실제 정답 기준으로도")
    print("     일관되게 최선(논문 핵심 주장 가장 강하게 뒷받침).")


if __name__ == "__main__":
    main()
