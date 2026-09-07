"""
국면 I 최종: LVIS-1203을 native vocabulary로 놓고 처음부터 calibration/S-H
분할을 다시 해서 Combined를 검증 (09-07). 49번(zero-shot 이식, COCO-80으로
학습한 걸 LVIS에 재학습 없이 꽂음)과 달리, 여기서는 애초에 LVIS-1203 자체를
대상으로 S/H_cal/H_eval을 나누고 그 위에서 calibration+학습+평가를 전부 한다.
"COCO-80에서 통하던 설계 원리(margin+neighbor 보존)가 훨씬 촘촘한 vocabulary
에서도 통하는가"를 재학습 없이 이식하는 것보다 공정하게 묻는 버전.

class pool: LVIS 1203개 중 frequency='f'(frequent, 405개)만 사용. rare(337개,
val 이미지 median 1장)/common(461개, median 7장)은 calib 이미지 몇십~백여 장
으로는 confident anchor가 거의 안 걸려서 margin_loss 신호 자체가 없다 --
COCO-80의 80개 class가 전부 "흔한 물체"였던 것과 맞추기 위해 frequent만 씀.
neighbor 탐색은 (COCO-80 때처럼) 전체 1203-class text embedding에서 하므로
S의 실제 최근접 이웃이 rare/common class여도 그대로 잡힌다.

S(pool의 50%)/H_cal(25%)/H_eval(25%) 분할, COCO-80과 동일 비율.

평가지표는 45/47/49번을 합쳐 전부: AP(전체+S/H_eval subset, 실제 LVIS GT+lvis-api
LVISEval) + 표준 Top1_flip + H_eval_flip(masked) + GT_MRR/R@1/lost/gained/UPIR
(실제 LVIS GT 앵커 매칭) + calib 시간/모델 크기.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/50_lvis_native.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --lvis-ann /data/taeho/lvis_datasets/annotations/lvis_v1_val.json \
        --calib 128 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys, time
import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround, AdaRoundQuantConv2d
from src.quant.fake_quant import QuantConv2d
from src.quant.brecq import optimize_brecq
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
                                          asymmetric=True, verbose=False)
    return m


# ---------------------------------------------------------------------------
# flip 지표(45/47과 동일 정의, P=1203로 일반화)
# ---------------------------------------------------------------------------

def group_flip(fp_sims, q_sims, group_idx, P, conf=0.25):
    Gm = torch.zeros(P, dtype=torch.bool)
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
# 실제 LVIS GT 기반 앵커 매칭 + GT_MRR/R@1/lost/gained/UPIR (45번의 COCO 버전을
# LVIS GT로 일반화: category_id -> 0-index는 -1만 하면 됨, coco91->80 remap 불필요)
# ---------------------------------------------------------------------------

def load_lvis_gt_by_path(lvis_gt, img_paths, img_ids):
    ann_by_img = {}
    for a in lvis_gt.dataset["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    out = {}
    for p, iid in zip(img_paths, img_ids):
        boxes = []
        for a in ann_by_img.get(iid, []):
            cls = a["category_id"] - 1
            x, y, w, h = a["bbox"]
            boxes.append((cls, x + w / 2, y + h / 2, w, h))
        out[p] = boxes
    return out


def get_grid_specs(h_fp, sample_img):
    h_fp.run_image(sample_img, -1)
    specs = []
    for i in sorted(h_fp._level_buf):
        B, P, H, W = h_fp._level_buf[i].shape
        specs.append((H, W))
    return specs


def build_gt_targets(fp_sims, img_paths, gt_by_path, grid_specs, imgsz):
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
        for (cls, cx, cy, bw, bh) in gt_by_path.get(p, []):
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
            best = max(candidates, key=lambda a: float(sf[a, cls]))
            img_targets.append((best, cls))
        targets.append(img_targets)
    return targets


def gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set):
    recip_ranks = []
    lost = gained = 0
    upir_num = upir_den = 0
    for i in range(len(fp_sims)):
        sf = fp_sims[i]; sq = q_sims[i]
        for (aidx, cls) in gt_targets[i]:
            fp_s = sf[aidx]; q_s = sq[aidx]
            fp_rank = int((fp_s > fp_s[cls]).sum()) + 1
            q_rank = int((q_s > q_s[cls]).sum()) + 1
            recip_ranks.append(1.0 / q_rank)
            if fp_rank == 1 and q_rank != 1:
                lost += 1
            if fp_rank != 1 and q_rank == 1:
                gained += 1
            if fp_rank == 1 and cls not in H_eval_set:
                upir_den += 1
                if int(q_s.argmax()) in H_eval_set:
                    upir_num += 1
    n = len(recip_ranks)
    mrr = sum(recip_ranks) / max(n, 1)
    r1 = sum(1 for r in recip_ranks if r == 1.0) / max(n, 1)
    upir = upir_num / max(upir_den, 1) * 100
    return dict(mrr=mrr, r1=r1, lost=lost, gained=gained, upir=upir, n=n, upir_n=upir_den)


def quantized_weight_mib(model_module):
    total = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            total += m.conv.weight.numel()
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# 실제 LVIS AP (전체 + S/H_eval subset) -- lvis-api LVISEval
# ---------------------------------------------------------------------------

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


def run_lvis_eval_full(lvis_gt, results, img_ids):
    from lvis import LVISEval, LVISResults
    if not results:
        return None
    lvis_dt = LVISResults(lvis_gt, results, max_dets=300)
    ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
    ev.params.img_ids = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def subset_ap(ev, cat_ids_1indexed):
    """ev.eval['precision']: [T,R,K,A] (K=ev.params.cat_ids 순서). 주어진
    category_id(1-indexed) 부분집합만 평균낸 AP(all-area, all-IoU)."""
    if ev is None:
        return float("nan")
    cat_idx = {c: i for i, c in enumerate(ev.params.cat_ids)}
    idxs = [cat_idx[c] for c in cat_ids_1indexed if c in cat_idx]
    if not idxs:
        return float("nan")
    aidx = ev.params.area_rng_lbl.index("all")
    s = ev.eval["precision"][:, :, idxs, aidx]
    return float(np.mean(s[s > -1])) if (s > -1).any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--lvis-ann", default="/data/taeho/lvis_datasets/annotations/lvis_v1_val.json")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--recon-iters-strong", type=int, default=2000)
    ap.add_argument("--qdrop-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--conf-thres", type=float, default=0.25)
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
    cats = sorted(lvis_gt.dataset["categories"], key=lambda c: c["id"])
    freq_pool = [c["id"] - 1 for c in cats if c["frequency"] == "f"]   # 0-indexed positions
    print(f"[pool] frequent bucket {len(freq_pool)}개 class를 S/H_cal/H_eval 분할 대상으로 사용")

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
    probe = [preprocess(p, args.imgsz, device) for p in probe_paths]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(freq_pool))
    n = len(freq_pool)
    n_s, n_hcal = n // 2, n // 4
    S = [freq_pool[i] for i in perm[:n_s]]
    H_cal = [freq_pool[i] for i in perm[n_s:n_s + n_hcal]]
    H_eval = [freq_pool[i] for i in perm[n_s + n_hcal:]]
    H_eval_set = set(H_eval)
    print(f"[split] S={len(S)} H_cal={len(H_cal)} H_eval={len(H_eval)} (seed {args.seed})")

    print("[build] FP (LVIS-1203 native vocab)")
    fp = build(YOLOWorld, args.model, lvis_names, device, calib, "fp")
    conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]
    models, calib_time = {}, {}
    for mode in conditions:
        print(f"[build] {mode}")
        t0 = time.perf_counter()
        models[mode] = build(YOLOWorld, args.model, lvis_names, device, calib, mode, fp=fp,
                             iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                             neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                             recon_iters_ada=args.recon_iters_ada,
                             recon_iters_strong=args.recon_iters_strong,
                             qdrop_prob=args.qdrop_prob)
        calib_time[mode] = time.perf_counter() - t0
    model_mib = quantized_weight_mib(models["adaround"].model)

    print(f"\n[gt] FP sim 계산 + LVIS GT anchor 매칭 ({len(probe_paths)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    grid_specs = get_grid_specs(h_fp, probe[0])
    fp_sims = [h_fp.run_image(t, i).sim for i, t in enumerate(probe)]
    h_fp.close()
    gt_by_path = load_lvis_gt_by_path(lvis_gt, probe_paths, probe_ids)
    gt_targets = build_gt_targets(fp_sims, probe_paths, gt_by_path, grid_specs, args.imgsz)
    n_gt = sum(len(t) for t in gt_targets)
    print(f"  매칭된 GT anchor 수={n_gt}")

    flip_results, std_flip_results, gt_results, ap_results = {}, {}, {}, {}
    for label, m in [("FP32", fp)] + [(c, models[c]) for c in conditions]:
        h_q = SimilarityHarness(m.model, device=device)
        q_sims = [h_q.run_image(t, i).sim for i, t in enumerate(probe)]
        h_q.close()
        if label != "FP32":
            flip_results[label], n1 = group_flip(fp_sims, q_sims, H_eval, P=1203)
            std_flip_results[label], n2 = standard_flip(fp_sims, q_sims)
            gt_results[label] = gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set)
            print(f"  {label:>10}: H_eval_flip={flip_results[label]:.2f}%(n={n1})  "
                  f"Top1_flip={std_flip_results[label]:.2f}%(n={n2})  "
                  f"GT_MRR={gt_results[label]['mrr']:.4f}  GT_R@1={gt_results[label]['r1']:.4f}  "
                  f"lost={gt_results[label]['lost']}  gained={gt_results[label]['gained']}  "
                  f"UPIR={gt_results[label]['upir']:.2f}%(n={gt_results[label]['upir_n']})")

        print(f"[predict+AP] {label} ...")
        preds = predict_lvis_results(m, probe_paths, probe_ids, args.imgsz, device)
        ev = run_lvis_eval_full(lvis_gt, preds, probe_ids)
        ap_results[label] = dict(
            overall=ev.get_results() if ev else {},
            s=subset_ap(ev, [c + 1 for c in S]),
            heval=subset_ap(ev, [c + 1 for c in H_eval]),
        )
        print(f"  {label}: AP={ap_results[label]['overall'].get('AP', float('nan')):.4f}  "
              f"S_AP={ap_results[label]['s']:.4f}  H_eval_AP={ap_results[label]['heval']:.4f}")

    print("\n" + "=" * 100)
    print(f" LVIS-native AP (전체/S/H_eval subset, real GT) -- seed {args.seed}, {len(probe_paths)}장")
    print("=" * 100)
    print(f"{'':>10} | {'AP':>7} | {'AP50':>7} | {'S_AP':>7} | {'H_eval_AP':>9}")
    for label in ["FP32"] + conditions:
        r = ap_results[label]
        o = r["overall"]
        print(f"{label:>10} | {o.get('AP',0):>7.4f} | {o.get('AP50',0):>7.4f} | "
              f"{r['s']:>7.4f} | {r['heval']:>9.4f}")

    print("\n" + "=" * 100)
    print(f" flip / GT 지표 (real GT) -- seed {args.seed}")
    print("=" * 100)
    print(f"{'':>10} | {'Heval_flip':>10} | {'Top1_flip':>9} | {'GT_MRR':>7} | {'GT_R@1':>7} | "
          f"{'lost':>5} | {'gained':>6} | {'UPIR':>7}")
    for mode in conditions:
        g = gt_results[mode]
        print(f"{mode:>10} | {flip_results[mode]:>9.2f}% | {std_flip_results[mode]:>8.2f}% | "
              f"{g['mrr']:>7.4f} | {g['r1']:>7.4f} | {g['lost']:>5} | {g['gained']:>6} | {g['upir']:>6.2f}%")

    print("\n" + "=" * 100)
    print(" 비용/크기")
    print("=" * 100)
    print(f"  양자화 weight 이론 크기: {model_mib:.2f} MiB")
    for mode in conditions:
        print(f"  {mode:>10} | calib {calib_time[mode]:>8.1f}s")

    print("\n판정:")
    print("  Combined의 S/H_eval AP·Top1_flip·UPIR이 baseline보다 나으면 -> LVIS를")
    print("     native로 놓고 처음부터 계산해도 설계 원리가 통한다(재설계 불필요,")
    print("     49번의 문제는 '재학습 없는 이식'이라는 시나리오 자체의 한계였음).")
    print("  여기서도 Combined가 최악이면 -> 촘촘한 vocabulary에서는 방법 설계")
    print("     자체(margin+neighbor 보존)가 원래 안 통한다는 뜻 -> 재설계 필요.")


if __name__ == "__main__":
    main()
