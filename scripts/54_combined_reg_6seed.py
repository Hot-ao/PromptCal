"""
재설계(scale_reg_weight 정규화, 09-07 §7.6~7.8)를 6-seed로 검증. COCO-80과
LVIS-1203(이식) 전체 지표를 한 스크립트에서 같이 잰다.

baseline(naive/AdaRound/QDrop/BRECQ)은 다시 안 돌린다 -- pidx(S)/H_eval을
전혀 안 쓰고 probe 이미지 선택도 seed와 무관해서 seed에 대해 상수임이 이미
확인됨(COCO-80: results_v1/diag/45_metrics_seed*_full.txt 전부 동일,
LVIS-이식: 51번 seed2 결과가 대표값). Combined(scale_reg_weight 적용판)만
6-seed로 새로 빌드해서, 그 baseline 상수들과 나란히 비교한다.

측정(COCO-80): AP(전체+S/H_eval subset), Top1_flip(표준)/H_eval_flip(masked),
GT_MRR/R@1/lost/gained, UPIR, calib 시간.
측정(LVIS-1203, 재학습 없이 이식): AP(진짜 GT, lvis-api), Top1_flip(표준),
GT_MRR/R@1/lost/gained (H_eval_flip/UPIR/subset AP는 구조상 이식에 적용 불가,
51번과 동일 이유).

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/54_combined_reg_6seed.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml --lvis-ann /data/taeho/lvis_datasets/annotations/lvis_v1_val.json \
        --calib 32 --eval 500 --scale-reg-weight 200.0 --seeds 0 1 2 4 5 7 --device 0
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
from src.quant.promptcal import optimize_promptcal_scale_neighbor


def load_names(which):
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / f"{which}.yaml"
    d = yaml.safe_load(open(p))
    n = d["names"]
    return [n[i] for i in range(len(n))]


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


def switch_vocab(model, names, device):
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.to(device).eval()


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
    Gm = torch.zeros(80, dtype=torch.bool)
    Gm[group_idx] = True
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
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
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        if conf_m.sum() == 0:
            continue
        idx = conf_m.nonzero(as_tuple=True)[0]
        c_q = sq.argmax(-1)
        fl += int((c_fp[idx] != c_q[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def load_coco_gt_by_path(ann_path, img_paths):
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


def gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set=None):
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
            if H_eval_set is not None and fp_rank == 1 and cls not in H_eval_set:
                upir_den += 1
                if int(q_s.argmax()) in H_eval_set:
                    upir_num += 1
    n = len(recip_ranks)
    mrr = sum(recip_ranks) / max(n, 1)
    r1 = sum(1 for r in recip_ranks if r == 1.0) / max(n, 1)
    out = dict(mrr=mrr, r1=r1, lost=lost, gained=gained, n=n)
    if H_eval_set is not None:
        out["upir"] = upir_num / max(upir_den, 1) * 100
        out["upir_n"] = upir_den
    return out


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


def quantized_weight_mib(model_module):
    total = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            total += m.conv.weight.numel()
    return total / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--gt-ann", default=None)
    ap.add_argument("--lvis-ann", default="/data/taeho/lvis_datasets/annotations/lvis_v1_val.json")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--scale-reg-weight", type=float, default=200.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 4, 5, 7])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    gt_ann = args.gt_ann or os.path.join(args.coco_root, "annotations", "instances_val2017.json")

    from lvis import LVIS
    print(f"[lvis] loading GT {args.lvis_ann}")
    lvis_gt = LVIS(args.lvis_ann)
    lvis_img_ids = set(lvis_gt.get_img_ids())

    coco = load_names("coco")
    lvis_names = load_names("lvis")
    from ultralytics import YOLOWorld
    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_paths = all_imgs[:args.calib]
    probe_paths = all_imgs[args.calib:args.calib + args.eval]
    lvis_probe_paths, lvis_probe_ids = [], []
    for p in probe_paths:
        iid = int(os.path.basename(p).split(".")[0])
        if iid in lvis_img_ids:
            lvis_probe_paths.append(p); lvis_probe_ids.append(iid)
    print(f"[filter] LVIS probe {len(probe_paths)}장 중 {len(lvis_probe_paths)}장만 사용")

    coco_gt_by_path = load_coco_gt_by_path(gt_ann, probe_paths)
    lvis_gt_by_path = load_lvis_gt_by_path(lvis_gt, lvis_probe_paths, lvis_probe_ids)

    all_results = {}
    for seed in args.seeds:
        print(f"\n{'#'*30} seed {seed} {'#'*30}")
        torch.manual_seed(args.torch_seed)
        torch.cuda.manual_seed_all(args.torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        calib = [preprocess(p, args.imgsz, device) for p in calib_paths]
        probe = [preprocess(p, args.imgsz, device) for p in probe_paths]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_eval = perm[60:80].tolist()
        H_eval_set = set(H_eval)

        print("[build] FP (COCO-80)")
        fp = YOLOWorld(args.model); fp.set_classes(coco); fp.fuse(); fp.model.to(device).eval()

        print(f"[build] combined (scale_reg_weight={args.scale_reg_weight})")
        t0 = time.perf_counter()
        cb = YOLOWorld(args.model); cb.set_classes(coco); cb.fuse()
        wrap_convs(cb.model, 8, 8); cb.model.to(device).eval()
        calibrate(cb.model, calib, device=device)
        convert_to_adaround(cb.model)
        optimize_adaround(cb.model, fp.model, calib, device, iters=args.recon_iters_ada, verbose=False)
        optimize_promptcal_scale_neighbor(cb.model, fp.model, calib, device, S, iters=args.iters,
                                          lr=args.lr, k=args.k, neighbor_k=args.neighbor_k,
                                          neighbor_weight=args.neighbor_weight,
                                          asymmetric=True, scale_reg_weight=args.scale_reg_weight,
                                          verbose=False)
        calib_time = time.perf_counter() - t0
        model_mib = quantized_weight_mib(cb.model)

        # ---------------- COCO-80 지표 ----------------
        h_fp = SimilarityHarness(fp.model, device=device)
        grid_specs = get_grid_specs(h_fp, probe[0])
        fp_sims = [h_fp.run_image(t, i).sim for i, t in enumerate(probe)]
        h_fp.close()
        coco_gt_targets = build_gt_targets(fp_sims, probe_paths, coco_gt_by_path, grid_specs, args.imgsz)

        h_q = SimilarityHarness(cb.model, device=device)
        q_sims = [h_q.run_image(t, i).sim for i, t in enumerate(probe)]
        h_q.close()
        heval_flip, n1 = group_flip(fp_sims, q_sims, H_eval)
        top1_flip, n2 = standard_flip(fp_sims, q_sims)
        coco_gt_res = gt_metrics_for_method(fp_sims, q_sims, coco_gt_targets, H_eval_set)

        (coco_ap, coco_ap50), coco_pc = measure_ap(cb, args.data, args.imgsz, args.device)
        s_ap = subset_map(coco_pc, S); heval_ap = subset_map(coco_pc, H_eval)

        print(f"  [COCO-80] AP={coco_ap:.2f} S_AP={s_ap:.2f} H_eval_AP={heval_ap:.2f} "
              f"Heval_flip={heval_flip:.2f}% Top1_flip={top1_flip:.2f}% "
              f"GT_MRR={coco_gt_res['mrr']:.4f} GT_R@1={coco_gt_res['r1']:.4f} "
              f"lost={coco_gt_res['lost']} gained={coco_gt_res['gained']} UPIR={coco_gt_res['upir']:.2f}%")

        # ---------------- LVIS-1203 지표 (재학습 없이 이식) ----------------
        switch_vocab(fp, lvis_names, device)
        switch_vocab(cb, lvis_names, device)

        h_fp = SimilarityHarness(fp.model, device=device)
        lvis_grid_specs = get_grid_specs(h_fp, probe[0])
        lvis_probe_tensors = [preprocess(p, args.imgsz, device) for p in lvis_probe_paths]
        lvis_fp_sims = [h_fp.run_image(t, i).sim for i, t in enumerate(lvis_probe_tensors)]
        h_fp.close()
        lvis_gt_targets = build_gt_targets(lvis_fp_sims, lvis_probe_paths, lvis_gt_by_path,
                                           lvis_grid_specs, args.imgsz)

        h_q = SimilarityHarness(cb.model, device=device)
        lvis_q_sims = [h_q.run_image(t, i).sim for i, t in enumerate(lvis_probe_tensors)]
        h_q.close()
        lvis_top1_flip, n3 = standard_flip(lvis_fp_sims, lvis_q_sims)
        lvis_gt_res = gt_metrics_for_method(lvis_fp_sims, lvis_q_sims, lvis_gt_targets)

        preds = predict_lvis_results(cb, lvis_probe_paths, lvis_probe_ids, args.imgsz, device)
        lvis_ap_res = run_lvis_eval(lvis_gt, preds, lvis_probe_ids)

        print(f"  [LVIS] AP={lvis_ap_res.get('AP',0):.4f} Top1_flip={lvis_top1_flip:.2f}% "
              f"GT_MRR={lvis_gt_res['mrr']:.4f} GT_R@1={lvis_gt_res['r1']:.4f} "
              f"lost={lvis_gt_res['lost']} gained={lvis_gt_res['gained']}")

        all_results[seed] = dict(
            coco_ap=coco_ap, s_ap=s_ap, heval_ap=heval_ap, heval_flip=heval_flip,
            top1_flip=top1_flip, coco_gt=coco_gt_res, calib_time=calib_time, model_mib=model_mib,
            lvis_ap=lvis_ap_res.get("AP", 0.0), lvis_ap50=lvis_ap_res.get("AP50", 0.0),
            lvis_top1_flip=lvis_top1_flip, lvis_gt=lvis_gt_res,
        )

    def agg(key, sub=None):
        vals = [all_results[s][key] if sub is None else all_results[s][key][sub] for s in args.seeds]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n" + "=" * 100)
    print(f" Combined(scale_reg_weight={args.scale_reg_weight}) 6-seed 집계 ({args.seeds})")
    print("=" * 100)
    for label, key, sub in [
        ("COCO-80 AP", "coco_ap", None), ("S_AP", "s_ap", None), ("H_eval_AP", "heval_ap", None),
        ("H_eval_flip%", "heval_flip", None), ("Top1_flip%", "top1_flip", None),
        ("GT_MRR", "coco_gt", "mrr"), ("GT_R@1", "coco_gt", "r1"),
        ("lost", "coco_gt", "lost"), ("gained", "coco_gt", "gained"), ("UPIR%", "coco_gt", "upir"),
        ("LVIS AP", "lvis_ap", None), ("LVIS Top1_flip%", "lvis_top1_flip", None),
        ("LVIS GT_MRR", "lvis_gt", "mrr"), ("LVIS GT_R@1", "lvis_gt", "r1"),
        ("LVIS lost", "lvis_gt", "lost"), ("LVIS gained", "lvis_gt", "gained"),
        ("calib_time(s)", "calib_time", None),
    ]:
        m, sd = agg(key, sub)
        print(f"  {label:>18}: {m:.4f} ± {sd:.4f}")

    print("\n참고 baseline(seed 무관 상수, results_v1/diag/45_metrics_seed2·51_lvis_transplant_seed2 기준):")
    print("  COCO-80 AP    naive=35.06 adaround=35.05 qdrop=35.08 brecq=35.01")
    print("  LVIS AP       naive=0.2175 adaround=0.2232 qdrop=0.2202 brecq=0.2217")
    print("  COCO-80 UPIR  naive=0.232 adaround=0.218 qdrop=0.207 brecq=0.165 (원조합 combined(rw=0)=0.142)")


if __name__ == "__main__":
    main()
