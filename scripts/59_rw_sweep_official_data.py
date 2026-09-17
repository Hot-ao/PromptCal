"""
scale_reg_weight 재스윕 (09-08) -- official-data 설정(calib=train2017 256장,
COCO-80 평가=val2017 전체, LVIS 평가=공식 minival)에서 rw=20이 여전히 최적
근방인지 확인. rw=20은 이전(calib=32, val2017 슬라이스) 스케일에서 고른
값이라, 새 데이터 스케일에서 재검증 없이 그대로 쓰는 건 근거가 약함.

naive/AdaRound/QDrop/BRECQ는 rw에 의존하지 않고 이미
`results_v1/diag/58_official_data_seed0.log`에 seed=0 기준 값이 있으므로
다시 빌드하지 않는다(이 스크립트도 --seed 0 고정 -- S/H_eval 분할을 그
로그와 맞추기 위함). Combined만 여러 scale_reg_weight로 빌드해서 비교한다.

스윕은 1 seed로 충분(방법론 합의, 09-08) -- 여기서 트렌드를 보고 승자 하나를
고른 뒤, 그 값만 scripts/58로 6-seed 최종 검증한다.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/59_rw_sweep_official_data.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml \
        --lvis-ann /data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json \
        --calib 256 --rw-list 10,20,30,50 --seed 0 --device 0
"""
import argparse, glob, os, sys, time
import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround, AdaRoundQuantConv2d, free_cpu_mem
from src.quant.fake_quant import QuantConv2d
from src.quant.promptcal import optimize_promptcal_scale_neighbor

# seed=0 기준 이미 확보된 baseline 참고값 (results_v1/diag/58_official_data_seed0.log)
BASELINE_REF = {
    "naive":    dict(coco_ap=33.54, lvis_ap=0.1264, heval_flip=10.51, top1_flip=0.95,
                     upir=0.39, lost=432, lvis_flip=6.48, lvis_lost=1224),
    "adaround": dict(coco_ap=33.24, lvis_ap=0.1242, heval_flip=9.20, top1_flip=0.74,
                     upir=0.26, lost=349, lvis_flip=5.67, lvis_lost=1212),
    "qdrop":    dict(coco_ap=33.30, lvis_ap=0.1235, heval_flip=9.25, top1_flip=0.72,
                     upir=0.33, lost=384, lvis_flip=5.71, lvis_lost=1192),
    "brecq":    dict(coco_ap=33.30, lvis_ap=0.1200, heval_flip=8.99, top1_flip=0.71,
                     upir=0.33, lost=327, lvis_flip=5.53, lvis_lost=1158),
}


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
    t = torch.from_numpy(im).unsqueeze(0).float()
    t.div_(255.0)          # in-place: out-of-place `/255.0`는 이미지당 여분의
    return t                # float32 버퍼를 만들고 버려서, probe 5000장 기준
                            # RSS가 이론치(~24GB)의 2배(~47GB)로 부풀었었다(09-08 발견).


def switch_vocab(model, names, device):
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.to(device).eval()


def measure_ap(model, data, imgsz, device):
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False, verbose=False, workers=0)
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


def predict_lvis_results(model, img_paths, img_ids, imgsz, device, conf=0.001, max_det=1000):
    # 09-17 버그 수정(pipeline/run_comparison.py, scripts/58과 동일): NMS
    # multi_label=True/False 불일치. model.predict()에 multi_label=True를
    # 넘겨도 DetectionPredictor가 안 읽어서 무시되므로 nms 모듈 함수 자체를
    # 임시 patch해야 한다. 실측(FP32, 공식 4809장 minival): AP 0.126→0.233.
    from ultralytics.utils import nms
    import functools
    orig_nms = nms.non_max_suppression
    nms.non_max_suppression = functools.partial(orig_nms, multi_label=True)
    try:
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
    finally:
        nms.non_max_suppression = orig_nms
    return results


def run_lvis_eval(lvis_gt, results, img_ids):
    # 09-17: Fixed AP 프로토콜 채택(pipeline/run_comparison.py와 동일 이유).
    from lvis import LVISEval, LVISResults
    from collections import defaultdict
    if not results:
        return dict(AP=0.0, AP50=0.0)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category_id"]].append(r)
    results = [r for rs in by_cat.values() for r in sorted(rs, key=lambda x: -x["score"])[:10000]]
    lvis_dt = LVISResults(lvis_gt, results, max_dets=-1)
    ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
    ev.params.img_ids = img_ids
    ev.run()
    return ev.get_results()


def compute_lvis_flip_gt_streaming(h_fp, h_models, probe_paths, gt_by_path, grid_specs, imgsz, conf=0.25):
    offsets = []
    off = 0
    for (H, W) in grid_specs:
        stride = imgsz // W
        offsets.append((off, H, W, stride))
        off += H * W

    tot = {m: 0 for m in h_models}; fl = {m: 0 for m in h_models}
    recip = {m: [] for m in h_models}
    lost = {m: 0 for m in h_models}; gained = {m: 0 for m in h_models}

    for i, p in enumerate(probe_paths):
        t = preprocess(p, imgsz, "cpu")
        sf = h_fp.run_image(t, i).sim

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

        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        conf_idx = conf_m.nonzero(as_tuple=True)[0]

        for mode, h_q in h_models.items():
            sq = h_q.run_image(t, i).sim
            if len(conf_idx) > 0:
                c_q = sq.argmax(-1)
                fl[mode] += int((c_fp[conf_idx] != c_q[conf_idx]).sum())
                tot[mode] += len(conf_idx)
            for (aidx, cls) in img_targets:
                fp_s = sf[aidx]; q_s = sq[aidx]
                fp_rank = int((fp_s > fp_s[cls]).sum()) + 1
                q_rank = int((q_s > q_s[cls]).sum()) + 1
                recip[mode].append(1.0 / q_rank)
                if fp_rank == 1 and q_rank != 1:
                    lost[mode] += 1
                if fp_rank != 1 and q_rank == 1:
                    gained[mode] += 1

        if (i + 1) % 500 == 0:
            print(f"    streaming {i+1}/{len(probe_paths)}장 처리")

    out = {}
    for mode in h_models:
        n = len(recip[mode])
        mrr = sum(recip[mode]) / max(n, 1)
        r1 = sum(1 for x in recip[mode] if x == 1.0) / max(n, 1)
        top1_flip = fl[mode] / max(tot[mode], 1) * 100
        out[mode] = dict(top1_flip=top1_flip, n_flip=tot[mode],
                         mrr=mrr, r1=r1, lost=lost[mode], gained=gained[mode], n=n)
    return out


def build_combined(model_cls, w, names, device, calib, fp, scale_reg_weight, iters=1500,
                   pidx=None, lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0,
                   h_eval=None, recon_iters_ada=1000,
                   channelwise_smult=False, identity_aware_margin=True):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    convert_to_adaround(m.model, channelwise_smult=channelwise_smult)
    optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
    optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                      lr=lr, k=k, neighbor_k=neighbor_k,
                                      neighbor_weight=neighbor_weight,
                                      asymmetric=True, scale_reg_weight=scale_reg_weight,
                                      exclude_from_neighbors=h_eval,
                                      identity_aware_margin=identity_aware_margin,
                                      verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--gt-ann", default=None)
    ap.add_argument("--lvis-ann",
                    default="/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--smult-per-tensor", action=argparse.BooleanOptionalAction, default=True,
                    help="09-16 §8.1 확정 설계(기본 True, run_comparison.py와 동일). "
                         "--no-smult-per-tensor로 이전(§8.11) per-channel 설계로 되돌릴 수 있음.")
    ap.add_argument("--identity-aware-margin", action=argparse.BooleanOptionalAction, default=True,
                    help="09-16 §8.1 확정 설계(기본 True). --no-identity-aware-margin으로 이전 "
                         "동작(정렬 비교)으로 되돌릴 수 있음.")
    ap.add_argument("--rw-list", default="10,20,30,50",
                    help="비교할 scale_reg_weight 후보 (콤마 구분)")
    ap.add_argument("--eval-cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    gt_ann = args.gt_ann or os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    print(f"[args] {vars(args)}")
    rw_list = [float(x) for x in args.rw_list.split(",")]

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

    calib_paths = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))[:args.calib]
    print(f"[data] calib {len(calib_paths)}장 (train2017)")

    probe_paths = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    if args.eval_cap > 0:
        probe_paths = probe_paths[:args.eval_cap]
    lvis_probe_paths, lvis_probe_ids = [], []
    for p in probe_paths:
        iid = int(os.path.basename(p).split(".")[0])
        if iid in lvis_img_ids:
            lvis_probe_paths.append(p); lvis_probe_ids.append(iid)
    print(f"[data] probe {len(probe_paths)}장(val2017 전체), 그중 LVIS minival과 겹치는 "
          f"{len(lvis_probe_paths)}장으로 LVIS 채점")

    calib = [preprocess(p, args.imgsz, device) for p in calib_paths]
    # probe(5000장)를 미리 리스트로 들고 있지 않는다 -- 장당 ~4.9MB로 프로세스당
    # ~24GB 고정 비용이 되어 동시 실행 가능한 seed 수를 제한하는 주범이었다
    # (scripts/58과 동일 문제, 09-10 확인). 필요할 때 preprocess()로 다시 읽는다.

    coco_gt_by_path = load_coco_gt_by_path(gt_ann, probe_paths)
    lvis_gt_by_path = load_lvis_gt_by_path(lvis_gt, lvis_probe_paths, lvis_probe_ids)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_eval = perm[60:80].tolist()
    H_eval_set = set(H_eval)

    print("[build] FP (COCO-80)")
    fp = YOLOWorld(args.model)
    fp.set_classes(coco)
    fp.fuse(); fp.model.to(device).eval()

    conditions = [f"combined_rw{rw:g}" for rw in rw_list]
    models, calib_time = {}, {}
    for rw, mode in zip(rw_list, conditions):
        print(f"[build] {mode}")
        t0 = time.perf_counter()
        models[mode] = build_combined(YOLOWorld, args.model, coco, device, calib, fp, rw,
                                      iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                                      neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                                      h_eval=H_eval, recon_iters_ada=args.recon_iters_ada,
                                      channelwise_smult=not args.smult_per_tensor,
                                      identity_aware_margin=args.identity_aware_margin)
        calib_time[mode] = time.perf_counter() - t0
        print(f"  {mode} 빌드 {calib_time[mode]:.1f}s")

    print(f"\n[gt] COCO-80 FP sim 계산 + anchor 매칭 ({len(probe_paths)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    grid_specs = get_grid_specs(h_fp, preprocess(probe_paths[0], args.imgsz, "cpu"))
    fp_sims = [h_fp.run_image(preprocess(p, args.imgsz, "cpu"), i).sim
              for i, p in enumerate(probe_paths)]
    h_fp.close()
    coco_gt_targets = build_gt_targets(fp_sims, probe_paths, coco_gt_by_path, grid_specs, args.imgsz)
    n_coco_gt = sum(len(t) for t in coco_gt_targets)
    print(f"  COCO GT anchor 수={n_coco_gt}")

    results = {}
    for mode in conditions:
        h_q = SimilarityHarness(models[mode].model, device=device)
        q_sims = [h_q.run_image(preprocess(p, args.imgsz, "cpu"), i).sim
                 for i, p in enumerate(probe_paths)]
        h_q.close()
        heval_flip, n1 = group_flip(fp_sims, q_sims, H_eval)
        top1_flip, n2 = standard_flip(fp_sims, q_sims)
        coco_gt_res = gt_metrics_for_method(fp_sims, q_sims, coco_gt_targets, H_eval_set)
        del q_sims
        free_cpu_mem()
        (coco_ap, coco_ap50), coco_pc = measure_ap(models[mode], args.data, args.imgsz, args.device)
        s_ap = subset_map(coco_pc, S); heval_ap = subset_map(coco_pc, H_eval)
        print(f"  [COCO-80][{mode}] AP={coco_ap:.2f} S_AP={s_ap:.2f} H_eval_AP={heval_ap:.2f} "
              f"Heval_flip={heval_flip:.2f}% Top1_flip={top1_flip:.2f}% "
              f"GT_MRR={coco_gt_res['mrr']:.4f} GT_R@1={coco_gt_res['r1']:.4f} "
              f"lost={coco_gt_res['lost']} gained={coco_gt_res['gained']} UPIR={coco_gt_res['upir']:.2f}%")
        results[mode] = dict(coco_ap=coco_ap, s_ap=s_ap, heval_ap=heval_ap, heval_flip=heval_flip,
                             top1_flip=top1_flip, coco_gt=coco_gt_res, calib_time=calib_time[mode])

    print(f"\n[switch] 전부 LVIS-1203 vocab으로 이식 (minival {len(lvis_probe_paths)}장)")
    switch_vocab(fp, lvis_names, device)
    for mode in conditions:
        switch_vocab(models[mode], lvis_names, device)

    h_fp = SimilarityHarness(fp.model, device=device)
    lvis_grid_specs = get_grid_specs(h_fp, preprocess(lvis_probe_paths[0], args.imgsz, "cpu"))
    h_models = {mode: SimilarityHarness(models[mode].model, device=device) for mode in conditions}

    print(f"  [streaming] flip/GT_MRR/R@1/lost/gained 계산 중 ({len(lvis_probe_paths)}장 x {len(conditions)}조건)")
    lvis_stream_res = compute_lvis_flip_gt_streaming(h_fp, h_models, lvis_probe_paths,
                                                     lvis_gt_by_path, lvis_grid_specs, args.imgsz)
    h_fp.close()
    for h_q in h_models.values():
        h_q.close()
    for mode in conditions:
        r = lvis_stream_res[mode]
        print(f"  [LVIS-flip/GT][{mode}] Top1_flip={r['top1_flip']:.2f}%(n={r['n_flip']}) "
              f"GT_MRR={r['mrr']:.4f} GT_R@1={r['r1']:.4f} lost={r['lost']} gained={r['gained']}")

    for mode in conditions:
        print(f"[predict+AP] {mode} (minival {len(lvis_probe_paths)}장) ...")
        preds = predict_lvis_results(models[mode], lvis_probe_paths, lvis_probe_ids, args.imgsz, device)
        lvis_ap_res = run_lvis_eval(lvis_gt, preds, lvis_probe_ids)
        print(f"  [LVIS-AP][{mode}] AP={lvis_ap_res.get('AP',0):.4f} AP50={lvis_ap_res.get('AP50',0):.4f} "
              f"APr={lvis_ap_res.get('APr',0):.4f} APc={lvis_ap_res.get('APc',0):.4f} "
              f"APf={lvis_ap_res.get('APf',0):.4f}")
        r = lvis_stream_res[mode]
        results[mode].update(lvis_ap=lvis_ap_res.get("AP", 0.0),
                             lvis_top1_flip=r["top1_flip"],
                             lvis_gt=dict(mrr=r["mrr"], r1=r["r1"], lost=r["lost"], gained=r["gained"]))

    print("\n" + "=" * 100)
    print(f" scale_reg_weight 스윕 (calib=train2017 {len(calib_paths)}장, LVIS=공식 minival) -- seed {args.seed}")
    print("=" * 100)
    print(f"{'method':>16} | {'COCO_AP':>8} | {'LVIS_AP':>8} | {'Heval_flip':>10} | "
          f"{'Top1_flip':>9} | {'UPIR':>6} | {'lost':>5} | {'LVIS_flip':>9} | {'LVIS_lost':>9}")
    for name, r in BASELINE_REF.items():
        print(f"{name:>16} | {r['coco_ap']:>8.2f} | {r['lvis_ap']:>8.4f} | {r['heval_flip']:>9.2f}% | "
              f"{r['top1_flip']:>8.2f}% | {r['upir']:>5.2f}% | {r['lost']:>5} | "
              f"{r['lvis_flip']:>8.2f}% | {r['lvis_lost']:>9}")
    for mode in conditions:
        r = results[mode]
        print(f"{mode:>16} | {r['coco_ap']:>8.2f} | {r['lvis_ap']:>8.4f} | {r['heval_flip']:>9.2f}% | "
              f"{r['top1_flip']:>8.2f}% | {r['coco_gt']['upir']:>5.2f}% | {r['coco_gt']['lost']:>5} | "
              f"{r['lvis_top1_flip']:>8.2f}% | {r['lvis_gt']['lost']:>9}")


if __name__ == "__main__":
    main()
