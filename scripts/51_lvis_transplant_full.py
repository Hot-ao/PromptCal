"""
COCO-80 calibration -> LVIS-1203로 재학습 없이 이식(47/49와 같은 시나리오)했을
때, 적용 가능한 전체 지표를 한 번에 측정 (09-07, 50번과 짝을 이루는 대조군).

47(flip/margin, pseudo-label)과 49(real GT AP)로 나눠서 쟀던 걸 하나로 합치고,
실제 GT 기반 GT_MRR/R@1/lost/gained까지 추가한다. verbose=True로 Combined의
s_mult 수렴 과정도 로그에 남긴다 -- 45(COCO-80 native)의 s_mult(~0.977, 1.0
아래로 수렴)와 비교해서 "이식 모델의 s_mult 자체는 COCO-80 학습 때 이미
고정된 값과 동일"이라는 걸 직접 확인하기 위함(45와 동일 레시피/시드이므로
이론적으로 같은 값이 나와야 함 -- 재확인 차원).

주의(50번과의 핵심 차이, 반드시 구분해서 해석할 것):
  - H_eval_flip(masked)과 UPIR은 여기 없다. 두 지표는 "같은 vocabulary 안에서
    특정 컬럼(H_eval)만 가리는" 구조인데, LVIS로 vocabulary를 통째로 바꾸면
    COCO의 H_eval 컬럼 자체가 출력에 존재하지 않는다 -- 구조적으로 계산 불가.
  - S/H_eval AP subset도 없다. S/H_eval은 COCO-80 class 인덱스인데 LVIS 출력
    공간에는 그 인덱스가 다른 class를 가리키므로 subset AP가 의미를 잃는다.
  - 즉 여기서 측정 가능한 건: AP(전체, 진짜 LVIS GT) + 표준 Top1_flip(항상
    well-defined) + GT_MRR/R@1/lost/gained(진짜 GT 기준, well-defined) + 비용.
  - 50(LVIS-native, S/H_eval을 LVIS class로 다시 나눠 처음부터 학습)에는 이
    제약이 없어 masked flip/UPIR/subset AP까지 전부 나온다 -- 두 실험은 서로
    다른 것을 측정하므로 표를 나란히 놓고 "이식 vs native"만 비교할 것.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python scripts/51_lvis_transplant_full.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --lvis-ann /data/taeho/lvis_datasets/annotations/lvis_v1_val.json \
        --calib 32 --eval 500 --seed 2 --device 0
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


def switch_vocab(model, names, device):
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.to(device).eval()


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


def gt_metrics_for_method(fp_sims, q_sims, gt_targets):
    """H_eval 개념이 없으므로(이식 모델은 LVIS class를 하나도 안 봤음) UPIR은
    계산하지 않는다 -- MRR/R@1/lost/gained만."""
    recip_ranks = []
    lost = gained = 0
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
    n = len(recip_ranks)
    mrr = sum(recip_ranks) / max(n, 1)
    r1 = sum(1 for r in recip_ranks if r == 1.0) / max(n, 1)
    return dict(mrr=mrr, r1=r1, lost=lost, gained=gained, n=n)


def quantized_weight_mib(model_module):
    total = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            total += m.conv.weight.numel()
    return total / (1024 * 1024)


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
    ap.add_argument("--lvis-ann", default="/data/taeho/lvis_datasets/annotations/lvis_v1_val.json")
    ap.add_argument("--calib", type=int, default=32)
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
    probe = [preprocess(p, args.imgsz, device) for p in probe_paths]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist()   # COCO-80 학습용(45와 동일 recipe)

    print("[build] FP (COCO-80 vocab, 학습 전용)")
    fp = build(YOLOWorld, args.model, coco, device, calib, "fp")
    conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]
    models, calib_time = {}, {}
    for mode in conditions:
        print(f"[build] {mode} (COCO-80 calibration/training)")
        t0 = time.perf_counter()
        models[mode] = build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
                             iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                             neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                             recon_iters_ada=args.recon_iters_ada,
                             recon_iters_strong=args.recon_iters_strong,
                             qdrop_prob=args.qdrop_prob)
        calib_time[mode] = time.perf_counter() - t0
    model_mib = quantized_weight_mib(models["adaround"].model)

    print(f"\n[switch] 전부 LVIS-1203 vocab으로 이식(재학습 없음)")
    switch_vocab(fp, lvis_names, device)
    for mode in conditions:
        switch_vocab(models[mode], lvis_names, device)

    print(f"[gt] FP sim 계산 + LVIS GT anchor 매칭 ({len(probe_paths)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    grid_specs = get_grid_specs(h_fp, probe[0])
    fp_sims = [h_fp.run_image(t, i).sim for i, t in enumerate(probe)]
    h_fp.close()
    gt_by_path = load_lvis_gt_by_path(lvis_gt, probe_paths, probe_ids)
    gt_targets = build_gt_targets(fp_sims, probe_paths, gt_by_path, grid_specs, args.imgsz)
    n_gt = sum(len(t) for t in gt_targets)
    print(f"  매칭된 GT anchor 수={n_gt}")

    std_flip_results, gt_results, ap_results = {}, {}, {}
    for mode in conditions:
        h_q = SimilarityHarness(models[mode].model, device=device)
        q_sims = [h_q.run_image(t, i).sim for i, t in enumerate(probe)]
        h_q.close()
        std_flip_results[mode], n2 = standard_flip(fp_sims, q_sims)
        gt_results[mode] = gt_metrics_for_method(fp_sims, q_sims, gt_targets)
        print(f"  {mode:>10}: Top1_flip={std_flip_results[mode]:.2f}%(n={n2})  "
              f"GT_MRR={gt_results[mode]['mrr']:.4f}  GT_R@1={gt_results[mode]['r1']:.4f}  "
              f"lost={gt_results[mode]['lost']}  gained={gt_results[mode]['gained']}")

        print(f"[predict+AP] {mode} ...")
        preds = predict_lvis_results(models[mode], probe_paths, probe_ids, args.imgsz, device)
        ap_results[mode] = run_lvis_eval(lvis_gt, preds, probe_ids)
        print(f"  {mode}: AP={ap_results[mode].get('AP',0):.4f} AP50={ap_results[mode].get('AP50',0):.4f}")

    print(f"[predict+AP] FP32 ...")
    preds = predict_lvis_results(fp, probe_paths, probe_ids, args.imgsz, device)
    ap_results["FP32"] = run_lvis_eval(lvis_gt, preds, probe_ids)

    print("\n" + "=" * 100)
    print(f" COCO-80 캘리브레이션 -> LVIS-1203 이식(zero-shot) -- seed {args.seed}, {len(probe_paths)}장")
    print("=" * 100)
    print(f"{'':>10} | {'AP':>7} | {'AP50':>7} | {'Top1_flip':>9} | {'GT_MRR':>7} | {'GT_R@1':>7} | {'lost':>5} | {'gained':>6}")
    for name in ["FP32"] + conditions:
        r = ap_results[name]
        if name == "FP32":
            print(f"{name:>10} | {r.get('AP',0):>7.4f} | {r.get('AP50',0):>7.4f} | {'-':>9} | {'-':>7} | {'-':>7} | {'-':>5} | {'-':>6}")
        else:
            g = gt_results[name]
            print(f"{name:>10} | {r.get('AP',0):>7.4f} | {r.get('AP50',0):>7.4f} | "
                  f"{std_flip_results[name]:>8.2f}% | {g['mrr']:>7.4f} | {g['r1']:>7.4f} | "
                  f"{g['lost']:>5} | {g['gained']:>6}")

    print("\n" + "=" * 100)
    print(" 비용/크기")
    print("=" * 100)
    print(f"  양자화 weight 이론 크기: {model_mib:.2f} MiB")
    for mode in conditions:
        print(f"  {mode:>10} | calib {calib_time[mode]:>8.1f}s")

    print("\n주의: 이 표는 50_lvis_native.py(같은 시각 실행)와 짝을 이룬다.")
    print("  이식(여기) vs native(50번) 둘 다에서 Combined가 나쁘면 -> neighbor-hinge")
    print("     설계 자체의 구조적 한계(재설계 필요).")
    print("  이식에서만 나쁘고 native에서는 괜찮으면 -> '재학습 없는 이식'이라는")
    print("     시나리오의 한계일 뿐, 설계 원리는 유효(스코프 명시로 충분).")


if __name__ == "__main__":
    main()
