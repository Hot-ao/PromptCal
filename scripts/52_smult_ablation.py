"""
s_mult 인과관계 확정 ablation (09-07, §7.3 가설 검증).

가설: Combined가 LVIS에서 AdaRound보다도 나쁜(49번 결과) 이유는 rounding이
아니라 s_mult가 학습 후 1.0 미만으로 수렴한 전역 편향 때문이다.

검증: 같은 Combined 모델(rounding+s_mult 둘 다 학습됨)에서, 추론 직전에
s_mult만 강제로 1.0으로 되돌리면(rounding은 그대로) -- _quantize_smult()의
scale = a_obs.scale * s_mult이므로 s_mult=1이면 순수 AdaRound quantize()와
수학적으로 동일해진다. 이 상태로 LVIS AP를 다시 재서:
  - AdaRound 수준(0.2232 근방)으로 회복 -> s_mult 드리프트가 범인 확정.
  - 여전히 나쁨(0.21 근방 이하) -> rounding 쪽(Combined 자체 AdaRound 스테이지)
    도 원인에 포함 -> 재설계 방향 재검토 필요.

세 조건 비교: adaround(참조) / combined(그대로) / combined_smult1(s_mult=1 강제).

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python scripts/52_smult_ablation.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --lvis-ann /data/taeho/lvis_datasets/annotations/lvis_v1_val.json \
        --calib 32 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys, copy
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

    print("[build] combined (rounding + s_mult 둘 다 학습)")
    cb = YOLOWorld(args.model); cb.set_classes(coco); cb.fuse()
    wrap_convs(cb.model, 8, 8); cb.model.to(device).eval()
    calibrate(cb.model, calib, device=device)
    convert_to_adaround(cb.model)
    optimize_adaround(cb.model, fp.model, calib, device, iters=args.recon_iters_ada, verbose=False)
    optimize_promptcal_scale_neighbor(cb.model, fp.model, calib, device, S, iters=args.iters,
                                      lr=args.lr, k=args.k, neighbor_k=args.neighbor_k,
                                      neighbor_weight=args.neighbor_weight,
                                      asymmetric=True, verbose=True)

    ada_convs = list_adaround_convs(cb.model)
    s_before = [float(c.s_mult.detach()) for c in ada_convs]
    print(f"\n[s_mult] 학습 직후 평균={np.mean(s_before):.4f} (min={min(s_before):.4f}, max={max(s_before):.4f})")

    print("\n[switch] 전부 LVIS-1203 vocab으로 이식")
    switch_vocab(fp, lvis_names, device)
    switch_vocab(ada, lvis_names, device)
    switch_vocab(cb, lvis_names, device)

    conditions_ap = {}
    for label, m in [("adaround", ada), ("combined", cb)]:
        print(f"[predict+AP] {label} ...")
        preds = predict_lvis_results(m, probe_paths, probe_ids, args.imgsz, device)
        conditions_ap[label] = run_lvis_eval(lvis_gt, preds, probe_ids)
        print(f"  {label}: AP={conditions_ap[label].get('AP',0):.4f}")

    print("\n[ablation] combined의 s_mult를 전부 1.0으로 강제(rounding은 그대로)")
    with torch.no_grad():
        for c in ada_convs:
            c.s_mult.fill_(1.0)
    print(f"[predict+AP] combined_smult1 ...")
    preds = predict_lvis_results(cb, probe_paths, probe_ids, args.imgsz, device)
    conditions_ap["combined_smult1"] = run_lvis_eval(lvis_gt, preds, probe_ids)
    print(f"  combined_smult1: AP={conditions_ap['combined_smult1'].get('AP',0):.4f}")

    print(f"\n[predict+AP] FP32 (참고) ...")
    preds = predict_lvis_results(fp, probe_paths, probe_ids, args.imgsz, device)
    conditions_ap["FP32"] = run_lvis_eval(lvis_gt, preds, probe_ids)

    print("\n" + "=" * 80)
    print(f" s_mult ablation -- LVIS-1203 이식, seed {args.seed}, {len(probe_paths)}장")
    print("=" * 80)
    print(f"{'':>18} | {'AP':>7} | {'AP50':>7}")
    for label in ["FP32", "adaround", "combined", "combined_smult1"]:
        r = conditions_ap[label]
        print(f"{label:>18} | {r.get('AP',0):>7.4f} | {r.get('AP50',0):>7.4f}")

    ada_ap = conditions_ap["adaround"]["AP"]
    cb_ap = conditions_ap["combined"]["AP"]
    fix_ap = conditions_ap["combined_smult1"]["AP"]
    print("\n판정:")
    print(f"  AdaRound={ada_ap:.4f}  Combined(그대로)={cb_ap:.4f}  Combined(s_mult=1)={fix_ap:.4f}")
    if fix_ap >= ada_ap - 0.005:
        print("  -> s_mult=1로 되돌리니 AdaRound 수준으로 회복 == s_mult 드리프트가 범인 확정.")
        print("     재설계는 s_mult 파라미터화(전역 스칼라)를 겨냥해야 함.")
    elif fix_ap > cb_ap + 0.005:
        print("  -> 일부 회복은 되지만 AdaRound에는 못 미침 == s_mult가 주범이지만 rounding도")
        print("     약간의 몫이 있을 수 있음. s_mult 재설계가 우선, rounding도 재확인 필요.")
    else:
        print("  -> s_mult=1로 되돌려도 안 좋아짐 == rounding(Combined 자체 AdaRound 스테이지)")
        print("     쪽에도 문제가 있다는 뜻. s_mult만 고쳐서는 부족 -> 원인 재조사 필요.")


if __name__ == "__main__":
    main()
