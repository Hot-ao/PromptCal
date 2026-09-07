"""
scale_reg_weight 스윕 (09-07, 재설계 1단계). §7.6의 "(s_mult-1)^2 정규화 먼저
시도" 실행. COCO->LVIS 이식 시나리오(51번과 동일 설정)에서 여러
scale_reg_weight 값에 대해 s_mult 평균/AP를 재서, "LVIS AP는 AdaRound
수준으로 회복시키면서 COCO-80 S_AP(margin_loss 효과)는 크게 안 깎이는" 값을
찾는다.

판정 기준: LVIS AP가 AdaRound(0.2232, seed 2 기준)에 가까워지고, S_AP(COCO-80,
margin_loss가 원래 지키려던 대상)가 reg 없는 버전 대비 크게 나빠지지 않으면
성공 -- 원래 COCO-80에서의 6/6 승리(45번)를 해치지 않으면서 LVIS 문제를
완화하는 게 목표.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/53_scale_reg_sweep.py \
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
    ap.add_argument("--reg-weights", type=float, nargs="+", default=[0.0, 1.0, 10.0, 50.0, 200.0])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

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

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist()

    results = {}
    for rw in args.reg_weights:
        torch.manual_seed(args.torch_seed)
        torch.cuda.manual_seed_all(args.torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        calib = [preprocess(p, args.imgsz, device) for p in calib_paths]
        print(f"\n{'='*20} scale_reg_weight={rw} {'='*20}")
        fp = YOLOWorld(args.model); fp.set_classes(coco); fp.fuse(); fp.model.to(device).eval()
        cb = YOLOWorld(args.model); cb.set_classes(coco); cb.fuse()
        wrap_convs(cb.model, 8, 8); cb.model.to(device).eval()
        calibrate(cb.model, calib, device=device)
        convert_to_adaround(cb.model)
        optimize_adaround(cb.model, fp.model, calib, device, iters=args.recon_iters_ada, verbose=False)
        optimize_promptcal_scale_neighbor(cb.model, fp.model, calib, device, S, iters=args.iters,
                                          lr=args.lr, k=args.k, neighbor_k=args.neighbor_k,
                                          neighbor_weight=args.neighbor_weight,
                                          asymmetric=True, scale_reg_weight=rw, verbose=True)
        ada_convs = list_adaround_convs(cb.model)
        smean = float(np.mean([float(c.s_mult.detach()) for c in ada_convs]))

        # COCO-80 AP(전체) -- margin_loss 효과가 안 깎였는지 확인
        coco_ap, coco_ap50 = measure_ap(cb, args.data, args.imgsz, args.device)

        # LVIS 이식 AP -- 이번 정규화가 실제로 도움이 되는지 확인
        switch_vocab(cb, lvis_names, device)
        preds = predict_lvis_results(cb, probe_paths, probe_ids, args.imgsz, device)
        lvis_res = run_lvis_eval(lvis_gt, preds, probe_ids)

        results[rw] = dict(s_mean=smean, coco_ap=coco_ap, lvis_ap=lvis_res.get("AP", 0.0))
        print(f"  [rw={rw}] s_mult 평균={smean:.4f}  COCO-80 AP={coco_ap:.2f}  LVIS AP={lvis_res.get('AP',0):.4f}")

    print("\n" + "=" * 80)
    print(f" scale_reg_weight 스윕 결과 (seed {args.seed})")
    print("=" * 80)
    print(f"{'reg_weight':>12} | {'s_mult 평균':>10} | {'COCO-80 AP':>10} | {'LVIS AP':>8}")
    for rw in args.reg_weights:
        r = results[rw]
        print(f"{rw:>12.1f} | {r['s_mean']:>10.4f} | {r['coco_ap']:>10.2f} | {r['lvis_ap']:>8.4f}")

    print("\n참고값: naive COCO-80 AP≈35.06, AdaRound LVIS AP≈0.2232(seed2)")
    print("판정: LVIS AP가 0.2232에 가까워지면서 COCO-80 AP가 naive(35.06)보다")
    print("  여전히 높게 유지되는 reg_weight을 찾는 게 목표.")


if __name__ == "__main__":
    main()
