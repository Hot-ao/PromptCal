"""
국면 I 보강 (09-07): LVIS-1203 vocabulary에서 진짜 GT 기반 AP까지 측정.

47_lvis_generality.py는 실제 LVIS 이미지/annotation 없이 FP32 pseudo-label로
flip/margin만 쟀다 -- 그런데 COCO의 굵은 class(dog)가 LVIS에서 세부 class
(Yorkshire Terrier 등)로 쪼개지면, "flip"이 실제로는 무해한 세부 재배치일 수도
있고 진짜 의미 붕괴일 수도 있어서 flip만으로는 구분이 안 된다(사용자 지적).
이걸 가르려면 진짜 GT 기반 AP가 필요 -> 공식 LVIS v1 val annotation을
(다른 사용자 파일이 아니라) https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
에서 새로 받아 /data/taeho/lvis_datasets/annotations/에 배치했다.

LVIS val(19809장)은 COCO val2017(5000장)과 다른 split이라 완전히 겹치지 않는다
(교집합 4809/5000, 96.2%). 우리 probe 이미지 중 LVIS val에 실제로 있는 것만
걸러서 쓴다. ultralytics lvis.yaml의 names[i]는 실제 LVIS category_id=i+1과
순서가 정확히 대응(직접 확인됨) -- 별도 remap 테이블 불필요.

federated 평가(이미지마다 exhaustively 라벨링 안 된 category가 있는 LVIS 특유
프로토콜)를 정확히 반영하기 위해 자체 채점 대신 공식 lvis-api(LVISEval)를 사용.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/49_lvis_ap.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --lvis-ann /data/taeho/lvis_datasets/annotations/lvis_v1_val.json \
        --calib 32 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float   # lvis-api 0.5.3(2020)가 numpy<1.20 시절 np.float를 씀 -- 최신 numpy 호환용 패치

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
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
                                          asymmetric=True, verbose=False)
    return m


def predict_lvis_results(model, img_paths, img_ids, imgsz, device, conf=0.001, max_det=300):
    """model.predict()로 원본 이미지 좌표계 xyxy를 얻어 LVIS 결과 포맷으로 변환.
    ultralytics predict()는 letterbox->원본 스케일 역변환을 내부에서 처리해준다."""
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
            results.append({
                "image_id": int(img_id),
                "category_id": int(c) + 1,   # names[i] <-> LVIS category_id=i+1
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc),
            })
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

    # 우리 probe 중 LVIS val split에 실제로 존재하는 것만 필터링(교집합, 09-07 확인: ~97%)
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

    print("[build] FP (COCO-80 vocab)")
    fp = build(YOLOWorld, args.model, coco, device, calib, "fp")
    conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]
    models = {}
    for mode in conditions:
        print(f"[build] {mode} (COCO-80 calibration/training)")
        models[mode] = build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
                             iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                             neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                             recon_iters_ada=args.recon_iters_ada,
                             recon_iters_strong=args.recon_iters_strong,
                             qdrop_prob=args.qdrop_prob)

    ap_results = {}
    for label, m in [("FP32", fp)] + [(c, models[c]) for c in conditions]:
        print(f"\n[predict+AP] {label} -- LVIS-1203 vocab, {len(probe_paths)}장 ...")
        switch_vocab(m, lvis_names, device)
        preds = predict_lvis_results(m, probe_paths, probe_ids, args.imgsz, device)
        print(f"  예측 {len(preds)}개, LVISEval 채점 중...")
        res = run_lvis_eval(lvis_gt, preds, probe_ids)
        ap_results[label] = res
        ap_key = [k for k in res if k.upper() in ("AP",)]
        print(f"  {label}: AP={res.get('AP', float('nan')):.4f} AP50={res.get('AP50', float('nan')):.4f}")

    print("\n" + "=" * 90)
    print(f" LVIS-1203 vocab 진짜 GT 기반 AP (seed {args.seed}, {len(probe_paths)}장)")
    print("=" * 90)
    print(f"{'':>10} | {'AP':>7} | {'AP50':>7} | {'APr':>7} | {'APc':>7} | {'APf':>7}")
    for label in ["FP32"] + conditions:
        r = ap_results[label]
        print(f"{label:>10} | {r.get('AP',0):>7.4f} | {r.get('AP50',0):>7.4f} | "
              f"{r.get('APr',0):>7.4f} | {r.get('APc',0):>7.4f} | {r.get('APf',0):>7.4f}")

    print("\n판정:")
    print("  47번(pseudo-label flip)에서 Combined가 baseline보다 flip이 높게 나왔던 게")
    print("  실제 AP 하락(진짜 의미 붕괴)과 같이 가면 -> 한계가 진짜다.")
    print("  flip은 높은데 AP는 baseline과 비슷/낫다면 -> 그 flip 상당수가 세부 class")
    print("  재배치 같은 무해한 흔들림이었다는 뜻 -> 47번 결과를 그대로 한계로 쓰면 과대 해석.")


if __name__ == "__main__":
    main()
