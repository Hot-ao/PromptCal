"""
C-3 진단: COCO vs LVIS 공정 비교를 위한 두 가지 확인.
  (1) 앵커 수가 프롬프트 집합(80 vs 1203)과 무관하게 동일한가
      → 같으면 같은 region 위에서 프롬프트만 바꾼 controlled 비교가 성립.
  (2) COCO 80 클래스가 LVIS 1203에서 어떤 이름에 대응하는가 (특히 person, car).

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/06b_lvis_diagnose.py \
        --coco-root /data/taeho/coco_datasets --device 0
"""

import argparse, glob, os, re, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness


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


def tokens(name):
    # 'car_(automobile)' / 'airplane/aeroplane' → 개별 토큰 집합
    return set(re.split(r"[/_()\s]+", name.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    coco = load_names("coco")
    lvis = load_names("lvis")
    lvis_tok = [tokens(n) for n in lvis]

    # ---- (2) 이름 매핑 ----
    print("=" * 66)
    print(" COCO 80 → LVIS 1203 이름 매핑")
    print("=" * 66)
    n_exact, n_none = 0, 0
    missing = []
    for c in coco:
        ctok = tokens(c)
        matches = [lvis[i] for i, t in enumerate(lvis_tok) if ctok & t]
        if matches:
            n_exact += 1
            if c in ["person", "car", "dog", "cat", "chair", "bottle"]:
                print(f"   {c:>14} → {matches[:3]}")
        else:
            n_none += 1
            missing.append(c)
    print("-" * 66)
    print(f" 매칭됨: {n_exact}/80,  매칭 실패: {n_none}")
    if missing:
        print(f" 실패 목록: {missing}")

    # ---- (1) 앵커 수 프롬프트 무관 확인 ----
    print("\n" + "=" * 66)
    print(" 앵커 수: 프롬프트 집합과 무관한가")
    print("=" * 66)
    from ultralytics import YOLOWorld
    img = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))[0]
    t = preprocess(img, args.imgsz, device)

    shapes = {}
    for tag, nm in [("COCO-80", coco), ("LVIS-1203", lvis)]:
        m = YOLOWorld(args.model); m.set_classes(nm); m.fuse()
        m.model.to(device).eval()
        h = SimilarityHarness(m.model, device=device)
        rec = h.run_image(t, image_id=0)
        h.close()
        shapes[tag] = tuple(rec.sim.shape)
        print(f"   {tag:>10}: 유사도 행렬 {shapes[tag]}  (앵커={rec.sim.shape[0]})")

    same = shapes["COCO-80"][0] == shapes["LVIS-1203"][0]
    print("-" * 66)
    print(f" 앵커 수 동일: {'OK ✅ (같은 region, 프롬프트만 다름 → 공정 비교)' if same else 'MISMATCH ⚠️'}")
    print("=" * 66)


if __name__ == "__main__":
    main()
