"""
C-3 (1단계): LVIS 1203 프롬프트 sanity check.

큰 변경(프롬프트 80→1203) 앞의 안전장치. 확인 항목:
  (1) LVIS 클래스명 로드 및 개수(1203) 확인, 샘플 출력.
  (2) set_classes(1203) 후 유사도 행렬이 [anchors, 1203]으로 나오는가.
  (3) value range가 정상(pre-sigmoid logit)인가.
  (4) 이웃 구조가 상식적인가 (dog→puppy/poodle 류의 촘촘한 이웃이 보이는가).
  (5) 1이미지 forward 시간(1203 프롬프트 처리 가능 여부).

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/06_lvis_sanity.py \
        --coco-root /data/taeho/coco_datasets --device 0
"""

import argparse, glob, os, sys, time
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness


def load_lvis_names():
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "lvis.yaml"
    d = yaml.safe_load(open(p))
    names = d["names"]
    return [names[i] for i in range(len(names))]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    names = load_lvis_names()
    print(f"[names] LVIS 클래스 수: {len(names)}")
    print(f"[names] 샘플(앞 10): {names[:10]}")
    print(f"[names] 샘플(무작위): {[names[i] for i in [100, 300, 600, 900, 1200]]}")

    from ultralytics import YOLOWorld
    print("[load] model + set_classes(1203) ...")
    t0 = time.time()
    model = YOLOWorld(args.model)
    model.set_classes(names)
    model.fuse()
    model.model.to(device).eval()
    print(f"[load] set_classes 완료 ({time.time()-t0:.1f}s)")

    # 텍스트 임베딩 → 이웃 sanity
    tf = model.model.txt_feats.detach().float()
    if tf.dim() == 3:
        tf = tf[0]
    tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim_txt = tf @ tf.T
    print("\n[neigh] 샘플 클래스의 의미적 top-5 이웃(LVIS는 촘촘해야 정상):")
    for target in ["dog", "car", "person", "cat", "bottle"]:
        if target in names:
            c = names.index(target)
            order = torch.argsort(sim_txt[c], descending=True).tolist()
            ns = [names[o] for o in order if o != c][:5]
            print(f"   {target:>8} → {ns}")
        else:
            print(f"   {target:>8} → (LVIS에 정확히 없음)")

    # forward + 유사도 행렬 shape 확인
    img = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))[0]
    print(f"\n[fwd] image: {img}")
    harness = SimilarityHarness(model.model, device=device)
    t1 = time.time()
    rec = harness.run_image(preprocess(img, args.imgsz, device), image_id=0)
    dt = time.time() - t1
    harness.close()

    sim = rec.sim
    print(f"[fwd] 유사도 행렬 shape: {tuple(sim.shape)} (기대 [anchors, 1203])")
    print(f"[fwd] value range: min={sim.min():.3f} max={sim.max():.3f} (pre-sigmoid logit)")
    print(f"[fwd] forward 시간: {dt*1000:.0f} ms/img")

    ok_shape = (sim.shape[1] == len(names))
    print("\n" + "=" * 50)
    print(f" LVIS SANITY: {'OK ✅' if ok_shape else 'MISMATCH ⚠️'}")
    if ok_shape:
        # 이 이미지의 top 예측(참고)
        prob = sim.sigmoid()
        maxp, amax = prob.max(-1)
        best = torch.argmax(maxp).item()
        print(f" 최고 confident region → '{names[int(amax[best])]}' (p={maxp[best]:.3f})")
        est = dt * 500
        print(f" 500장 예상 소요: ~{est:.0f}s (본 측정 eval 규모 참고)")
    else:
        print(f" 행렬 열 수({sim.shape[1]})가 프롬프트 수({len(names)})와 불일치.")
    print("=" * 50)


if __name__ == "__main__":
    main()
