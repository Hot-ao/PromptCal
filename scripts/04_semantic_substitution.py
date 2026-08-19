"""
국면 C-1: Semantic Class Substitution — 양자화가 의미적으로 가까운 프롬프트로
의사결정을 밀어내는가? (논문의 OVOD 고유 축 · abstract "의미적 class substitution")

가설: 양자화 손상은 작은 margin(경계)에 집중된다(국면 B). 의미적으로 가까운 프롬프트는
텍스트 임베딩이 비슷해 margin이 작으므로, top-1 flip이 random이 아니라 '의미적 이웃'으로
편향되어야 한다. 이것이 관측되면 "양자화가 semantic decision을 망가뜨린다"의 직접 증거.

측정(전부 GT-free):
  - 각 클래스의 의미적 이웃 = CLIP 텍스트 임베딩 코사인 top-k.
  - confident region에서 FP top-1 = c_fp, quant top-1 = c_q.
  - c_q != c_fp인 flip 중, c_q가 c_fp의 top-k 이웃인 비율.
  - 우연 기대치 k/(nc-1)과 비교. 훨씬 높으면 semantic substitution 확정.

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/04_semantic_substitution.py \
        --coco-root /data/taeho/coco_datasets --calib 64 --eval 200 --k 5 --device 0
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness, paired_run
from src.quant.quant_model import wrap_convs, calibrate


def load_coco_names():
    import ultralytics
    import yaml
    from pathlib import Path
    cfg = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    with open(cfg) as f:
        names = yaml.safe_load(f)["names"]
    return [names[i] for i in range(len(names))]


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    bottom, right = new - nh - top, new - nw - left
    return cv2.copyMakeBorder(im_r, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(path, imgsz, device):
    im0 = cv2.imread(path)
    im = letterbox(im0, imgsz)
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im).float().unsqueeze(0).to(device) / 255.0


def get_text_embeddings(model, names, device):
    """set_classes 후 저장된 텍스트 임베딩 [nc, dim]을 뽑아 정규화해 반환."""
    tf = getattr(model.model, "txt_feats", None)
    if tf is None:
        raise RuntimeError("model.model.txt_feats 없음. set_classes 후 접근 필요.")
    tf = tf.detach().float()
    if tf.dim() == 3:      # [1, nc, dim]
        tf = tf[0]
    tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return tf.to(device)


def build_neighbors(txt, k):
    """각 클래스의 의미적 top-k 이웃(자기 제외) 집합 리스트 반환."""
    sim = txt @ txt.T                       # [nc, nc] 코사인
    nc = sim.shape[0]
    neigh = []
    for c in range(nc):
        order = torch.argsort(sim[c], descending=True).tolist()
        order = [o for o in order if o != c][:k]
        neigh.append(set(order))
    return neigh, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--k", type=int, default=5, help="의미적 이웃 개수")
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()
    nc = len(names)

    from ultralytics import YOLOWorld
    print("[load] FP32 + quant models")
    model_fp = YOLOWorld(args.model); model_fp.set_classes(names)
    model_q = YOLOWorld(args.model); model_q.set_classes(names)
    model_fp.fuse(); model_q.fuse()
    wrap_convs(model_q.model, w_bits=8, a_bits=8)
    model_fp.model.to(device).eval(); model_q.model.to(device).eval()

    # 의미적 이웃 구조
    txt = get_text_embeddings(model_fp, names, device)
    neigh, sim = build_neighbors(txt, args.k)
    print(f"[neigh] k={args.k} 이웃 구조 구축. 샘플(sanity check):")
    for c in [names.index(x) for x in ["dog", "car", "apple"] if x in names]:
        ns = [names[i] for i in sorted(neigh[c])]
        print(f"   {names[c]:>10} → {ns}")

    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_imgs = all_imgs[:args.calib]
    eval_imgs = all_imgs[args.calib:args.calib + args.eval]
    print(f"[calib] {len(calib_imgs)}장...")
    calibrate(model_q.model, (preprocess(p, args.imgsz, device) for p in calib_imgs), device=device)

    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    neigh_t = [torch.tensor(sorted(s)) for s in neigh]  # 빠른 조회용
    total_conf = 0
    total_flip = 0
    flip_to_neigh = 0
    # 대조: flip이 아닌 경우까지 포함해 top-2가 이웃인 비율(경쟁 구조 확인)
    top2_is_neigh = 0

    print(f"[eval] {len(eval_imgs)}장...")
    for i, p in enumerate(eval_imgs):
        t = preprocess(p, args.imgsz, device)
        rec_fp, rec_q = paired_run(h_fp, h_q, t, image_id=i)
        sim_fp, sim_q = rec_fp.sim, rec_q.sim

        prob = sim_fp.sigmoid()
        maxprob, c_fp = prob.max(-1)
        mask = maxprob > args.conf_thres
        if mask.sum() == 0:
            continue
        c_fp_m = c_fp[mask]
        c_q_m = sim_q[mask].argmax(-1)
        # top-2 (FP)
        top2_fp = sim_fp[mask].topk(2, dim=-1).indices[:, 1]

        for j in range(c_fp_m.shape[0]):
            a = int(c_fp_m[j]); b = int(c_q_m[j]); t2 = int(top2_fp[j])
            total_conf += 1
            if t2 in neigh[a]:
                top2_is_neigh += 1
            if b != a:
                total_flip += 1
                if b in neigh[a]:
                    flip_to_neigh += 1

        if (i + 1) % 50 == 0:
            r = flip_to_neigh / total_flip * 100 if total_flip else 0
            print(f"  [{i+1}/{len(eval_imgs)}] flips={total_flip} →이웃 {r:.1f}%")

    h_fp.close(); h_q.close()

    chance = args.k / (nc - 1) * 100
    print("\n" + "=" * 56)
    print(" SEMANTIC CLASS SUBSTITUTION (confident regions)")
    print("=" * 56)
    print(f" confident regions      : {total_conf}")
    print(f" 그중 top-1 flip        : {total_flip} "
          f"({total_flip/max(total_conf,1)*100:.1f}%)")
    print(f" flip → 의미적 이웃(top-{args.k}) : "
          f"{flip_to_neigh/max(total_flip,1)*100:.1f}%")
    print(f" 우연 기대치            : {chance:.1f}%")
    ratio = (flip_to_neigh/max(total_flip,1)) / (args.k/(nc-1))
    print(f" 편향 배수 (관측/우연)  : {ratio:.1f}x")
    print("-" * 56)
    print(f" [대조] confident region의 top-2가 이웃인 비율: "
          f"{top2_is_neigh/max(total_conf,1)*100:.1f}%")
    print("=" * 56)
    print("\n해석: flip→이웃 비율이 우연 기대치보다 크게 높으면(편향 배수 >> 1),")
    print("양자화가 random이 아니라 '의미적으로 가까운 프롬프트'로 의사결정을 밀어냄")
    print("= semantic class substitution. 프롬프트 축이 논문 손상의 본질임을 입증.")


if __name__ == "__main__":
    main()
