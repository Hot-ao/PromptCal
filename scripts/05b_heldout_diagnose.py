"""
C-2 진단: held-out 악화 배수가 seed마다 튀는 원인 규명.
가설 — seed 3,4는 'person'처럼 빈도 높고 확실한(margin 큰, seen-flip 낮은) 클래스가
held-out에 몰려서, target 수가 폭증하고 seen(분모)이 작아져 배수가 부풀려진다.

각 seed에 대해 출력:
  - held-out에 'person' 포함 여부 및 그 비중
  - target region에 가장 많이 기여하는 held-out 클래스 top-5 (수/비중/그 클래스 seen-flip)
  - target region의 평균 maxprob (확실한 클래스일수록 높음)

실행:
    CUDA_VISIBLE_DEVICES=0 python scripts/05b_heldout_diagnose.py \
        --coco-root /data/taeho/coco_datasets --calib 64 --eval 1000 \
        --heldout 40 --seeds 0 1 2 3 4 --device 0
"""

import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness, paired_run
from src.quant.quant_model import wrap_convs, calibrate


def load_coco_names():
    import ultralytics, yaml
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
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names(); nc = len(names)

    H_masks, H_lists = {}, {}
    for s in args.seeds:
        H = np.random.default_rng(s).permutation(nc)[:args.heldout]
        m = torch.zeros(nc, dtype=torch.bool); m[H.tolist()] = True
        H_masks[s] = m; H_lists[s] = set(H.tolist())

    from ultralytics import YOLOWorld
    model_fp = YOLOWorld(args.model); model_fp.set_classes(names)
    model_q = YOLOWorld(args.model); model_q.set_classes(names)
    model_fp.fuse(); model_q.fuse()
    wrap_convs(model_q.model, w_bits=8, a_bits=8)
    model_fp.model.to(device).eval(); model_q.model.to(device).eval()

    all_imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_imgs = all_imgs[:args.calib]
    eval_imgs = all_imgs[args.calib:args.calib + args.eval]
    print(f"[calib] {len(calib_imgs)}장..."); 
    calibrate(model_q.model, (preprocess(p, args.imgsz, device) for p in calib_imgs), device=device)

    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    # seed -> class_idx -> [target_count, seen_flip_count, maxprob_sum]
    stat = {s: np.zeros((nc, 3)) for s in args.seeds}

    print(f"[eval] {len(eval_imgs)}장...")
    for i, p in enumerate(eval_imgs):
        t = preprocess(p, args.imgsz, device)
        rec_fp, rec_q = paired_run(h_fp, h_q, t, image_id=i)
        sim_fp, sim_q = rec_fp.sim, rec_q.sim
        prob = sim_fp.sigmoid(); maxprob, c_fp = prob.max(-1)
        conf = maxprob > args.conf_thres
        q_full = sim_q.argmax(-1)
        seen_flip_all = (q_full != c_fp)
        for s in args.seeds:
            tgt = conf & H_masks[s][c_fp]
            if tgt.sum() == 0: continue
            idx = tgt.nonzero(as_tuple=True)[0]
            for j in idx.tolist():
                c = int(c_fp[j])
                stat[s][c, 0] += 1
                stat[s][c, 1] += int(seen_flip_all[j])
                stat[s][c, 2] += float(maxprob[j])
        if (i + 1) % 400 == 0: print(f"  [{i+1}/{len(eval_imgs)}]")

    h_fp.close(); h_q.close()

    person = names.index("person") if "person" in names else -1
    print("\n" + "=" * 70)
    print(" C-2 진단: seed별 target 구성")
    print("=" * 70)
    for s in args.seeds:
        st = stat[s]; tot = st[:, 0].sum()
        person_share = st[person, 0] / max(tot, 1) * 100 if person in H_lists[s] else 0.0
        in_ho = "person∈H" if person in H_lists[s] else "person∉H"
        print(f"\n[seed {s}] targets={int(tot)}  ({in_ho}, person 비중 {person_share:.1f}%)")
        # target 기여 top-5 held-out 클래스
        order = np.argsort(-st[:, 0])
        print(f"   {'class':>14} | {'targets':>8} | {'%tot':>5} | {'seen-flip':>9} | {'avg maxprob':>11}")
        for c in order[:5]:
            if st[c, 0] == 0: break
            cnt = st[c, 0]; sf = st[c, 1] / cnt * 100; mp = st[c, 2] / cnt
            print(f"   {names[int(c)]:>14} | {int(cnt):>8} | {cnt/tot*100:>4.1f}% | "
                  f"{sf:>8.2f}% | {mp:>11.3f}")

    print("\n" + "=" * 70)
    print("확인 포인트: seed 3,4에서 person 등 고빈도·고확신(seen-flip 낮음) 클래스가")
    print("target을 지배하면, 분모(seen)가 작아져 악화배수가 부풀려진 것 = 배수 지표의 한계.")


if __name__ == "__main__":
    main()
