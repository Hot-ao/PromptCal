"""
seed별 유불리 원인 분석 (09-05 진단 후속, 39번 다음 단계).

39번(결정성 적용 반복실험)에서 seed 0/1은 AdaRound+scale이 지고 seed 2만 이겼다
(35번 원본의 "2승 1패"와 정확히 반대 패턴). 이 스크립트는 학습을 다시 하지 않고
FP 모델만으로 그 차이의 후보 원인들을 본다:

  1. 각 seed의 H_eval class 목록 자체(질적 비교) -- 어떤 class들이 held-out으로
     빠졌는지.
  2. H_eval class들이 text embedding 공간에서 S(calibration에 쓴 class)와
     얼마나 가까운가 -- src/quant/semantic_calib.py의 text_neighbor_order를
     재사용. 가까운 class가 많으면 S에서 배운 margin 조정이 H_eval에도
     "우연히" 잘 맞을 가능성, 멀면 그 반대일 가능성.
  3. H_eval 대상 confident anchor 수(n) -- FP 모델의 top-1이 H_eval class이고
     confidence>0.25인 anchor 개수. 이건 quantized model과 무관하게 FP+probe
     이미지만으로 결정되므로 학습 없이 바로 계산 가능. n이 seed마다 크게
     다르면 애초에 flip% 자체의 표본 크기가 달라 노이즈에 더 취약했을 수 있다.

실행:
    CUDA_VISIBLE_DEVICES=7 python scripts/40_seed_analysis.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --eval 500 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.semantic_calib import get_txt_feats, text_neighbor_order


def load_coco_names():
    import ultralytics, yaml
    from pathlib import Path
    d = yaml.safe_load(open(Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"))
    return [d["names"][i] for i in range(len(d["names"]))]


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


def confident_heval_anchor_count(h_fp, probe, H_eval, conf=0.25):
    Hm = torch.zeros(80, dtype=torch.bool)
    Hm[H_eval] = True
    tot = 0
    per_class = {c: 0 for c in H_eval}
    for i, t in enumerate(probe):
        sf = h_fp.run_image(t, i).sim
        prob = sf.sigmoid()
        mp, c_fp = prob.max(-1)
        conf_m = mp > conf
        target = conf_m & Hm[c_fp]
        idx = target.nonzero(as_tuple=True)[0]
        tot += len(idx)
        for c in c_fp[idx].tolist():
            per_class[c] += 1
    return tot, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()

    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    probe = [preprocess(p, args.imgsz, device) for p in imgs[32:32 + args.eval]]  # 29~39와 동일 오프셋

    print("[load] FP")
    fp = YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    h_fp = SimilarityHarness(fp.model, device=device)

    txt_feats = get_txt_feats(fp.model).to(device)  # [80, 512], L2-normalized
    neighbor_order = text_neighbor_order(txt_feats)  # [80, 80] 유사도 내림차순 인덱스

    # 41번 asymmetric neighbor preservation(neighbor_weight=1.0) 결과, seed 0~9.
    # 값 = Asymmetric_H_eval - AdaRound_H_eval (pp). 음수=승(개선), 양수=패(악화).
    outcome_gap = {0: -0.10, 1: +0.10, 2: -0.71, 3: -0.56, 4: +0.42,
                  5: -1.11, 6: -0.95, 7: +1.21, 8: +0.16, 9: -0.79}
    outcome = {s: (f"승 ({g:+.2f}pp)" if g < 0 else f"패 ({g:+.2f}pp)")
              for s, g in outcome_gap.items()}

    print("\n" + "=" * 100)
    print(" seed별 H_eval 구성 분석")
    print("=" * 100)

    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()

        heval_names = sorted(names[c] for c in H_eval)
        s_names = set(names[c] for c in S)

        # H_eval 각 class가 S 안에서 가장 가까운 이웃과의 코사인 유사도
        sim_to_s = []
        for c in H_eval:
            order = neighbor_order[c].tolist()
            nearest_s = next(o for o in order if o in S)
            sim = float(txt_feats[c] @ txt_feats[nearest_s])
            sim_to_s.append((names[c], names[nearest_s], sim))
        sim_to_s.sort(key=lambda x: -x[2])
        avg_sim = np.mean([x[2] for x in sim_to_s])

        n_conf, per_class = confident_heval_anchor_count(h_fp, probe, H_eval)

        print(f"\n--- seed {s} : 39번 결과 = {outcome.get(s, '?')} ---")
        print(f"  H_eval class 20개: {heval_names}")
        print(f"  H_eval -> S 최근접 이웃 평균 cos-sim: {avg_sim:.4f}")
        print(f"  H_eval -> S 최근접 이웃 top5 (가장 가까운 순):")
        for cname, nname, sim in sim_to_s[:5]:
            print(f"    {cname:15s} <-> {nname:15s}  sim={sim:.4f}")
        print(f"  H_eval confident anchor 총 개수(FP 기준, probe {len(probe)}장): {n_conf}")
        top_classes = sorted(per_class.items(), key=lambda x: -x[1])[:5]
        print(f"  가장 많이 등장한 H_eval class: "
              f"{[(names[c], n) for c, n in top_classes]}")

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'gap(pp)':>8} | {'결과':>12} | {'avg_sim(H_eval->S)':>19} | {'n_confident':>11}")
    gaps, avg_sims_all, n_confs_all = [], [], []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_eval = perm[60:80].tolist()
        sims = []
        for c in H_eval:
            order = neighbor_order[c].tolist()
            nearest_s = next(o for o in order if o in S)
            sims.append(float(txt_feats[c] @ txt_feats[nearest_s]))
        n_conf, _ = confident_heval_anchor_count(h_fp, probe, H_eval)
        gap = outcome_gap.get(s)
        tag = outcome.get(s, "?")
        gap_str = f"{gap:+.2f}" if gap is not None else "?"
        print(f"{s:>4} | {gap_str:>8} | {tag:>12} | {np.mean(sims):>19.4f} | {n_conf:>11}")
        if gap is not None:
            gaps.append(gap); avg_sims_all.append(np.mean(sims)); n_confs_all.append(n_conf)

    h_fp.close()

    if len(gaps) >= 3:
        gaps_a = np.array(gaps); sims_a = np.array(avg_sims_all); nconf_a = np.array(n_confs_all)
        corr_sim = np.corrcoef(gaps_a, sims_a)[0, 1]
        corr_n = np.corrcoef(gaps_a, nconf_a)[0, 1]
        print(f"\nPearson 상관계수: gap vs avg_sim(H_eval->S) = {corr_sim:+.3f}   "
              f"gap vs n_confident = {corr_n:+.3f}")
        print("  (+ : avg_sim/n_confident가 클수록 gap도 커진다 = 악화와 연관.")
        print("   - : avg_sim/n_confident가 클수록 gap은 작아진다 = 개선과 연관.)")

    print("\n해석 가이드:")
    print("  avg_sim(H_eval->S)이 높을수록 H_eval class가 S class와 text embedding상")
    print("  가깝다 -- S에서 배운 margin 조정이 우연히 H_eval에도 통했을 가능성.")
    print("  n_confident가 seed마다 크게 다르면 flip%의 표본 크기 자체가 달라")
    print("  노이즈에 대한 민감도가 다를 수 있음(작은 n은 flip%가 쉽게 크게 흔들림).")


if __name__ == "__main__":
    main()
