"""
방향 C(연속 s_mult) 검증 (09-05 진단 후속, 33번 다음 단계).

33번에서 AdaRound warm-start(1000 iters, 완전 수렴)를 거치면 alpha가
h→0/1≈99%로 포화돼서, 그 위에 얹은 semantic objective가 rounding 임계값을
못 넘고 사실상 무력화된다는 게 확인됐다(3 seed 모두 AdaRound와 discrete
모델이 bit-for-bit 동일). 즉 discrete alpha rounding은 "포화 이전(raw init,
29/31에서 확인한 강한 negative transfer)"과 "포화 이후(33, no-op)" 둘 중
하나뿐이고, 중간의 유효한 지점을 잡기가 구조적으로 fragile하다.

이 스크립트는 아예 다른 축을 쓴다: src/quant/promptcal.py의
optimize_promptcal_scale("방향 C")은 weight rounding(alpha)은 건드리지 않고
(soft=False, alpha.requires_grad_(False) -- round-to-nearest로 고정) 대신
연속값인 activation quantization scale multiplier(s_mult)를 margin_loss로
학습한다. 연속 파라미터라 포화-절벽 문제가 없다.

주의(비교 축이 다름): 이 조건은 weight 쪽은 naive와 동일한 round-to-nearest이고
activation scale만 조정된다. AdaRound/PromptCal(alpha 조건)은 반대로 weight
rounding을 조정한다. 그래서 "PromptCal-scale vs AdaRound"는 서로 다른 지렛대를
비교하는 것이고, "PromptCal-scale vs naive"가 이 방향의 순수 marginal effect를
보는 더 공정한 비교다. 두 비교를 모두 표에 남긴다.

pidx=S(31/33과 동일 조건)로 H_cal은 objective에서 배제, H_eval만 최종 헤드라인.

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/34_promptcal_scale.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.promptcal import optimize_promptcal_scale, margin_loss


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


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "promptcal_scale":
        convert_to_adaround(m.model)
        optimize_promptcal_scale(m.model, fp.model, calib, device, pidx, iters=iters,
                                 lr=lr, k=k, verbose=True)
    return m


def group_flip(h_fp, h_q, imgs, group_idx, conf=0.25):
    Gm = torch.zeros(80, dtype=torch.bool)
    Gm[group_idx] = True
    tot = fl = 0
    for i, t in enumerate(imgs):
        sf = h_fp.run_image(t, i).sim
        sq = h_q.run_image(t, i).sim
        prob = sf.sigmoid()
        mp, c_fp = prob.max(-1)
        conf_m = mp > conf
        target = conf_m & Gm[c_fp]
        if target.sum() == 0:
            continue
        idx = target.nonzero(as_tuple=True)[0]
        fp_K = sf.clone(); fp_K[:, Gm] = -1e9
        q_K = sq.clone();  q_K[:, Gm] = -1e9
        fl += int((fp_K.argmax(-1)[idx] != q_K.argmax(-1)[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def group_margin(h_fp, h_q, imgs, pidx, k=5, boundary_w=3.0, conf_thres=0.25):
    pidx_t = torch.tensor(pidx, dtype=torch.long)
    tot_loss, tot_n = 0.0, 0
    for i, t in enumerate(imgs):
        sf = h_fp.run_image(t, i).sim
        sq = h_q.run_image(t, i).sim
        prob = sf.sigmoid()
        mp, _ = prob.max(-1)
        conf_m = mp > conf_thres
        if conf_m.sum() == 0:
            continue
        sf_c = sf[conf_m][:, pidx_t]
        sq_c = sq[conf_m][:, pidx_t]
        n = sf_c.shape[0]
        ml = float(margin_loss(sq_c, sf_c, k=k, boundary_w=boundary_w))
        tot_loss += ml * n
        tot_n += n
    return tot_loss / max(tot_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    names = load_coco_names()

    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib = [preprocess(p, args.imgsz, device) for p in imgs[:args.calib]]
    probe = [preprocess(p, args.imgsz, device) for p in imgs[args.calib:args.calib + args.eval]]

    print("[load] FP")
    fp = YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    print("[build] naive")
    nv = YOLOWorld(args.model); nv.set_classes(names); nv.fuse()
    wrap_convs(nv.model, 8, 8); nv.model.to(device).eval(); calibrate(nv.model, calib, device=device)
    print("[build] AdaRound (baseline, seed-무관)")
    ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_nv = SimilarityHarness(nv.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)

    print("\n" + "=" * 100)
    print(" 방향 C: PromptCal-scale(연속 s_mult, S-only) vs naive vs AdaRound")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
        pidx = S

        print(f"\n--- seed {s} ---")

        nv_s_flip, _ = group_flip(h_fp, h_nv, calib, S)
        ad_s_flip, _ = group_flip(h_fp, h_ad, calib, S)
        ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
        nv_heval_flip, _ = group_flip(h_fp, h_nv, probe, H_eval)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)

        pc = build(YOLOWorld, args.model, names, device, calib, "promptcal_scale", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, k=args.k)
        h_pc = SimilarityHarness(pc.model, device=device)
        pc_s_flip, _ = group_flip(h_fp, h_pc, calib, S)
        pc_s_margin = group_margin(h_fp, h_pc, calib, S, k=args.k)
        pc_hcal_flip, _ = group_flip(h_fp, h_pc, calib, H_cal)
        pc_heval_flip, _ = group_flip(h_fp, h_pc, probe, H_eval)
        h_pc.close()

        print(f"\n  [S, 최적화 타깃]")
        print(f"    naive              flip={nv_s_flip:6.2f}%")
        print(f"    AdaRound           flip={ad_s_flip:6.2f}%  margin={ad_s_margin:.4f}")
        print(f"    PromptCal-scale    flip={pc_s_flip:6.2f}%  margin={pc_s_margin:.4f}")
        print(f"  [H_cal, 미사용 대조군]")
        print(f"    PromptCal-scale    flip={pc_hcal_flip:6.2f}%")
        print(f"  [H_eval, 최종 헤드라인]")
        print(f"    naive              flip={nv_heval_flip:6.2f}%")
        print(f"    AdaRound           flip={ad_heval_flip:6.2f}%")
        print(f"    PromptCal-scale    flip={pc_heval_flip:6.2f}%")

        rows.append(dict(seed=s, nv_s=nv_s_flip, ad_s=ad_s_flip, pc_s=pc_s_flip,
                         ad_s_m=ad_s_margin, pc_s_m=pc_s_margin, pc_hcal=pc_hcal_flip,
                         nv_heval=nv_heval_flip, ad_heval=ad_heval_flip, pc_heval=pc_heval_flip))

    h_fp.close(); h_nv.close(); h_ad.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'nv_S':>6} | {'Ada_S':>7} | {'PCs_S':>7} | {'Ada_S_m':>8} | {'PCs_S_m':>8} | "
          f"{'nv_Heval':>9} | {'Ada_Heval':>9} | {'PCs_Heval':>9}")
    for r in rows:
        print(f"{r['seed']:>4} | {r['nv_s']:>5.2f}% | {r['ad_s']:>6.2f}% | {r['pc_s']:>6.2f}% | "
              f"{r['ad_s_m']:>8.4f} | {r['pc_s_m']:>8.4f} | "
              f"{r['nv_heval']:>8.2f}% | {r['ad_heval']:>8.2f}% | {r['pc_heval']:>8.2f}%")

    print("\n판정:")
    print("  PCs_S margin이 Ada_S_m보다 개선 안 됐으면 -> 이 실험도 objective가 작동 안 함(무효).")
    print("  (naive 대비 순수 marginal effect) PCs_Heval vs nv_Heval:")
    print("     PCs_Heval < nv_Heval        -> 연속 파라미터화가 discrete alpha의 negative")
    print("                                    transfer 문제를 실제로 피한다. 유망한 방향.")
    print("     PCs_Heval >= nv_Heval       -> 파라미터화를 바꿔도 같은 문제가 재현됨.")
    print("                                    '어떤 형태로든 semantic objective로 alpha/scale을")
    print("                                    데이터 기반으로 움직이면 held-out이 나빠진다'는")
    print("                                    가설이 더 강하게 지지됨(파라미터화 무관 현상).")


if __name__ == "__main__":
    main()
