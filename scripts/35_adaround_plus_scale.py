"""
AdaRound weight rounding + 방향 C(연속 s_mult) scale 결합 (09-05 진단 후속, 34번 다음 단계).

34번에서 순수 방향 C(활동 스케일만, weight rounding은 naive 수준 round-to-nearest)가
naive 대비 H_eval을 3 seed 모두 개선시켰다(첫 긍정 신호). 하지만 weight rounding
자체는 AdaRound의 reconstruction 최적화를 안 거쳤기 때문에 AdaRound 자체보다는
대부분 seed에서 약간 못 미쳤다.

이 스크립트는 그 둘을 합친다:
  1. optimize_adaround로 weight rounding(alpha)을 먼저 완전히 최적화(1000 iters,
     33번에서 확인했듯 이 시점에 h→0/1≈99%로 포화됨).
  2. alpha는 그대로 고정(optimize_promptcal_scale이 alpha.requires_grad_(False)로
     freeze)하고, 그 위에 연속 s_mult만 semantic margin objective(S-only)로 추가
     최적화.

33번과의 차이: 33번은 포화된 alpha "자체"를 다시 움직이려다 실패했다(연속
파라미터가 없어서 gradient가 죽음). 이번엔 alpha는 건드리지 않고 별도의 연속
knob(activation scale)만 움직이므로 포화 문제 자체가 없다 -- 34번에서 이미
확인된 성질.

비교축: AdaRound(weight rounding만) vs AdaRound+scale(동일 weight rounding +
추가 scale 튜닝). weight rounding이 완전히 동일하므로 이번이 지금까지 중
가장 공정한 "semantic objective의 순마진 효과" 비교다.

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/35_adaround_plus_scale.py \
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
    elif mode == "combined":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
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
    print("[build] AdaRound (baseline, seed-무관)")
    ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)

    print("\n" + "=" * 100)
    print(" AdaRound(weight) + 방향 C(scale, S-only) 결합 vs AdaRound 단독")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
        pidx = S

        print(f"\n--- seed {s} ---")

        ad_s_flip, _ = group_flip(h_fp, h_ad, calib, S)
        ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)

        cb = build(YOLOWorld, args.model, names, device, calib, "combined", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, k=args.k)
        h_cb = SimilarityHarness(cb.model, device=device)
        cb_s_flip, _ = group_flip(h_fp, h_cb, calib, S)
        cb_s_margin = group_margin(h_fp, h_cb, calib, S, k=args.k)
        cb_hcal_flip, _ = group_flip(h_fp, h_cb, calib, H_cal)
        cb_heval_flip, _ = group_flip(h_fp, h_cb, probe, H_eval)
        h_cb.close()

        print(f"\n  [S, 최적화 타깃]")
        print(f"    AdaRound         flip={ad_s_flip:6.2f}%  margin={ad_s_margin:.4f}")
        print(f"    AdaRound+scale   flip={cb_s_flip:6.2f}%  margin={cb_s_margin:.4f}")
        print(f"  [H_cal, 미사용 대조군]")
        print(f"    AdaRound+scale   flip={cb_hcal_flip:6.2f}%")
        print(f"  [H_eval, 최종 헤드라인]")
        print(f"    AdaRound         flip={ad_heval_flip:6.2f}%")
        print(f"    AdaRound+scale   flip={cb_heval_flip:6.2f}%")

        rows.append(dict(seed=s, ad_s=ad_s_flip, cb_s=cb_s_flip,
                         ad_s_m=ad_s_margin, cb_s_m=cb_s_margin, cb_hcal=cb_hcal_flip,
                         ad_heval=ad_heval_flip, cb_heval=cb_heval_flip))

    h_fp.close(); h_ad.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'Ada_S':>7} | {'Comb_S':>7} | {'Ada_S_m':>8} | {'Comb_S_m':>8} | "
          f"{'Ada_Heval':>9} | {'Comb_Heval':>10}")
    for r in rows:
        print(f"{r['seed']:>4} | {r['ad_s']:>6.2f}% | {r['cb_s']:>6.2f}% | "
              f"{r['ad_s_m']:>8.4f} | {r['cb_s_m']:>8.4f} | "
              f"{r['ad_heval']:>8.2f}% | {r['cb_heval']:>9.2f}%")

    print("\n참고: 34번(방향 C 단독, weight=naive) H_eval = 7.67% / 9.12% / 9.07% (seed 0/1/2)")
    print("참고: naive H_eval = 8.14% / 9.56% / 9.18% (seed 0/1/2)")

    print("\n판정:")
    print("  Comb_S_m이 Ada_S_m보다 개선 안 됐으면 -> objective가 이 조합에서 작동 안 함(무효).")
    print("  Comb_Heval < Ada_Heval (동일 weight rounding 대비 순수 semantic 효과)")
    print("     -> AdaRound의 reconstruction 품질을 유지하면서 semantic 보존도 얻는 조합.")
    print("        held-out flip 절대 수치로 지금까지 중 최선. 다음은 seed 확대/논문화.")
    print("  Comb_Heval >= Ada_Heval")
    print("     -> scale 튜닝을 얹는 것 자체가 이미 최적화된 AdaRound 예측을 흔들기만 함.")
    print("        방향 C의 34번 개선은 naive(약한 baseline) 대비였을 뿐, 이미 좋은")
    print("        baseline(AdaRound) 위에서는 추가 이득이 없다는 뜻.")


if __name__ == "__main__":
    main()
