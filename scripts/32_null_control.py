"""
Zero-semantic null control (09-05 진단 후속, 31번 S-only 다음 단계).

31번(S-only)에서 H_cal을 objective에서 완전히 뺐는데도 H_eval flip이 29번
(S+H_cal)과 거의 같은 크기로 나빠졌다. 그래서 "H_cal 특정 identity를 외우는
것"이 아니라 "alpha를 어느 방향으로든 움직이면 held-out이 나빠진다"는 가설이
지지됐다. 이 스크립트는 그 가설을 한 번 더 좁힌다.

이 스크립트는 PromptCal loss에서 semantic 성분(decision loss, margin loss)을
완전히 제거하고 alpha regularizer(reg_loss)만 남긴 채, PromptCal과 완전히
동일한 초기화(raw convert_to_adaround, AdaRound reconstruction warm-start
없음)/lr/iters/beta로 alpha를 "최적화"한다. reg_loss는 alpha 자체의 함수라서
forward pass도 FP 데이터도 전혀 필요 없다 -- 즉 이 조건(null)은 "어떤 데이터/
목표에도 의존하지 않고 초기 soft alpha를 그냥 hardening만 시킨 것"과 같다.
seed/prompt split에 의존하지 않으므로 null 모델은 1회만 빌드한다(naive/
AdaRound와 동일하게 "seed-무관").

왜 중요한가: 지금까지 PromptCal(20/29/31번)의 build()는 AdaRound의
reconstruction 최적화(optimize_adaround, 1000 iters)를 거치지 않고 raw init
에서 바로 semantic objective를 얹었다(20_promptcal_minimal.py의 build() 주석:
"AdaRound 초기화 없이... 초기화 제거"). 즉 "AdaRound 대비 PromptCal이 나쁘다"
는 지금까지의 비교에는 두 가지 다른 요인이 섞여 있었을 수 있다:
  (a) semantic objective 자체가 held-out 일반화를 해친다
  (b) PromptCal이 애초에 AdaRound의 reconstruction 최적화를 거치지 않아서
      비교 자체가 불공정하다(단순히 초기화/최적화 품질이 나쁨)

이 null(semantic=0, reg만, raw init)이 이미 AdaRound baseline보다 H_eval이
나쁘게 나온다면, PromptCal 실패의 상당 부분이 (a)가 아니라 (b)일 가능성이
높아진다 -- 그 경우 다음 실험은 "AdaRound로 먼저 warm-start한 뒤 그 위에
semantic objective를 얹기"가 되어야 한다.
반대로 null이 AdaRound와 비슷한 H_eval을 유지한다면 (b)는 기각되고, 지금까지
관찰한 degradation은 진짜로 (a) semantic objective(어떤 형태로든 alpha를
데이터 기반으로 움직이는 것) 자체에서 온다는 뜻이 된다.

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/32_null_control.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround, list_adaround_convs, h_alpha
from src.quant.promptcal import margin_loss


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


def optimize_null(quant_model, iters, lr, reg_weight, beta=2.0, verbose=True):
    """PromptCal과 동일한 alpha regularizer(reg_loss, beta=2.0 고정)만 최적화.
    decision/margin 없음 -> forward pass, FP 데이터 전부 불필요.
    PromptCal의 optimize_promptcal과 동일한 초기화(raw convert_to_adaround)/
    lr/iters/beta/optimizer/grad-clip을 맞춰 semantic 항만 뺀 대조군."""
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = True
        ac.ste = True

    alphas = [ac.alpha for ac in ada]
    alpha_before = [a.detach().clone() for a in alphas]
    opt = torch.optim.Adam(alphas, lr=lr)

    if verbose:
        print(f"[null] alpha {len(ada)}개, reg-only(beta={beta}), FP/calib 데이터 미사용")

    for it in range(iters):
        opt.zero_grad()
        reg = sum(ac.reg_loss(beta=beta, reduction="mean") for ac in ada) / len(ada)
        loss = reg_weight * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(alphas, max_norm=1.0)
        opt.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            hc = sum(float(((h_alpha(a) < 0.05) | (h_alpha(a) > 0.95)).float().mean())
                    for a in alphas) / len(alphas) * 100
            print(f"  [{it+1}/{iters}] reg={float(reg.detach()):.4f} h→0/1={hc:.0f}%")

    for ac in ada:
        ac.soft = False
        ac.ste = False

    if verbose:
        tot_change = sum(float((a.detach() - b).abs().sum()) for a, b in zip(alphas, alpha_before))
        print(f"[null] 완료 (alpha 총 변화량={tot_change:.2f})")


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1000,
          lr=3e-3, reg_weight=0.1):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "null":
        convert_to_adaround(m.model)
        optimize_null(m.model, iters=iters, lr=lr, reg_weight=reg_weight)
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
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--reg-weight", type=float, default=0.1)
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
    print("[build] Null control (semantic=0, reg only, seed-무관)")
    nl = build(YOLOWorld, args.model, names, device, calib, "null",
              iters=args.iters, lr=args.lr, reg_weight=args.reg_weight)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)
    h_nl = SimilarityHarness(nl.model, device=device)

    print("\n" + "=" * 100)
    print(" Null control: semantic loss=0, reg_loss만으로 alpha 최적화(raw init, 데이터 미사용)")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()

        ad_s_flip, _ = group_flip(h_fp, h_ad, calib, S)
        ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
        ad_hcal_flip, _ = group_flip(h_fp, h_ad, calib, H_cal)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)

        nl_s_flip, _ = group_flip(h_fp, h_nl, calib, S)
        nl_s_margin = group_margin(h_fp, h_nl, calib, S, k=args.k)
        nl_hcal_flip, _ = group_flip(h_fp, h_nl, calib, H_cal)
        nl_heval_flip, _ = group_flip(h_fp, h_nl, probe, H_eval)

        print(f"\n--- seed {s} ---")
        print(f"  [S]      AdaRound flip={ad_s_flip:6.2f}% margin={ad_s_margin:.4f}   "
              f"Null flip={nl_s_flip:6.2f}% margin={nl_s_margin:.4f}")
        print(f"  [H_cal]  AdaRound flip={ad_hcal_flip:6.2f}%   Null flip={nl_hcal_flip:6.2f}%")
        print(f"  [H_eval] AdaRound flip={ad_heval_flip:6.2f}%   Null flip={nl_heval_flip:6.2f}%")

        rows.append(dict(seed=s, ad_s=ad_s_flip, nl_s=nl_s_flip,
                         ad_s_m=ad_s_margin, nl_s_m=nl_s_margin,
                         ad_hcal=ad_hcal_flip, nl_hcal=nl_hcal_flip,
                         ad_heval=ad_heval_flip, nl_heval=nl_heval_flip))

    h_fp.close(); h_ad.close(); h_nl.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'Ada_S':>7} | {'Null_S':>7} | {'Ada_S_m':>8} | {'Null_S_m':>8} | "
          f"{'Ada_Hcal':>9} | {'Null_Hcal':>9} | {'Ada_Heval':>9} | {'Null_Heval':>10}")
    for r in rows:
        print(f"{r['seed']:>4} | {r['ad_s']:>6.2f}% | {r['nl_s']:>6.2f}% | "
              f"{r['ad_s_m']:>8.4f} | {r['nl_s_m']:>8.4f} | "
              f"{r['ad_hcal']:>8.2f}% | {r['nl_hcal']:>8.2f}% | "
              f"{r['ad_heval']:>8.2f}% | {r['nl_heval']:>9.2f}%")

    print("\n판정:")
    print("  Null_Heval이 Ada_Heval과 비슷 -> raw init 자체는 문제가 아님.")
    print("     지금까지의 degradation은 진짜로 semantic objective(어떤 형태로든")
    print("     alpha를 데이터 기반으로 움직이는 것) 자체에서 온다.")
    print("  Null_Heval도 이미 Ada_Heval보다 나쁨 -> raw init(AdaRound reconstruction")
    print("     warm-start 없음) 자체가 원인의 상당 부분. 다음 실험은 AdaRound로")
    print("     먼저 warm-start한 뒤 그 위에 semantic objective를 얹는 조건으로 재설계.")


if __name__ == "__main__":
    main()
