"""
AdaRound warm-start + semantic objective (09-05 진단 후속, 32번 다음 단계).

32번(null control)에서 raw init(AdaRound reconstruction 없이 바로 alpha
regularizer만 적용)도 이미 AdaRound baseline보다 H_eval이 약간 나빴다(+3~9%).
하지만 PromptCal(31번, S-only)의 degradation(+9~21%)은 그보다 뚜렷하게 컸다.
즉 "raw-init 비용"과 "semantic objective 자체의 비용"이 섞여 있었다.

이 스크립트는 그 둘을 분리한다: AdaRound의 reconstruction 최적화
(optimize_adaround, 1000 iters)로 먼저 warm-start한 뒤, 그 위에
  (a) 아무 semantic 목표도 없는 null(reg-only) 추가 최적화
  (b) PromptCal semantic objective(pidx=S, margin+decision) 추가 최적화
를 각각 얹어서 비교한다. pidx=S를 쓴 이유는 31번과 동일 조건으로 맞춰
"H_cal을 objective에 넣었는지"가 아니라 "warm-start 여부"만 다르게 하기 위함.

읽는 법:
  warmstart+null_Heval ≈ Ada_Heval
    -> warm-start 파이프라인 변경 자체는 부작용이 없다(sanity check 통과).
  warmstart+semantic_Heval이 여전히 Ada_Heval보다 뚜렷이 나쁨
    -> raw-init 문제였다는 가설은 기각. semantic objective 자체가 순수하게
       held-out 일반화를 해친다는 결론이 굳어진다(29/31과 동일한 결론이지만
       이번엔 init 교란 요인이 통제된 상태).
  warmstart+semantic_Heval이 Ada_Heval에 훨씬 가까워짐(raw-init PromptCal,
  31번의 +9~21%보다 확연히 개선)
    -> raw-init이 지금까지 관찰된 degradation의 상당 부분을 설명한다.
       앞으로 PromptCal은 반드시 AdaRound warm-start를 거쳐야 한다.

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/33_warmstart_semantic.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround, list_adaround_convs, h_alpha
from src.quant.promptcal import optimize_promptcal, margin_loss


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
    """32번과 동일: semantic 없이 alpha regularizer만. warm-start 이후에 걸면
    이미 h_alpha가 0/1 근처라서 거의 안 움직여야 정상(sanity check)."""
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = True
        ac.ste = True
    alphas = [ac.alpha for ac in ada]
    alpha_before = [a.detach().clone() for a in alphas]
    opt = torch.optim.Adam(alphas, lr=lr)
    if verbose:
        print(f"[null] alpha {len(ada)}개, reg-only(beta={beta}), warm-start 위에 적용")
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


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=3e-3, reg_weight=0.1, k=5):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "warmstart_null":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_null(m.model, iters=iters, lr=lr, reg_weight=reg_weight)
    elif mode == "warmstart_semantic":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_promptcal(m.model, fp.model, calib, device, pidx, iters=iters,
                           lr=lr, reg_weight=reg_weight, k=k, verbose=True)
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
    print("[build] warm-start + null (sanity check, seed-무관)")
    wn = build(YOLOWorld, args.model, names, device, calib, "warmstart_null", fp=fp,
              iters=args.iters, lr=args.lr, reg_weight=args.reg_weight)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)
    h_wn = SimilarityHarness(wn.model, device=device)

    print("\n" + "=" * 100)
    print(" AdaRound warm-start + semantic(S-only) vs AdaRound vs warm-start+null(sanity)")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
        pidx = S  # 31번과 동일 조건(H_cal 미포함)으로 warm-start 유무만 다르게

        print(f"\n--- seed {s} ---")

        ad_s_flip, _ = group_flip(h_fp, h_ad, calib, S)
        ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)

        wn_heval_flip, _ = group_flip(h_fp, h_wn, probe, H_eval)

        ws = build(YOLOWorld, args.model, names, device, calib, "warmstart_semantic", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, reg_weight=args.reg_weight, k=args.k)
        h_ws = SimilarityHarness(ws.model, device=device)
        ws_s_flip, _ = group_flip(h_fp, h_ws, calib, S)
        ws_s_margin = group_margin(h_fp, h_ws, calib, S, k=args.k)
        ws_heval_flip, _ = group_flip(h_fp, h_ws, probe, H_eval)
        h_ws.close()

        print(f"\n  [S, 최적화 타깃]")
        print(f"    AdaRound            flip={ad_s_flip:6.2f}%  margin={ad_s_margin:.4f}")
        print(f"    warmstart+semantic  flip={ws_s_flip:6.2f}%  margin={ws_s_margin:.4f}")
        print(f"  [H_eval, 최종 헤드라인]")
        print(f"    AdaRound            flip={ad_heval_flip:6.2f}%")
        print(f"    warmstart+null      flip={wn_heval_flip:6.2f}%  (sanity, ≈AdaRound 기대)")
        print(f"    warmstart+semantic  flip={ws_heval_flip:6.2f}%")

        rows.append(dict(seed=s, ad_s=ad_s_flip, ws_s=ws_s_flip,
                         ad_s_m=ad_s_margin, ws_s_m=ws_s_margin,
                         ad_heval=ad_heval_flip, wn_heval=wn_heval_flip, ws_heval=ws_heval_flip))

    h_fp.close(); h_ad.close(); h_wn.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'Ada_S':>7} | {'WS_S':>7} | {'Ada_S_m':>8} | {'WS_S_m':>8} | "
          f"{'Ada_Heval':>9} | {'WNull_Heval':>11} | {'WSem_Heval':>10}")
    for r in rows:
        print(f"{r['seed']:>4} | {r['ad_s']:>6.2f}% | {r['ws_s']:>6.2f}% | "
              f"{r['ad_s_m']:>8.4f} | {r['ws_s_m']:>8.4f} | "
              f"{r['ad_heval']:>8.2f}% | {r['wn_heval']:>10.2f}% | {r['ws_heval']:>9.2f}%")

    print("\n참고: 31번(raw-init PromptCal, S-only) H_eval flip = 8.69% / 10.10% / 10.80% (seed 0/1/2)")
    print("\n판정:")
    print("  WNull_Heval ≈ Ada_Heval이 아니면 -> sanity check 실패, warm-start 파이프라인 자체를 점검.")
    print("  WSem_Heval이 Ada_Heval보다 여전히 뚜렷이 나쁨(31번과 비슷한 폭)")
    print("     -> raw-init 문제였다는 가설 기각. semantic objective 자체가 원인(init과 무관).")
    print("  WSem_Heval이 Ada_Heval에 훨씬 가까워짐(31번보다 확연히 개선)")
    print("     -> raw-init이 지금까지 degradation의 상당 부분을 설명. PromptCal은 항상")
    print("        AdaRound warm-start를 거쳐야 함.")


if __name__ == "__main__":
    main()
