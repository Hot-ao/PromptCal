"""
진단 실험 (09-03 문서 §23~30): PromptCal이 실제로 뭘 하고 있는지 soft/discrete,
H_cal/H_eval로 쪼개서 본다. 새 loss를 만들기 전에 원인을 분리하기 위한 스크립트.

배경: optimize_promptcal()의 ac.soft/ac.ste가 실수로 최적화 내내 False로 남아있던
버그를 고쳤다(9/3 저녁 "checking problem3" 커밋에서 True->False로 바뀐 뒤 안 돌아왔음).
soft=False면 margin/decision loss가 alpha로 gradient를 못 보낸다(reg_loss만 alpha에
연결됨) -- 즉 최근 결과(s5/s6.txt)는 목적함수가 사실상 작동 안 한 상태에서 나온 것일
가능성이 높다. 버그를 고친 뒤 이 스크립트로 재검증한다.

측정 6종 (seed별):
  1. naive / AdaRound / PromptCal(discrete)의 H_eval flip           -- 최종 헤드라인 지표
  2. naive / AdaRound / PromptCal(soft) / PromptCal(discrete)의
     H_cal flip   -- H_eval flip과 동일한 공식을, H_eval 대신 H_cal을 가리고 측정.
                      "학습이 자기 타깃(H_cal)에서라도 실제로 통했는가"를 본다.
  3. 위와 동일 조합의 H_cal margin(margin_loss 값, S+H_cal 열에서)

Case 판정 (09-03 문서 §23):
  A. soft->discrete mismatch : PromptCal soft H_cal 개선, discrete H_cal 개선 안 됨
  B. H_cal->H_eval 실패      : discrete H_cal 개선, H_eval flip 개선 안 됨
  C. objective가 애초에 무력 : soft H_cal조차 AdaRound보다 안 좋음
  D. 방향 유효               : discrete H_cal 개선 AND H_eval flip 개선

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/29_diag_transfer.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
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


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=3e-3, reg_weight=0.1, k=5, post_train_hook=None):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "promptcal":
        convert_to_adaround(m.model)
        optimize_promptcal(m.model, fp.model, calib, device, pidx, iters=iters,
                           lr=lr, reg_weight=reg_weight, k=k, verbose=True,
                           post_train_hook=post_train_hook)
    return m


def group_flip(h_fp, h_q, imgs, group_idx, conf=0.25):
    """group_idx 열을 가린 뒤, FP full-top1이 group_idx에 속하는 confident anchor에서
    K-only(group 제거) argmax의 FP vs quant 불일치율. heldout_flip의 일반화 버전
    (H_eval 대신 임의의 group을 가림 -- H_cal 진단에도 같은 공식을 쓰기 위함)."""
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
    """margin_loss(학습에 쓴 것과 동일 공식)를 confident anchor x pidx 열에서
    이미지 전체에 대해 anchor-가중 평균. 학습 목적함수가 실제로 그 값을 줄였는지 확인용."""
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
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
    nv = build(YOLOWorld, args.model, names, device, calib, "naive")
    print("[build] AdaRound (baseline, seed-무관)")
    ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_nv = SimilarityHarness(nv.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)

    print("\n" + "=" * 100)
    print(" 진단: soft/discrete x H_cal/H_eval 분리 측정")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
        pidx = S + H_cal

        print(f"\n--- seed {s} ---")

        # AdaRound: H_cal에 대해 "아무것도 안 한" 기준선
        ad_hcal_flip, ad_hcal_n = group_flip(h_fp, h_ad, calib, H_cal)
        ad_hcal_margin = group_margin(h_fp, h_ad, calib, pidx, k=args.k)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)
        nv_heval_flip, _ = group_flip(h_fp, h_nv, probe, H_eval)

        soft_capture = {}

        def snapshot_soft(qmodel):
            h_soft = SimilarityHarness(qmodel, device=device)
            f, n = group_flip(h_fp, h_soft, calib, H_cal)
            m = group_margin(h_fp, h_soft, calib, pidx, k=args.k)
            soft_capture["flip"] = f
            soft_capture["flip_n"] = n
            soft_capture["margin"] = m
            h_soft.close()

        pc = build(YOLOWorld, args.model, names, device, calib, "promptcal", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, reg_weight=args.reg_weight,
                  k=args.k, post_train_hook=snapshot_soft)

        h_pc = SimilarityHarness(pc.model, device=device)
        pc_hcal_flip_disc, pc_hcal_n = group_flip(h_fp, h_pc, calib, H_cal)
        pc_hcal_margin_disc = group_margin(h_fp, h_pc, calib, pidx, k=args.k)
        pc_heval_flip, _ = group_flip(h_fp, h_pc, probe, H_eval)
        h_pc.close()

        print(f"\n  [H_cal, n_anchor≈{ad_hcal_n}]  (n_anchor은 confident+H_cal-top1 anchor 수)")
        print(f"    AdaRound(무관)        flip={ad_hcal_flip:6.2f}%  margin={ad_hcal_margin:.4f}")
        print(f"    PromptCal soft        flip={soft_capture['flip']:6.2f}%  margin={soft_capture['margin']:.4f}  (n={soft_capture['flip_n']})")
        print(f"    PromptCal discrete    flip={pc_hcal_flip_disc:6.2f}%  margin={pc_hcal_margin_disc:.4f}  (n={pc_hcal_n})")
        print(f"  [H_eval, 최종 헤드라인]")
        print(f"    naive     flip={nv_heval_flip:6.2f}%")
        print(f"    AdaRound  flip={ad_heval_flip:6.2f}%")
        print(f"    PromptCal flip={pc_heval_flip:6.2f}%")

        rows.append(dict(
            seed=s,
            ad_hcal_flip=ad_hcal_flip, ad_hcal_margin=ad_hcal_margin,
            pc_soft_hcal_flip=soft_capture["flip"], pc_soft_hcal_margin=soft_capture["margin"],
            pc_disc_hcal_flip=pc_hcal_flip_disc, pc_disc_hcal_margin=pc_hcal_margin_disc,
            nv_heval=nv_heval_flip, ad_heval=ad_heval_flip, pc_heval=pc_heval_flip,
        ))
        h_pc = None  # already closed

    h_fp.close(); h_nv.close(); h_ad.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    hdr = (f"{'seed':>4} | {'Ada_Hcal':>9} | {'PCsoft_Hcal':>11} | {'PCdisc_Hcal':>11} | "
           f"{'Ada_Hcal_m':>10} | {'PCsoft_Hcal_m':>13} | {'PCdisc_Hcal_m':>13} | "
           f"{'nv_Heval':>9} | {'Ada_Heval':>9} | {'PC_Heval':>9}")
    print(hdr)
    for r in rows:
        print(f"{r['seed']:>4} | {r['ad_hcal_flip']:>8.2f}% | {r['pc_soft_hcal_flip']:>10.2f}% | "
              f"{r['pc_disc_hcal_flip']:>10.2f}% | {r['ad_hcal_margin']:>10.4f} | "
              f"{r['pc_soft_hcal_margin']:>13.4f} | {r['pc_disc_hcal_margin']:>13.4f} | "
              f"{r['nv_heval']:>8.2f}% | {r['ad_heval']:>8.2f}% | {r['pc_heval']:>8.2f}%")

    print("\n판정 기준 (09-03 문서 §23):")
    print("  soft H_cal flip < Ada_Hcal flip  이고  discrete H_cal flip이 그 개선을 못 지키면 -> Case A (soft->discrete mismatch)")
    print("  discrete H_cal flip < Ada_Hcal flip 인데 PC_Heval >= Ada_Heval 이면          -> Case B (H_cal->H_eval 일반화 실패)")
    print("  soft H_cal flip조차 Ada_Hcal보다 안 좋으면                                    -> Case C (objective 자체가 무력)")
    print("  discrete H_cal flip 개선 AND PC_Heval < Ada_Heval 이면                        -> Case D (방향 유효, 다음은 seed 확대)")


if __name__ == "__main__":
    main()
