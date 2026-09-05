"""
S-only control (09-05 진단 후속): Case B(H_cal->H_eval 전이 실패)가 확정된 뒤,
"H_cal을 objective에 넣는 것 자체"가 문제인지 아니면 "alpha를 어떤 semantic
objective로든 움직이면 held-out 일반화가 구조적으로 나빠지는 것"인지를 분리한다.

29_diag_transfer.py와 동일한 seed별 S/H_cal/H_eval 분할을 쓰되, PromptCal
optimize 대상 prompt pool을 pidx = S+H_cal 대신 pidx = S로만 좁힌다.
즉 H_cal은 objective에서 완전히 배제되고(어떤 gradient도 받지 않음), 오직
S(이미 "seen"인 40개)에 대해서만 margin/decision을 맞춘다.

측정 (seed별):
  1. S flip/margin      -- 최적화 타깃 그룹. 여기가 AdaRound보다 개선돼야
                           "objective가 작동은 했다"는 확인이 된다.
  2. H_cal flip/margin  -- objective가 전혀 건드리지 않은 대조군. AdaRound와
                           비슷해야 정상(오염이 없다는 뜻).
  3. H_eval flip        -- 최종 헤드라인. 여기가 29_full.txt(H_cal 포함 버전)와
                           비슷하게 나빠지면, 문제는 "H_cal 특정 identity를
                           외우는 것"이 아니라 "이 alpha 최적화 방식 자체가
                           held-out 일반화를 해친다"는 뜻이 된다.

해석:
  - S-only인데도 H_eval flip이 29_full.txt 수준으로 나빠짐
      -> 일반적 부작용 가설 지지. capacity 제약(recon_w 강화, 건드릴 conv
         레이어 제한 등) 쪽으로 방향 전환 필요.
  - S-only에서는 H_eval flip이 AdaRound 수준을 유지
      -> H_cal(held-out)을 objective에 넣는 것 자체가 원인. class-agnostic
         reformulation을 계속 파는 게 맞다(단 30번 시도는 이미 실패했으므로
         30과 다른 방식 필요).

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/31_s_only_control.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --device 0
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
    elif mode == "promptcal":
        convert_to_adaround(m.model)
        optimize_promptcal(m.model, fp.model, calib, device, pidx, iters=iters,
                           lr=lr, reg_weight=reg_weight, k=k, verbose=True)
    return m


def group_flip(h_fp, h_q, imgs, group_idx, conf=0.25):
    """group_idx 열을 가린 뒤, FP full-top1이 group_idx에 속하는 confident anchor에서
    K-only(group 제거) argmax의 FP vs quant 불일치율."""
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
    """margin_loss(학습에 쓴 것과 동일 공식)를 confident anchor x pidx 열에서 평균."""
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
    print("[build] naive")
    nv = build(YOLOWorld, args.model, names, device, calib, "naive")
    print("[build] AdaRound (baseline, seed-무관)")
    ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_nv = SimilarityHarness(nv.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)

    print("\n" + "=" * 100)
    print(" S-only control: H_cal을 objective에서 완전 배제, pidx=S(40개)만 사용")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
        pidx = S  # <-- 핵심 변경: H_cal 제외, S만 objective에 사용

        print(f"\n--- seed {s} ---")

        ad_s_flip, _ = group_flip(h_fp, h_ad, calib, S)
        ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
        ad_hcal_flip, _ = group_flip(h_fp, h_ad, calib, H_cal)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)
        nv_heval_flip, _ = group_flip(h_fp, h_nv, probe, H_eval)

        pc = build(YOLOWorld, args.model, names, device, calib, "promptcal", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, reg_weight=args.reg_weight,
                  k=args.k)

        h_pc = SimilarityHarness(pc.model, device=device)
        pc_s_flip, _ = group_flip(h_fp, h_pc, calib, S)
        pc_s_margin = group_margin(h_fp, h_pc, calib, S, k=args.k)
        pc_hcal_flip, _ = group_flip(h_fp, h_pc, calib, H_cal)
        pc_heval_flip, _ = group_flip(h_fp, h_pc, probe, H_eval)
        h_pc.close()

        print(f"\n  [S, 최적화 타깃 -- 개선돼야 objective가 작동했다는 뜻]")
        print(f"    AdaRound  flip={ad_s_flip:6.2f}%  margin={ad_s_margin:.4f}")
        print(f"    PromptCal flip={pc_s_flip:6.2f}%  margin={pc_s_margin:.4f}")
        print(f"  [H_cal, objective가 전혀 안 건드린 대조군 -- AdaRound와 비슷해야 정상]")
        print(f"    AdaRound  flip={ad_hcal_flip:6.2f}%")
        print(f"    PromptCal flip={pc_hcal_flip:6.2f}%")
        print(f"  [H_eval, 최종 헤드라인]")
        print(f"    naive     flip={nv_heval_flip:6.2f}%")
        print(f"    AdaRound  flip={ad_heval_flip:6.2f}%")
        print(f"    PromptCal flip={pc_heval_flip:6.2f}%")

        rows.append(dict(
            seed=s,
            ad_s_flip=ad_s_flip, ad_s_margin=ad_s_margin,
            pc_s_flip=pc_s_flip, pc_s_margin=pc_s_margin,
            ad_hcal_flip=ad_hcal_flip, pc_hcal_flip=pc_hcal_flip,
            nv_heval=nv_heval_flip, ad_heval=ad_heval_flip, pc_heval=pc_heval_flip,
        ))

    h_fp.close(); h_nv.close(); h_ad.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    hdr = (f"{'seed':>4} | {'Ada_S':>7} | {'PC_S':>7} | {'Ada_S_m':>8} | {'PC_S_m':>8} | "
           f"{'Ada_Hcal':>9} | {'PC_Hcal':>8} | {'nv_Heval':>9} | {'Ada_Heval':>9} | {'PC_Heval':>9}")
    print(hdr)
    for r in rows:
        print(f"{r['seed']:>4} | {r['ad_s_flip']:>6.2f}% | {r['pc_s_flip']:>6.2f}% | "
              f"{r['ad_s_margin']:>8.4f} | {r['pc_s_margin']:>8.4f} | "
              f"{r['ad_hcal_flip']:>8.2f}% | {r['pc_hcal_flip']:>7.2f}% | "
              f"{r['nv_heval']:>8.2f}% | {r['ad_heval']:>8.2f}% | {r['pc_heval']:>8.2f}%")

    print("\n판정:")
    print("  PC_S flip/margin이 Ada_S보다 개선 안 됐다면 -> 이 실험 자체가 무효(objective가 작동 안 함).")
    print("  PC_S 개선 확인된 상태에서:")
    print("    PC_Heval이 29_full.txt(H_cal 포함) 수준으로 여전히 악화")
    print("       -> 일반적 부작용 가설: alpha를 어느 방향으로 움직이든 held-out이 나빠진다.")
    print("          capacity 제약(recon_w 강화, 대상 conv 레이어 축소) 방향으로 전환.")
    print("    PC_Heval이 AdaRound 수준 유지")
    print("       -> H_cal(held-out)을 objective에 넣는 것 자체가 원인. class-agnostic")
    print("          reformulation을 계속 파되 30번과 다른 방식 필요.")


if __name__ == "__main__":
    main()
