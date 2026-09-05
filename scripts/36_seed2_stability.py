"""
seed 2 불안정성 진단 (09-05 진단 후속, 35번 다음 단계).

35번(AdaRound weight + 방향 C scale 결합)에서 seed 0/1은 AdaRound보다 개선됐지만
seed 2는 S margin 자체가 악화됐다(0.1349 -> 0.1415, margin_loss 로그도 수렴 없이
진동: 0.16->0.07->0.08->0.12->0.23->0.24->0.20->0.17->0.05->0.08). lr=1e-2가 이
seed의 데이터 조합에서 너무 커서 발산/진동했을 가능성을 확인한다.

optimize_promptcal_scale에 eval_hook을 추가해서(1500 iters 중 150 iter마다 콜백),
학습 중 실제 group_margin(S)/group_flip(H_eval) 궤적을 직접 찍는다 -- 지금까지는
마지막 1개 이미지의 margin_loss만 로그로 봤는데, 이번엔 전체 calib/probe 기준
지표가 학습 도중 어떻게 움직이는지(발산하는지, 특정 iter에서 최적인지) 본다.

seed 2 고정, lr을 [1e-2(기존), 5e-3, 3e-3, 1e-3]로 스윕한다.

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/36_seed2_stability.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seed 2 --device 0
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
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 5e-3, 3e-3, 1e-3])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
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
    h_fp = SimilarityHarness(fp.model, device=device)

    print("[build] AdaRound (weight rounding, seed-무관, 재사용될 base)")
    def build_adaround():
        m = YOLOWorld(args.model); m.set_classes(names); m.fuse()
        wrap_convs(m.model, 8, 8); m.model.to(device).eval()
        calibrate(m.model, calib, device=device)
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        return m

    ad_ref = build_adaround()
    h_ad = SimilarityHarness(ad_ref.model, device=device)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
    pidx = S

    ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
    ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)
    print(f"\n[seed {args.seed} 기준선] AdaRound  S_margin={ad_s_margin:.4f}  H_eval flip={ad_heval_flip:.2f}%")
    h_ad.close()

    print("\n" + "=" * 100)
    print(f" seed {args.seed} lr sweep: 학습 중 S_margin/H_eval flip 궤적")
    print("=" * 100)

    results = []
    for lr in args.lrs:
        print(f"\n--- lr={lr} ---")
        cb = build_adaround()  # 매 lr마다 동일한 AdaRound weight에서 새로 시작
        h_cb = SimilarityHarness(cb.model, device=device)
        traj = []

        def eval_hook(it, model, h_cb=h_cb):
            m = group_margin(h_fp, h_cb, calib, S, k=args.k)
            f, _ = group_flip(h_fp, h_cb, probe, H_eval)
            traj.append((it, m, f))
            print(f"    [eval@{it}] S_margin={m:.4f}  H_eval flip={f:.2f}%")

        optimize_promptcal_scale(cb.model, fp.model, calib, device, pidx,
                                 iters=args.iters, lr=lr, k=args.k, verbose=True,
                                 eval_hook=eval_hook)

        final_s_margin = group_margin(h_fp, h_cb, calib, S, k=args.k)
        final_heval, _ = group_flip(h_fp, h_cb, probe, H_eval)
        h_cb.close()

        best_it, best_m, best_f = min(traj, key=lambda r: r[2])  # H_eval flip 기준 최선 체크포인트
        print(f"  최종: S_margin={final_s_margin:.4f}  H_eval flip={final_heval:.2f}%  "
              f"(궤적 중 최선 체크포인트: iter={best_it} H_eval={best_f:.2f}% S_margin={best_m:.4f})")
        results.append(dict(lr=lr, final_s_margin=final_s_margin, final_heval=final_heval,
                            best_it=best_it, best_heval=best_f, best_s_margin=best_m, traj=traj))

    h_fp.close()

    print("\n" + "=" * 100)
    print(f" 요약 표 (seed {args.seed}, 기준선 AdaRound: S_margin={ad_s_margin:.4f} H_eval={ad_heval_flip:.2f}%)")
    print("=" * 100)
    print(f"{'lr':>8} | {'final_S_m':>10} | {'final_Heval':>11} | {'best_it':>8} | {'best_Heval':>10}")
    for r in results:
        print(f"{r['lr']:>8} | {r['final_s_margin']:>10.4f} | {r['final_heval']:>10.2f}% | "
              f"{r['best_it']:>8} | {r['best_heval']:>9.2f}%")

    print("\n판정:")
    print("  어떤 lr에서 final_S_m < 0.1349(AdaRound) AND final_Heval < 8.90%(AdaRound)면")
    print("     -> lr을 낮추는 것만으로 seed 2 불안정성이 해결됨. 그 lr로 seed 0/1도")
    print("        재검증해서 여전히 이기는지 확인 필요.")
    print("  모든 lr에서 계속 나쁘거나 궤적이 계속 진동하면")
    print("     -> lr 문제가 아니라 이 seed의 prompt 조합/데이터 자체가 이 방법과 안 맞는")
    print("        경우일 가능성. best_it 체크포인트(early stopping)가 유의미하게 좋다면")
    print("        '너무 오래 학습하면 오히려 나빠진다'는 방향으로 재해석.")


if __name__ == "__main__":
    main()
