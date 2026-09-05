"""
Run-to-run 분산 직접 측정 (09-05 진단 후속, 36번 다음 단계).

36번에서 seed=2, lr=0.01로 완전히 동일한 설정인데 35번(S_margin=0.1415,
H_eval=9.58%, 악화)과 36번(S_margin=0.1186, H_eval=8.64%, 개선)이 정반대로
나왔다. 코드 차이는 eval_hook 추가뿐이고 이건 @torch.no_grad()라 학습에
영향을 줄 수 없다. 이 스크립트는 그 의심 -- torch/cuDNN non-determinism
때문에 "동일한" 설정도 실행마다 결과가 크게 요동친다 -- 을 직접 검증한다.

방법: AdaRound weight rounding을 1회만 최적화해서 고정하고(state 동결),
그 지점에서 optimize_promptcal_scale(seed=2, S-only, lr=0.01, iters=1500)을
torch.manual_seed 없이 N회 반복 실행한다. 매 trial은 AdaRound 체크포인트의
독립적인 deepcopy에서 시작하므로 초기 조건은 완전히 동일하고, 차이가 있다면
전적으로 학습 중 GPU 연산의 비결정성에서 온다.

실행:
    CUDA_VISIBLE_DEVICES=1 python scripts/37_variance_check.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seed 2 --trials 6 --device 0
"""
import argparse, copy, glob, os, sys
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
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-seed", type=int, default=0,
                    help="torch.manual_seed 고정값. 37번에서 확인된 run-to-run "
                         "non-determinism(같은 설정도 실행마다 결과가 다름)을 없애기 위함.")
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[determinism] torch.manual_seed={args.torch_seed}, cudnn.deterministic=True, cudnn.benchmark=False")

    names = load_coco_names()

    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib = [preprocess(p, args.imgsz, device) for p in imgs[:args.calib]]
    probe = [preprocess(p, args.imgsz, device) for p in imgs[args.calib:args.calib + args.eval]]

    print("[load] FP")
    fp = YOLOWorld(args.model); fp.set_classes(names); fp.fuse(); fp.model.to(device).eval()
    h_fp = SimilarityHarness(fp.model, device=device)

    print("[build] AdaRound (weight rounding, 1회만 최적화, 이후 모든 trial의 공통 시작점)")
    ad = YOLOWorld(args.model); ad.set_classes(names); ad.fuse()
    wrap_convs(ad.model, 8, 8); ad.model.to(device).eval()
    calibrate(ad.model, calib, device=device)
    convert_to_adaround(ad.model)
    optimize_adaround(ad.model, fp.model, calib, device, iters=1000, verbose=False)

    h_ad = SimilarityHarness(ad.model, device=device)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
    pidx = S

    ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
    ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)
    print(f"\n[seed {args.seed} 기준선] AdaRound  S_margin={ad_s_margin:.4f}  H_eval flip={ad_heval_flip:.2f}%")
    h_ad.close()

    print("\n" + "=" * 100)
    print(f" 동일 설정(seed={args.seed}, lr={args.lr}) {args.trials}회 반복 -- torch.manual_seed 없음")
    print("=" * 100)

    results = []
    for trial in range(args.trials):
        print(f"\n--- trial {trial} ---")
        cb_model = copy.deepcopy(ad.model)  # AdaRound 체크포인트의 독립적 복사본에서 시작
        optimize_promptcal_scale(cb_model, fp.model, calib, device, pidx,
                                 iters=args.iters, lr=args.lr, k=args.k, verbose=False)
        h_cb = SimilarityHarness(cb_model, device=device)
        s_margin = group_margin(h_fp, h_cb, calib, S, k=args.k)
        heval_flip, _ = group_flip(h_fp, h_cb, probe, H_eval)
        h_cb.close()
        print(f"  S_margin={s_margin:.4f}  H_eval flip={heval_flip:.2f}%  "
              f"({'개선' if heval_flip < ad_heval_flip else '악화'} vs AdaRound)")
        results.append((s_margin, heval_flip))

    h_fp.close()

    margins = np.array([r[0] for r in results])
    hevals = np.array([r[1] for r in results])

    print("\n" + "=" * 100)
    print(" 요약")
    print("=" * 100)
    print(f"AdaRound 기준선: S_margin={ad_s_margin:.4f}  H_eval={ad_heval_flip:.2f}%")
    print(f"{args.trials}회 반복 (동일 설정, seed={args.seed}, lr={args.lr}):")
    print(f"  S_margin : mean={margins.mean():.4f}  std={margins.std():.4f}  "
          f"min={margins.min():.4f}  max={margins.max():.4f}")
    print(f"  H_eval   : mean={hevals.mean():.2f}%  std={hevals.std():.2f}  "
          f"min={hevals.min():.2f}%  max={hevals.max():.2f}%")
    n_better = int((hevals < ad_heval_flip).sum())
    print(f"  AdaRound보다 H_eval 개선된 trial: {n_better}/{args.trials}")

    print("\n판정:")
    print("  std가 AdaRound와의 평균 격차보다 크거나 비슷하면 -> '개선/악화'를 1회 실행으로")
    print("     판단하는 것 자체가 통계적으로 무의미. 지금까지 29~35의 개별 seed 결론들도")
    print("     같은 잣대로 재해석해야 함(여러 번 반복해서 평균+분산으로 봐야 함).")
    print("  std가 충분히 작고 mean이 뚜렷하게 AdaRound보다 낫거나 나쁘면 -> non-determinism은")
    print("     결과를 뒤집을 정도는 아니고, 원래 결론(방향 C 유망/불안정)을 유지해도 됨.")


if __name__ == "__main__":
    main()
