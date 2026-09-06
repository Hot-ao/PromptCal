"""
Neighbor preservation 단독 검증 (09-05 §40 진단 이후, 논문 §4.2 Prompt Selection의
빠진 조각을 채우는 항).

39번(결정성 적용)에서 AdaRound+scale은 seed 0/1에서 지고 seed 2에서만 이겼다.
40번 분석: H_eval이 S와 text embedding상 가까운 seed(0,1)일수록 악화폭이 컸다 --
S의 margin을 맞추려는 s_mult 조정이 class-agnostic해서, S와 가까운 class로
collateral shift가 전파되는 것으로 해석됨.

이 스크립트는 optimize_promptcal_scale_neighbor(src/quant/promptcal.py)를
단독으로 검증한다: 기존 S margin objective에 "S 각 class의 text-embedding
최근접 이웃(비-S)의 similarity를 FP와 가깝게 유지" 항 하나만 추가한 것.
AdaRound weight는 동일(35/39와 같은 방식으로 재구축), 결정성 적용(torch_seed
고정), GPU 7 고정.

비교:
  AdaRound            (baseline)
  AdaRound+scale      (39번 결과 참고치, 이 스크립트에서도 재확인차 같이 계산)
  AdaRound+scale+near (신규, neighbor_weight로 강도 조절)

실행:
    CUDA_VISIBLE_DEVICES=7 python scripts/41_neighbor_preserve.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seeds 0 1 2 --neighbor-weight 1.0 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.promptcal import optimize_promptcal_scale, optimize_promptcal_scale_neighbor, margin_loss


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


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0, asymmetric=False):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "scale":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_promptcal_scale(m.model, fp.model, calib, device, pidx, iters=iters,
                                 lr=lr, k=k, verbose=True)
    elif mode == "scale_neighbor":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=asymmetric, verbose=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--asymmetric", action="store_true",
                    help="neighbor loss를 one-sided hinge(경쟁자가 FP보다 강해지는 "
                         "방향만 억제)로 사용. 기본은 symmetric MSE.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--torch-seed", type=int, default=0)
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
    print("[build] AdaRound (baseline, seed-무관)")
    ad = build(YOLOWorld, args.model, names, device, calib, "adaround", fp=fp)

    h_fp = SimilarityHarness(fp.model, device=device)
    h_ad = SimilarityHarness(ad.model, device=device)

    print("\n" + "=" * 100)
    print(f" AdaRound vs +scale vs +scale+neighbor(k={args.neighbor_k}, w={args.neighbor_weight}, "
          f"asymmetric={args.asymmetric})")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_eval = perm[60:80].tolist()
        pidx = S

        print(f"\n--- seed {s} ---")

        ad_s_margin = group_margin(h_fp, h_ad, calib, S, k=args.k)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)

        sc = build(YOLOWorld, args.model, names, device, calib, "scale", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, k=args.k)
        h_sc = SimilarityHarness(sc.model, device=device)
        sc_s_margin = group_margin(h_fp, h_sc, calib, S, k=args.k)
        sc_heval_flip, _ = group_flip(h_fp, h_sc, probe, H_eval)
        h_sc.close()

        sn = build(YOLOWorld, args.model, names, device, calib, "scale_neighbor", fp=fp,
                  iters=args.iters, pidx=pidx, lr=args.lr, k=args.k,
                  neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                  asymmetric=args.asymmetric)
        h_sn = SimilarityHarness(sn.model, device=device)
        sn_s_margin = group_margin(h_fp, h_sn, calib, S, k=args.k)
        sn_heval_flip, _ = group_flip(h_fp, h_sn, probe, H_eval)
        h_sn.close()

        print(f"\n  [S margin]      AdaRound={ad_s_margin:.4f}  +scale={sc_s_margin:.4f}  +scale+near={sn_s_margin:.4f}")
        print(f"  [H_eval flip]   AdaRound={ad_heval_flip:.2f}%  +scale={sc_heval_flip:.2f}%  +scale+near={sn_heval_flip:.2f}%")

        rows.append(dict(seed=s, ad_m=ad_s_margin, sc_m=sc_s_margin, sn_m=sn_s_margin,
                         ad_h=ad_heval_flip, sc_h=sc_heval_flip, sn_h=sn_heval_flip))

    h_fp.close(); h_ad.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'Ada_H':>7} | {'Scale_H':>8} | {'ScaleNear_H':>11} | "
          f"{'Ada_m':>7} | {'Scale_m':>8} | {'ScaleNear_m':>11}")
    for r in rows:
        print(f"{r['seed']:>4} | {r['ad_h']:>6.2f}% | {r['sc_h']:>7.2f}% | {r['sn_h']:>10.2f}% | "
              f"{r['ad_m']:>7.4f} | {r['sc_m']:>8.4f} | {r['sn_m']:>11.4f}")

    print("\n참고: 39번(결정성 적용, neighbor 없음) H_eval = 7.99%/9.09%/9.24% (seed 0/1/2)")

    print("\n판정:")
    print("  ScaleNear_H < Scale_H (동일 seed에서 neighbor 항이 순수하게 도움이 됐는지)")
    print("     -> collateral shift 억제가 실제로 작동. seed 0/1의 악화가 줄어드는지가 핵심.")
    print("  ScaleNear_H < Ada_H 까지 되면 -> AdaRound baseline 자체를 이긴 것(논문 핵심 주장 지지).")
    print("  변화 없거나 악화 -> neighbor 항의 강도(neighbor_weight)나 범위(neighbor_k) 조정 필요,")
    print("     혹은 이 메커니즘 자체가 억제 대상이 아닐 수 있음.")


if __name__ == "__main__":
    main()
