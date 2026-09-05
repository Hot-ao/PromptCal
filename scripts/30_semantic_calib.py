"""
Semantic Calibration + Utility-Constrained Refinement 평가.

v1(20/29_*)과 동일한 S/H_cal/H_eval 3분할, 동일 heldout-flip(group_flip) 공식을 쓰되,
최적화 방식만 src/quant/semantic_calib.py의 새 방법(동적 semantic neighbor 경쟁집합 +
local reconstruction + utility constraint)으로 바꾼다.

비교: naive / AdaRound(baseline) / Ours(semantic_pcal).
측정: H_eval flip(헤드라인) + H_cal flip/margin(soft/discrete, 29번과 같은 진단표).

실행:
    CUDA_VISIBLE_DEVICES=3 python scripts/30_semantic_calib.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 1000 --iters 1500 --seeds 0 1 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.semantic_calib import optimize_semantic_pcal
from src.quant.promptcal import margin_loss  # 기존 margin 공식 재사용(H_cal 진단표 비교용)


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
          lr=3e-3, reg_weight=0.1, seed=0, post_train_hook=None, **kw):
    m = model_cls(w)
    m.set_classes(names)
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=1000, verbose=False)
    elif mode == "ours":
        convert_to_adaround(m.model)
        optimize_semantic_pcal(m.model, fp.model, calib, device, pidx, iters=iters,
                               lr=lr, reg_weight=reg_weight, seed=seed, verbose=True,
                               post_train_hook=post_train_hook, **kw)
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
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--k-near", type=int, default=5)
    ap.add_argument("--k-far", type=int, default=3)
    ap.add_argument("--margin-thres", type=float, default=0.5)
    ap.add_argument("--recon-w", type=float, default=1.0)
    ap.add_argument("--stage2-frac", type=float, default=0.3)
    ap.add_argument("--thresh-w", type=float, default=1.0)
    ap.add_argument("--box-w", type=float, default=0.5)
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
    print(" Semantic Calibration + Utility-Constrained Refinement 평가")
    print("=" * 100)

    rows = []
    for s in args.seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(80)
        S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
        pidx = S + H_cal

        print(f"\n--- seed {s} ---")

        ad_hcal_flip, ad_hcal_n = group_flip(h_fp, h_ad, calib, H_cal)
        ad_hcal_margin = group_margin(h_fp, h_ad, calib, pidx)
        ad_heval_flip, _ = group_flip(h_fp, h_ad, probe, H_eval)
        nv_heval_flip, _ = group_flip(h_fp, h_nv, probe, H_eval)

        soft_capture = {}

        def snapshot_soft(qmodel):
            h_soft = SimilarityHarness(qmodel, device=device)
            f, n = group_flip(h_fp, h_soft, calib, H_cal)
            m = group_margin(h_fp, h_soft, calib, pidx)
            soft_capture["flip"] = f; soft_capture["flip_n"] = n; soft_capture["margin"] = m
            h_soft.close()

        ours = build(YOLOWorld, args.model, names, device, calib, "ours", fp=fp,
                    iters=args.iters, pidx=pidx, lr=args.lr, reg_weight=args.reg_weight,
                    seed=s, post_train_hook=snapshot_soft,
                    k_near=args.k_near, k_far=args.k_far, margin_thres=args.margin_thres,
                    recon_w=args.recon_w, stage2_frac=args.stage2_frac,
                    thresh_w=args.thresh_w, box_w=args.box_w)

        h_ours = SimilarityHarness(ours.model, device=device)
        ours_hcal_flip_disc, ours_hcal_n = group_flip(h_fp, h_ours, calib, H_cal)
        ours_hcal_margin_disc = group_margin(h_fp, h_ours, calib, pidx)
        ours_heval_flip, _ = group_flip(h_fp, h_ours, probe, H_eval)
        h_ours.close()

        print(f"\n  [H_cal, n_anchor≈{ad_hcal_n}]")
        print(f"    AdaRound(무관)     flip={ad_hcal_flip:6.2f}%  margin={ad_hcal_margin:.4f}")
        print(f"    Ours soft          flip={soft_capture['flip']:6.2f}%  margin={soft_capture['margin']:.4f}  (n={soft_capture['flip_n']})")
        print(f"    Ours discrete      flip={ours_hcal_flip_disc:6.2f}%  margin={ours_hcal_margin_disc:.4f}  (n={ours_hcal_n})")
        print(f"  [H_eval, 최종 헤드라인]")
        print(f"    naive     flip={nv_heval_flip:6.2f}%")
        print(f"    AdaRound  flip={ad_heval_flip:6.2f}%")
        print(f"    Ours      flip={ours_heval_flip:6.2f}%")

        rows.append(dict(seed=s, ad_hcal=ad_hcal_flip, soft_hcal=soft_capture["flip"],
                         disc_hcal=ours_hcal_flip_disc, nv_heval=nv_heval_flip,
                         ad_heval=ad_heval_flip, ours_heval=ours_heval_flip))

    h_fp.close(); h_nv.close(); h_ad.close()

    print("\n" + "=" * 100)
    print(" 요약 표")
    print("=" * 100)
    print(f"{'seed':>4} | {'Ada_Hcal':>9} | {'Ours_soft_Hcal':>14} | {'Ours_disc_Hcal':>14} | "
          f"{'nv_Heval':>9} | {'Ada_Heval':>9} | {'Ours_Heval':>10} | {'vs_Ada':>8}")
    for r in rows:
        vs = (r['ad_heval'] - r['ours_heval']) / max(r['ad_heval'], 1e-9) * 100
        print(f"{r['seed']:>4} | {r['ad_hcal']:>8.2f}% | {r['soft_hcal']:>13.2f}% | "
              f"{r['disc_hcal']:>13.2f}% | {r['nv_heval']:>8.2f}% | {r['ad_heval']:>8.2f}% | "
              f"{r['ours_heval']:>9.2f}% | {vs:>7.1f}%")
    m_ad = np.mean([r['ad_heval'] for r in rows])
    m_ours = np.mean([r['ours_heval'] for r in rows])
    print(f"\n mean AdaRound H_eval={m_ad:.2f}%  Ours H_eval={m_ours:.2f}%  "
          f"({(m_ad - m_ours) / max(m_ad, 1e-9) * 100:+.1f}% vs AdaRound)")


if __name__ == "__main__":
    main()
