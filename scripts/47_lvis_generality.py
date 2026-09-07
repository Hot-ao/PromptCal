"""
국면 I: Combined(및 다른 baseline)가 COCO-80이 아닌 LVIS-1203 vocabulary로
배포됐을 때도 이점이 유지되는가 (09-07 착수, MASTER_SUMMARY.md §8 국면 I).

09_lvis_compare.py(구 07번)와 같은 방식: 실제 LVIS 이미지/annotation은 안 쓰고,
COCO val2017 이미지에 vocabulary만 LVIS-1203으로 바꿔서 붙인다(ultralytics
lvis.yaml의 class name 리스트 재사용). GT가 없으므로 AP는 측정 불가 -- 대신
C-3와 동일하게 FP32confident 예측을 pseudo-label로 삼아 flip rate / mean
margin / semantic-neighbor bias를 잰다.

핵심 질문: naive/AdaRound/QDrop/BRECQ/Combined 5개 모두 COCO-80 calibration
으로 학습된 채로 고정하고, 추론 시점에만 vocabulary를 LVIS-1203으로 바꿔
꽂았을 때 -- COCO-80에서 봤던 "Combined가 표준 flip에서 최선급"이라는 순위가
그대로 유지되는가, 아니면 완전히 낯선(calibration과 무관한) vocabulary에서는
무너지는가.

실행:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python scripts/47_lvis_generality.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --calib 32 --eval 500 --seed 2 --device 0
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness import SimilarityHarness
from src.quant.quant_model import wrap_convs, calibrate
from src.quant.adaround import convert_to_adaround, optimize_adaround
from src.quant.brecq import optimize_brecq
from src.quant.promptcal import optimize_promptcal_scale_neighbor


def load_names(which):
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / f"{which}.yaml"
    d = yaml.safe_load(open(p))
    n = d["names"]
    return [n[i] for i in range(len(n))]


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


def switch_vocab(model, names, device):
    """vocabulary 교체. CPU로 내려 CLIP을 새로 빌드해 txt_feats를 계산한 뒤
    복귀 -- 양자화 buffer(alpha/s_mult 등)는 vision 경로에만 있어 이동에도 보존됨."""
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.to(device).eval()


def neighbors_from_txt(txt, k):
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = txt @ txt.T
    idx = sim.argsort(dim=-1, descending=True)
    neigh = []
    for c in range(sim.shape[0]):
        row = idx[c].tolist()
        neigh.append(set([o for o in row if o != c][:k]))
    return neigh


def measure_vocab(names, model_fp, model_q, probe_imgs, k, conf_thres, imgsz, device):
    """C-3와 동일 방식: FP confident 예측을 pseudo-GT로, flip rate/margin/
    semantic-neighbor bias를 잰다. 실제 GT 불필요(LVIS 이미지가 아니므로)."""
    switch_vocab(model_fp, names, device)
    switch_vocab(model_q, names, device)
    P = len(names)
    txt = model_fp.model.txt_feats.detach().float()
    if txt.dim() == 3:
        txt = txt[0]
    neigh = neighbors_from_txt(txt, k)

    h_fp = SimilarityHarness(model_fp.model, device=device)
    h_q = SimilarityHarness(model_q.model, device=device)

    n_conf = n_flip = flip_to_neigh = 0
    margin_sum = 0.0
    small_margin = 0
    for i, t in enumerate(probe_imgs):
        sim_fp = h_fp.run_image(t, i).sim
        sim_q = h_q.run_image(t, i).sim
        prob = sim_fp.sigmoid()
        maxp, c_fp = prob.max(-1)
        conf = maxp > conf_thres
        if conf.sum() == 0:
            continue
        c_q = sim_q.argmax(-1)
        top2 = sim_fp.topk(2, -1).values
        margin = top2[:, 0] - top2[:, 1]

        idxc = conf.nonzero(as_tuple=True)[0]
        n_conf += int(conf.sum())
        margin_sum += float(margin[idxc].sum())
        small_margin += int((margin[idxc] < 0.5).sum())
        for j in idxc.tolist():
            a = int(c_fp[j]); b = int(c_q[j])
            if b != a:
                n_flip += 1
                if b in neigh[a]:
                    flip_to_neigh += 1

    h_fp.close(); h_q.close()
    chance = k / (P - 1) * 100
    flip_rate = n_flip / max(n_conf, 1) * 100
    to_neigh = flip_to_neigh / max(n_flip, 1) * 100
    bias = (flip_to_neigh / max(n_flip, 1)) / (k / (P - 1)) if n_flip else 0.0
    mean_margin = margin_sum / max(n_conf, 1)
    small_pct = small_margin / max(n_conf, 1) * 100
    return dict(P=P, n_conf=n_conf, flip_rate=flip_rate, to_neigh=to_neigh,
               chance=chance, bias=bias, mean_margin=mean_margin, small_pct=small_pct)


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0,
          recon_iters_ada=1000, recon_iters_strong=2000, qdrop_prob=0.5):
    m = model_cls(w)
    m.set_classes(names)
    if mode == "fp":
        m.fuse(); m.model.to(device).eval()
        return m
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "naive":
        pass
    elif mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
    elif mode == "qdrop":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_strong,
                          qdrop_prob=qdrop_prob, verbose=False)
    elif mode == "brecq":
        convert_to_adaround(m.model)
        optimize_brecq(m.model, fp.model, calib, device, iters=recon_iters_strong, verbose=False)
    elif mode == "combined":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=True, verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--recon-iters-strong", type=int, default=2000)
    ap.add_argument("--qdrop-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2)
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

    coco = load_names("coco")
    lvis = load_names("lvis")
    from ultralytics import YOLOWorld
    imgs = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    calib_paths = imgs[:args.calib]
    probe_paths = imgs[args.calib:args.calib + args.eval]
    calib = [preprocess(p, args.imgsz, device) for p in calib_paths]
    probe = [preprocess(p, args.imgsz, device) for p in probe_paths]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist()

    print("[build] FP (COCO-80 vocab)"); fp = build(YOLOWorld, args.model, coco, device, calib, "fp")
    conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]
    models = {}
    for mode in conditions:
        print(f"[build] {mode} (COCO-80 calibration/training)")
        models[mode] = build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
                             iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                             neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                             recon_iters_ada=args.recon_iters_ada,
                             recon_iters_strong=args.recon_iters_strong,
                             qdrop_prob=args.qdrop_prob)

    results = {}
    for mode in conditions:
        print(f"\n[measure] {mode} -- COCO-80 vocab (sanity) ...")
        r_coco = measure_vocab(coco, fp, models[mode], probe, args.k, args.conf_thres, args.imgsz, device)
        print(f"[measure] {mode} -- LVIS-1203 vocab ...")
        r_lvis = measure_vocab(lvis, fp, models[mode], probe, args.k, args.conf_thres, args.imgsz, device)
        results[mode] = (r_coco, r_lvis)
        print(f"  {mode:>10}: COCO flip={r_coco['flip_rate']:.2f}% margin={r_coco['mean_margin']:.2f}  |  "
              f"LVIS flip={r_lvis['flip_rate']:.2f}% margin={r_lvis['mean_margin']:.2f} bias={r_lvis['bias']:.1f}x")

    print("\n" + "=" * 100)
    print(f" COCO-80 vocab (sanity, calibration과 같은 vocabulary) -- seed {args.seed}")
    print("=" * 100)
    print(f"{'':>10} | {'flip%':>7} | {'margin':>7} | {'small%':>7} | {'→이웃%':>7} | {'편향':>6}")
    for mode in conditions:
        r, _ = results[mode]
        print(f"{mode:>10} | {r['flip_rate']:>6.2f}% | {r['mean_margin']:>7.2f} | "
              f"{r['small_pct']:>6.1f}% | {r['to_neigh']:>6.1f}% | {r['bias']:>5.1f}x")

    print("\n" + "=" * 100)
    print(f" LVIS-1203 vocab (calibration에 전혀 없던 vocabulary) -- seed {args.seed}")
    print("=" * 100)
    print(f"{'':>10} | {'flip%':>7} | {'margin':>7} | {'small%':>7} | {'→이웃%':>7} | {'편향':>6}")
    for mode in conditions:
        _, r = results[mode]
        print(f"{mode:>10} | {r['flip_rate']:>6.2f}% | {r['mean_margin']:>7.2f} | "
              f"{r['small_pct']:>6.1f}% | {r['to_neigh']:>6.1f}% | {r['bias']:>5.1f}x")

    print("\n판정:")
    print("  COCO-80에서의 순위(Combined가 flip 최저권)가 LVIS-1203에서도 유지되면")
    print("     -> calibration vocabulary와 무관한 진짜 일반화.")
    print("  LVIS에서 Combined가 naive/AdaRound보다 나빠지면 -> collateral shift가")
    print("     calibration vocabulary 밖에서는 억제되지 않는다는 뜻 -> 한계로 서술.")


if __name__ == "__main__":
    main()
