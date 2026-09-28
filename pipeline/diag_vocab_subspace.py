"""
E1 진단 (09-28): 양자화 임베딩 오차가 텍스트 부분공간에 얼마나 실려 있는가.

vocab-metric 재구성(quant/vocab_metric.py)은 BRECQ의 재구성 손실을 "임의 vocabulary에서의
기대 유사도 오차" tau^2 * v^T C v 로 바꾼다(v = normalize(x_q) - normalize(x_fp), x = cv4 입력).
돌리기 전에 싼 비용으로 보는 것:

  err_iso    = E_a[ w_a ||v_a||^2 ]              (방향 오차 총량, 항등 metric)
  err_metric = E_a[ w_a v_a^T C v_a ]            (C는 평균 고유값 1로 정규화 -> 등방이면 err_iso와 같음)
  rho        = err_metric / err_iso              (1 = 오차가 텍스트 방향과 무관, >1 = 텍스트 방향에 몰림,
                                                  <1 = 대부분 텍스트가 거의 안 쓰는 방향)
  frac@k     = C의 상위 k 고유방향 부분공간에 실린 오차 에너지 비율 (무작위 방향이면 k/D)
  C 유효 차원 = (tr C)^2 / tr(C^2)

읽는 법: C 유효 차원이 D(512)보다 훨씬 작고(= metric이 소수 방향에 집중) BRECQ의 rho가 1 근처
이하라면, BRECQ가 유사도에 거의 영향 없는 방향의 오차를 줄이는 데 용량을 쓰고 있다는 뜻 --
metric을 바꿔 그 용량을 텍스트 방향으로 옮길 여지가 있다. 반대로 rho가 이미 크고 frac@k가
높으면(오차가 원래 텍스트 방향에 몰려 있으면) 재가중만으로 얻을 이득은 작다.

실행 예 (head 포함 W4A8, 첫/마지막 8bit):
    python pipeline/diag_vocab_subspace.py --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \\
        --w-bits 4 --a-bits 8 --no-skip-head --first-last-bits 8 --modes naive,brecq \\
        --vm-vocab configs/vocab_generic.txt --calib 256 --n-eval 200 --device 0 --deterministic
"""
import argparse, glob, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_comparison as rc                                   # build/preprocess/load_names 재사용
from quant.pdquant import _find_head, _CV4Capture
from quant.vocab_metric import VocabMetric, load_vocab, encode_text_bank


@torch.no_grad()
def cv4_inputs(model, t, device):
    head = _find_head(model.model)
    cap = _CV4Capture(head, capture_input=True)
    model.model(t.to(device))
    xs = [cap.inbuf[i].detach() for i in sorted(cap.inbuf)]
    cap.close()
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--skip-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--first-last-bits", type=int, default=8)
    ap.add_argument("--modes", default="naive,brecq")
    ap.add_argument("--vm-vocab", default="configs/vocab_generic.txt")
    ap.add_argument("--vm-lam-mean", type=float, default=1.0)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--recon-iters", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=200, help="val2017 앞에서부터 사용할 이미지 수")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    print(f"[args] {vars(args)}")
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    from ultralytics import YOLOWorld
    coco, lvis_names = rc.load_names("coco"), rc.load_names("lvis")
    calib_paths = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))[:args.calib]
    eval_paths = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))[:args.n_eval]
    calib = [rc.preprocess(p, args.imgsz, device) for p in calib_paths]

    names, identity = load_vocab(args.vm_vocab, coco, lvis_names)
    assert not identity, "진단은 비항등 metric이 필요합니다(--vm-vocab identity 불가)"
    bank = encode_text_bank(YOLOWorld, args.model, names, device)
    vm = VocabMetric(bank, lam_mean=args.vm_lam_mean)
    D = vm.C.shape[0]
    evals, evecs = torch.linalg.eigh(vm.C)
    order = evals.argsort(descending=True)
    evecs = evecs[:, order]
    ks = [k for k in (1, 8, 16, 32, 64, 128) if k <= D]
    print(f"[vm] vocab={args.vm_vocab} ({len(names)}개) C 유효 차원 {vm.effective_rank():.1f}/{D}, "
          f"상위 고유값 {[round(float(e), 2) for e in evals[order][:5]]}")

    fp = rc.build(YOLOWorld, args.model, coco, device, calib, "fp")
    vm.bind_head(_find_head(fp.model))

    # FP 쪽 cv4 입력과 anchor 가중치는 1회만 계산
    fp_x, fp_w = [], []
    for p in eval_paths:
        xs = cv4_inputs(fp, rc.preprocess(p, args.imgsz, "cpu"), device)
        fp_x.append([x.cpu() for x in xs])
        fp_w.append([vm.anchor_weights(x, i).cpu() for i, x in enumerate(xs)])

    rows = []
    for mode in [m.strip() for m in args.modes.split(",")]:
        q = rc.build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
                     w_bits=args.w_bits, a_bits=args.a_bits, skip_head=args.skip_head,
                     first_last_bits=args.first_last_bits, recon_iters_strong=args.recon_iters,
                     brecq_two_stage=False, brecq_batch=1, neck_layerwise=False)
        acc = dict(w=0.0, iso=0.0, met=0.0, frac={k: 0.0 for k in ks}, mu=0.0)
        mu = F.normalize(vm.bank.mean(0), dim=0)
        for i, p in enumerate(eval_paths):
            xs_q = cv4_inputs(q, rc.preprocess(p, args.imgsz, "cpu"), device)
            for lvl, xq in enumerate(xs_q):
                f = F.normalize(VocabMetric.flat(fp_x[i][lvl].to(device)).float(), dim=-1)
                g = F.normalize(VocabMetric.flat(xq).float(), dim=-1)
                v = g - f
                w = fp_w[i][lvl].to(device)
                e2 = v.pow(2).sum(-1)
                acc["w"] += float(w.sum())
                acc["iso"] += float((w * e2).sum())
                acc["met"] += float((w * ((v @ vm.C) * v).sum(-1)).sum())
                acc["mu"] += float((w * (v @ mu).pow(2)).sum())
                proj = v @ evecs                               # 고유기저 좌표
                for k in ks:
                    acc["frac"][k] += float((w * proj[:, :k].pow(2).sum(-1)).sum())
        iso = acc["iso"] / acc["w"]
        met = acc["met"] / acc["w"]
        row = dict(mode=mode, iso=iso, met=met, rho=met / max(iso, 1e-12),
                   mu=acc["mu"] / max(acc["iso"], 1e-12),
                   frac={k: acc["frac"][k] / max(acc["iso"], 1e-12) for k in ks})
        rows.append(row)
        del q
        rc.free_cpu_mem()
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print(f" 임베딩 방향 오차의 텍스트 부분공간 분해 -- W{args.w_bits}A{args.a_bits}, "
          f"head {'제외' if args.skip_head else '포함'}, 첫/마지막 {args.first_last_bits or '-'}bit, "
          f"eval {len(eval_paths)}장")
    print("=" * 100)
    head = f"{'':>8} | {'err_iso':>10} | {'err_metric':>10} | {'rho':>6} | {'mu-frac':>7} | " + \
        " | ".join(f"frac@{k:<3}" for k in ks)
    print(head)
    print(f"{'random':>8} | {'':>10} | {'':>10} | {1.0:>6.3f} | {1 / D:>7.4f} | " +
          " | ".join(f"{k / D:>8.4f}" for k in ks))
    for r in rows:
        print(f"{r['mode']:>8} | {r['iso']:>10.4e} | {r['met']:>10.4e} | {r['rho']:>6.3f} | {r['mu']:>7.4f} | " +
              " | ".join(f"{r['frac'][k]:>8.4f}" for k in ks))


if __name__ == "__main__":
    main()
