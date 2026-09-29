"""
0단계 진단 (09-29): W4 weight 손상이 어느 블록에 몰려 있는가.

brecq_vm(손실 교체)이 W4A8에서 전부 실패해서(runs/123~126) 양자화기 쪽 수정(채널별 scale 이전 /
민감도 기반 혼합 정밀도)으로 방향을 바꾼다. 어느 쪽이 본체가 될지는 "손상이 소수 블록에 몰려 있나,
전체에 퍼져 있나"에 달려 있으므로 학습 없이(naive rounding) 두 방향으로 잰다.

  drop    : 전부 W8A8에서 그룹 하나만 W4로 내림     -> 그 그룹이 혼자 만드는 손상
  restore : W4A8(첫/마지막 8bit)에서 그룹 하나만 W8로 올림 -> 그 그룹을 지켰을 때 되찾는 양

그룹 = model.model[i] (backbone/neck 모듈 하나) + head cv2/cv3 레벨별. 첫/마지막 레이어(stem 첫 conv,
cv2/cv3 마지막 1x1)는 저비트 프로토콜상 항상 8bit라 어느 그룹에서도 바꾸지 않는다.
activation 범위(observer)는 기준 모델(drop=W8A8, restore=W4A8)에서 1회 calibrate한 값을 고정한다.

지표 (val2017 앞 n_eval장, FP 대비):
  emb_err   : cv4 입력(region 임베딩) 방향 오차 ||x̂_q - x̂_fp||^2, FP LVIS 확신도 가중 평균
  LVIS_flip : FP LVIS-1203 확신도 > conf 인 anchor에서 top-1 LVIS 클래스가 바뀐 비율(%)
  COCO_flip : 같은 방식, COCO-80
  box_err   : cv2 출력(박스 분포 logit) 상대 L2 오차 ||q - fp||^2 / ||fp||^2
  Mparam    : 그룹의 양자화 weight 원소 수(백만) -- 혼합 정밀도 비용 계산용

실행 예:
    python pipeline/diag_w4_sensitivity.py --n-eval 200 --device 4 --deterministic
"""
import argparse, glob, os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_comparison as rc
from quant.fake_quant import QuantConv2d
from quant.pdquant import _find_head, _CV4Capture
from quant.vocab_metric import VocabMetric, encode_text_bank


def first_last_ids(det_model):
    """set_first_last_bits와 같은 규칙으로 항상 8bit인 conv의 id 집합."""
    ids = set()
    for mod in det_model.model[0].modules():
        if isinstance(mod, QuantConv2d):
            ids.add(id(mod))
            break
    head = det_model.model[-1]
    for branch in ("cv2", "cv3"):
        for lvl in getattr(head, branch, []):
            if isinstance(lvl[-1], QuantConv2d):
                ids.add(id(lvl[-1]))
    return ids


def groups_of(det_model, expand=()):
    """[(이름, [QuantConv2d...])] -- 첫/마지막 레이어 제외, 빈 그룹 제외.
    expand: 이 인덱스의 모듈은 블록 대신 conv 하나하나를 그룹으로 쪼갠다."""
    fixed = first_last_ids(det_model)
    seq = det_model.model
    out = []
    for i, blk in enumerate(seq[:-1]):
        if i in expand:
            for n, m in blk.named_modules():
                if isinstance(m, QuantConv2d) and id(m) not in fixed:
                    out.append((f"{i}.{n}", [m]))
            continue
        convs = [m for m in blk.modules() if isinstance(m, QuantConv2d) and id(m) not in fixed]
        if convs:
            out.append((f"{i}:{type(blk).__name__}", convs))
    head = seq[-1]
    for branch in ("cv2", "cv3"):
        for li, lvl in enumerate(getattr(head, branch, [])):
            convs = [m for m in lvl.modules() if isinstance(m, QuantConv2d) and id(m) not in fixed]
            if convs:
                out.append((f"head.{branch}.{li}", convs))
    return out


def set_wbits(convs, bits):
    for c in convs:
        c.w_bits = bits
        c.freeze_weight_quant()


class HeadTap:
    """cv4 입력(region 임베딩)과 cv2 출력(박스 logit)을 레벨별로 잡는다."""
    def __init__(self, head):
        self.cv4 = _CV4Capture(head, capture_input=True)
        self.box = {}
        self.handles = [lvl.register_forward_hook(self._mk(i)) for i, lvl in enumerate(head.cv2)]

    def _mk(self, i):
        def hook(_m, _inp, out):
            self.box[i] = out.detach()
        return hook

    def emb(self):
        return [self.cv4.inbuf[i].detach() for i in sorted(self.cv4.inbuf)]

    def boxes(self):
        return [self.box[i] for i in sorted(self.box)]

    def close(self):
        self.cv4.close()
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_tap(model, t, device):
    tap = HeadTap(_find_head(model.model))
    model.model(t.to(device))
    e, b = tap.emb(), tap.boxes()
    tap.close()
    return e, b


def sims(xn, vm, lvl):
    tau, b = vm.levels[lvl]
    return tau * (xn @ vm.bank.T) + b


@torch.no_grad()
def reference(fp, evals, device, vm_lvis, vm_coco, conf):
    """FP 쪽 기준값을 이미지별로 1회 계산해 CPU에 보관."""
    refs = []
    for t in evals:
        e, b = run_tap(fp, t, device)
        per = []
        for lvl, x in enumerate(e):
            xn = F.normalize(VocabMetric.flat(x).float(), dim=-1)
            sl, sc = sims(xn, vm_lvis, lvl), sims(xn, vm_coco, lvl)
            pl, cl = sl.sigmoid().max(-1)
            pc, cc = sc.sigmoid().max(-1)
            per.append(dict(xn=xn.half().cpu(), w=(pl + 0.05).cpu(),
                            lvis_m=(pl > conf).cpu(), lvis_c=cl.cpu(),
                            coco_m=(pc > conf).cpu(), coco_c=cc.cpu(),
                            box=b[lvl].float().cpu()))
        refs.append(per)
    return refs


@torch.no_grad()
def measure(q, evals, refs, device, vm_lvis, vm_coco):
    a = dict(w=0.0, emb=0.0, lf=0, ln=0, cf=0, cn=0, bnum=0.0, bden=0.0)
    for t, per in zip(evals, refs):
        e, b = run_tap(q, t, device)
        for lvl, x in enumerate(e):
            r = per[lvl]
            xn = F.normalize(VocabMetric.flat(x).float(), dim=-1)
            f = r["xn"].to(device).float()
            w = r["w"].to(device)
            a["w"] += float(w.sum())
            a["emb"] += float((w * (xn - f).pow(2).sum(-1)).sum())
            lm, cm = r["lvis_m"].to(device), r["coco_m"].to(device)
            if lm.any():
                ql = sims(xn[lm], vm_lvis, lvl).argmax(-1)
                a["lf"] += int((ql != r["lvis_c"].to(device)[lm]).sum()); a["ln"] += int(lm.sum())
            if cm.any():
                qc = sims(xn[cm], vm_coco, lvl).argmax(-1)
                a["cf"] += int((qc != r["coco_c"].to(device)[cm]).sum()); a["cn"] += int(cm.sum())
            fb = r["box"].to(device)
            a["bnum"] += float((b[lvl].float() - fb).pow(2).sum()); a["bden"] += float(fb.pow(2).sum())
    return dict(emb=a["emb"] / max(a["w"], 1e-12), lvis=100 * a["lf"] / max(a["ln"], 1),
                coco=100 * a["cf"] / max(a["cn"], 1), box=a["bnum"] / max(a["bden"], 1e-12),
                n_lvis=a["ln"], n_coco=a["cn"])


def fmt(name, mp, m, base=None):
    s = f"{name:>22} | {mp:>6.3f} | {m['emb']:>9.3e} | {m['lvis']:>6.2f} | {m['coco']:>6.2f} | {m['box']:>9.3e}"
    if base is not None:
        s += " | " + " | ".join(f"{m[k] - base[k]:>+9.3e}" if k in ("emb", "box") else f"{m[k] - base[k]:>+6.2f}"
                                for k in ("emb", "lvis", "coco", "box"))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--low-bits", type=int, default=4)
    ap.add_argument("--first-last-bits", type=int, default=8)
    ap.add_argument("--views", default="drop,restore")
    ap.add_argument("--expand", default="", help="conv 단위로 쪼갤 model.model 인덱스(쉼표 구분)")
    ap.add_argument("--only-expanded", action="store_true", help="쪼갠 conv 그룹만 측정")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.25)
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
    evals = [rc.preprocess(p, args.imgsz, "cpu") for p in eval_paths]

    fp = rc.build(YOLOWorld, args.model, coco, device, calib, "fp")
    head_fp = _find_head(fp.model)
    vm_lvis = VocabMetric(encode_text_bank(YOLOWorld, args.model, lvis_names, device)).bind_head(head_fp)
    vm_coco = VocabMetric(encode_text_bank(YOLOWorld, args.model, coco, device)).bind_head(head_fp)
    refs = reference(fp, evals, device, vm_lvis, vm_coco, args.conf)

    hdr = f"{'group':>22} | {'Mparam':>6} | {'emb_err':>9} | {'LVISfl':>6} | {'COCOfl':>6} | {'box_err':>9}"
    dhdr = " | " + " | ".join(f"{'d_' + k:>9}" if k in ("emb", "box") else f"{'d_' + k:>6}"
                              for k in ("emb", "lvis", "coco", "box"))
    for view in [v.strip() for v in args.views.split(",")]:
        base_bits, grp_bits = (8, args.low_bits) if view == "drop" else (args.low_bits, 8)
        q = rc.build(YOLOWorld, args.model, coco, device, calib, "naive", fp=fp,
                     w_bits=base_bits, a_bits=args.a_bits, skip_head=False,
                     first_last_bits=args.first_last_bits)
        expand = {int(x) for x in args.expand.split(",") if x.strip()}
        groups = groups_of(q.model, expand)
        if args.only_expanded:
            groups = [g for g in groups if g[0].split(".")[0].isdigit() and int(g[0].split(".")[0]) in expand
                      and ":" not in g[0]]
        base = measure(q, evals, refs, device, vm_lvis, vm_coco)
        print("\n" + "=" * 130)
        print(f" view={view}: 기준 W{base_bits}A{args.a_bits}, 그룹 하나만 W{grp_bits} "
              f"(첫/마지막 {args.first_last_bits}bit 고정), eval {len(evals)}장, "
              f"anchor LVIS {base['n_lvis']} / COCO {base['n_coco']}")
        print("=" * 130)
        print(hdr + dhdr)
        tot = sum(c.conv.weight.numel() for _, cs in groups for c in cs) / 1e6
        print(fmt(f"base W{base_bits}", tot, base))
        rows = []
        for name, convs in groups:
            mp = sum(c.conv.weight.numel() for c in convs) / 1e6
            set_wbits(convs, grp_bits)
            m = measure(q, evals, refs, device, vm_lvis, vm_coco)
            set_wbits(convs, base_bits)
            rows.append((name, mp, m))
            print(fmt(name, mp, m, base), flush=True)
        # 요약: LVIS_flip 변화량 기준 정렬 + 누적 비중
        sign = 1 if view == "drop" else -1
        rows.sort(key=lambda r: sign * (r[2]["lvis"] - base["lvis"]), reverse=True)
        total = sum(max(sign * (r[2]["lvis"] - base["lvis"]), 0) for r in rows)
        print(f"\n [{view}] LVIS_flip 기여 상위 (누적 비중 / 누적 Mparam)")
        cum = cmp = 0.0
        for name, mp, m in rows[:10]:
            d = max(sign * (m["lvis"] - base["lvis"]), 0)
            cum += d; cmp += mp
            print(f"   {name:>22}  d={sign * (m['lvis'] - base['lvis']):>+6.2f}  "
                  f"누적 {100 * cum / max(total, 1e-9):>5.1f}%  누적 {cmp:.3f}/{tot:.3f} Mparam")
        del q
        rc.free_cpu_mem()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
