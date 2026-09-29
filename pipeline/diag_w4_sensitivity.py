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
    hi = len(seq) - 1
    for branch in ("cv2", "cv3"):
        for li, lvl in enumerate(getattr(head, branch, [])):
            if hi in expand:                              # head도 conv 단위("23.cv3.0.1" 형식)
                for n, m in lvl.named_modules():
                    if isinstance(m, QuantConv2d) and id(m) not in fixed:
                        out.append((f"{hi}.{branch}.{li}.{n}" if n else f"{hi}.{branch}.{li}", [m]))
                continue
            convs = [m for m in lvl.modules() if isinstance(m, QuantConv2d) and id(m) not in fixed]
            if convs:
                out.append((f"head.{branch}.{li}", convs))
    return out


def set_wbits(convs, bits):
    for c in convs:
        c.w_bits = bits
        c.freeze_weight_quant()


def set_abits(convs, bits, model, calib):
    """09-29: activation(conv 입력) 비트를 바꾸고 그 conv의 observer만 MSE로 재보정(calib 이미지로 forward).
    다른 conv는 양자화 상태 그대로(재보정 중 대상 conv의 입력은 양자화 없이 통과)."""
    for c in convs:
        o = c.a_obs
        o.bits = bits
        o.min_val.fill_(float("inf")); o.max_val.fill_(float("-inf"))
        o.mm_min.fill_(float("inf")); o.mm_max.fill_(float("-inf"))
        o._mse_buf, o.ready, o.method = None, False, "mse"
        c.calibrating = True
    with torch.no_grad():
        for t in calib:
            model.model(t)
    for c in convs:
        c.calibrating = False
        c.a_obs.freeze()


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


class HeadSim:
    """09-29: head 종류별 임베딩/유사도. ContrastiveHead(v1) = L2 정규화 후 tau*cos+b,
    BNContrastiveHead(v2) = BN(FP, eval) 후 exp(logit_scale)*<e, w_hat>+b. emb_err는 v1은 단위벡터 차이
    ||e_q - e_fp||^2(기존과 동일), v2는 상대 오차 ||e_q - e_fp||^2 / ||e_fp||^2 (logit이 크기에도 선형이라)."""
    def __init__(self, fp_head):
        self.subs = list(fp_head.cv4)
        self.kind = type(self.subs[0]).__name__
        assert self.kind in ("ContrastiveHead", "BNContrastiveHead"), self.kind

    @torch.no_grad()
    def embed(self, x, lvl):
        if self.kind == "ContrastiveHead":
            return F.normalize(VocabMetric.flat(x).float(), dim=-1)
        return VocabMetric.flat(self.subs[lvl].norm(x.float())).float()

    def emb_err(self, eq, ef):
        d = (eq - ef).pow(2).sum(-1)
        return d if self.kind == "ContrastiveHead" else d / ef.pow(2).sum(-1).clamp(min=1e-12)

    def sims(self, e, bank, lvl):
        sub = self.subs[lvl]
        return float(sub.logit_scale.exp()) * (e @ bank.T) + float(sub.bias)


HS = None                                                 # main()에서 FP head로 초기화


def sims(e, vm, lvl):
    return HS.sims(e, vm.bank, lvl)


@torch.no_grad()
def reference(fp, evals, device, vm_lvis, vm_coco, conf, weight_vocab="lvis"):
    """FP 쪽 기준값을 이미지별로 1회 계산해 CPU에 보관. weight_vocab: emb_err의 anchor 가중치를 어느
    vocabulary 확신도로 줄지('lvis' = 기존, 평가 vocabulary라 선택 용도로는 누수 / 'coco' = calibration
    vocabulary / 'uniform')."""
    refs = []
    for t in evals:
        e, b = run_tap(fp, t, device)
        per = []
        for lvl, x in enumerate(e):
            xn = HS.embed(x, lvl)
            sl, sc = sims(xn, vm_lvis, lvl), sims(xn, vm_coco, lvl)
            pl, cl = sl.sigmoid().max(-1)
            pc, cc = sc.sigmoid().max(-1)
            w = {"lvis": pl + 0.05, "coco": pc + 0.05, "uniform": torch.ones_like(pl)}[weight_vocab]
            per.append(dict(xn=xn.half().cpu(), w=w.cpu(), cls=sc.half().cpu(),
                            lvis_m=(pl > conf).cpu(), lvis_c=cl.cpu(),
                            coco_m=(pc > conf).cpu(), coco_c=cc.cpu(),
                            box=b[lvl].float().cpu()))
        refs.append(per)
    return refs


@torch.no_grad()
def measure(q, evals, refs, device, vm_lvis, vm_coco):
    a = dict(w=0.0, emb=0.0, lf=0, ln=0, cf=0, cn=0, bnum=0.0, bden=0.0, cnum=0.0, cden=0.0)
    for t, per in zip(evals, refs):
        e, b = run_tap(q, t, device)
        for lvl, x in enumerate(e):
            r = per[lvl]
            xn = HS.embed(x, lvl)
            f = r["xn"].to(device).float()
            w = r["w"].to(device)
            a["w"] += float(w.sum())
            a["emb"] += float((w * HS.emb_err(xn, f)).sum())
            lm, cm = r["lvis_m"].to(device), r["coco_m"].to(device)
            if lm.any():
                ql = sims(xn[lm], vm_lvis, lvl).argmax(-1)
                a["lf"] += int((ql != r["lvis_c"].to(device)[lm]).sum()); a["ln"] += int(lm.sum())
            if cm.any():
                qc = sims(xn[cm], vm_coco, lvl).argmax(-1)
                a["cf"] += int((qc != r["coco_c"].to(device)[cm]).sum()); a["cn"] += int(cm.sum())
            # 출력 MSE 기준(09-29): 검출기 최종 출력 중 COCO(=calibration vocabulary) 클래스 logit의 상대 오차.
            # Hessian 기반 배분(HAWQ류)이 근사하려는 "출력 변화"를 직접 잰 값 -- 순위 비교 기준으로 쓴다.
            fc = r["cls"].to(device).float()
            a["cnum"] += float((sims(xn, vm_coco, lvl) - fc).pow(2).sum()); a["cden"] += float(fc.pow(2).sum())
            fb = r["box"].to(device)
            a["bnum"] += float((b[lvl].float() - fb).pow(2).sum()); a["bden"] += float(fb.pow(2).sum())
    return dict(emb=a["emb"] / max(a["w"], 1e-12), lvis=100 * a["lf"] / max(a["ln"], 1),
                cls=a["cnum"] / max(a["cden"], 1e-12),
                coco=100 * a["cf"] / max(a["cn"], 1), box=a["bnum"] / max(a["bden"], 1e-12),
                n_lvis=a["ln"], n_coco=a["cn"])


def fmt(name, mp, m, base=None):
    s = f"{name:>22} | {mp:>6.3f} | {m['emb']:>9.3e} | {m['lvis']:>6.2f} | {m['coco']:>6.2f} | {m['box']:>9.3e}"
    if base is not None:
        s += " | " + " | ".join(f"{m[k] - base[k]:>+9.3e}" if k in ("emb", "box") else f"{m[k] - base[k]:>+6.2f}"
                                for k in ("emb", "lvis", "coco", "box"))
    return s


def rank_convs_leakfree(model_path, coco_root, device, n_eval=200, n_calib=256, imgsz=640, conf=0.25,
                        low_bits=4, first_last_bits=8, seed=0):
    """09-29 (P0): 보호할 conv를 고르기 위한 누수 없는 conv 단위 drop 진단.
    - 이미지: train2017 앞 n_eval장(calibration 부분집합). 평가셋(val2017) 미사용.
    - 어휘: COCO(= calibration vocabulary)만 로드. LVIS는 로드조차 하지 않는다(내부 LVIS 슬롯에도 COCO를 넣음).
    - 기준 모델: naive W8A8(head 포함, 첫/마지막 first_last_bits). conv 하나씩 weight만 low_bits로 내려 변화량을 잰다.
    - 전역 torch RNG 상태를 저장·복원한다(호출 뒤 BRECQ 등의 RNG 스트림이 이 진단 때문에 밀리지 않도록).
    반환: [{name, mparam, emb, coco, box, cls}] (변화량, 기준 대비)."""
    global HS
    from ultralytics import YOLOWorld
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        coco = rc.load_names("coco")
        paths = sorted(glob.glob(os.path.join(coco_root, "train2017", "*.jpg")))
        calib = [rc.preprocess(p, imgsz, device) for p in paths[:n_calib]]
        evals = [rc.preprocess(p, imgsz, "cpu") for p in paths[:n_eval]]
        fp = rc.build(YOLOWorld, model_path, coco, device, calib, "fp")
        HS = HeadSim(_find_head(fp.model))
        vm_coco = VocabMetric(encode_text_bank(YOLOWorld, model_path, coco, device))
        refs = reference(fp, evals, device, vm_coco, vm_coco, conf, "coco")
        q = rc.build(YOLOWorld, model_path, coco, device, calib, "naive", fp=fp, w_bits=8, a_bits=8,
                     skip_head=False, first_last_bits=first_last_bits)
        groups = groups_of(q.model, set(range(len(q.model.model))))
        base = measure(q, evals, refs, device, vm_coco, vm_coco)
        rows = []
        for name, convs in groups:
            set_wbits(convs, low_bits)
            m = measure(q, evals, refs, device, vm_coco, vm_coco)
            set_wbits(convs, 8)
            rows.append(dict(name=name, mparam=sum(c.conv.weight.numel() for c in convs) / 1e6,
                             **{k: m[k] - base[k] for k in ("emb", "coco", "box", "cls")}))
        del q, fp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def select_protected(rows, budget_frac, criterion="coco"):
    """예산(전체 양자화 weight 파라미터 대비 비율) 안에서 criterion 변화량이 큰 순으로 greedy 선택.
    runs/134 select.py와 같은 규칙: 예산의 110%를 넘지 않게 담고, 90% 이상 차면 멈춘다."""
    total = sum(r["mparam"] for r in rows)
    budget = budget_frac * total
    sel, tot = [], 0.0
    for r in sorted(rows, key=lambda r: r[criterion], reverse=True):
        if tot + r["mparam"] <= budget * 1.1:
            sel.append(r["name"]); tot += r["mparam"]
        if tot >= budget * 0.9:
            break
    return sel, tot, total


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
    ap.add_argument("--conv-level", action="store_true", help="모든 모듈(head 포함)을 conv 단위로")
    ap.add_argument("--drop-kind", choices=("w", "a"), default="w",
                    help="w = weight 비트를 내림(기존), a = conv 입력 activation 비트를 내림(weight는 8bit 유지)")
    ap.add_argument("--recal-images", type=int, default=32, help="--drop-kind a에서 observer 재보정 이미지 수")
    ap.add_argument("--eval-source", choices=("val", "train"), default="val",
                    help="진단 이미지. 'train' = calibration과 같은 train2017 앞쪽(평가셋 누수 없음)")
    ap.add_argument("--weight-vocab", choices=("lvis", "coco", "uniform"), default="lvis",
                    help="emb_err anchor 가중치 vocabulary. 선택 용도로는 coco/uniform(평가 vocabulary 누수 없음)")
    ap.add_argument("--rank-json", default="", help="drop view의 conv별 결과를 JSON으로 저장")
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
    eval_paths = sorted(glob.glob(os.path.join(args.coco_root, "val2017" if args.eval_source == "val"
                                               else "train2017", "*.jpg")))[:args.n_eval]
    calib = [rc.preprocess(p, args.imgsz, device) for p in calib_paths]
    evals = [rc.preprocess(p, args.imgsz, "cpu") for p in eval_paths]

    fp = rc.build(YOLOWorld, args.model, coco, device, calib, "fp")
    head_fp = _find_head(fp.model)
    global HS
    HS = HeadSim(head_fp)
    print(f"[head] {HS.kind}")
    vm_lvis = VocabMetric(encode_text_bank(YOLOWorld, args.model, lvis_names, device))
    vm_coco = VocabMetric(encode_text_bank(YOLOWorld, args.model, coco, device))
    refs = reference(fp, evals, device, vm_lvis, vm_coco, args.conf, args.weight_vocab)

    hdr = f"{'group':>22} | {'Mparam':>6} | {'emb_err':>9} | {'LVISfl':>6} | {'COCOfl':>6} | {'box_err':>9}"
    dhdr = " | " + " | ".join(f"{'d_' + k:>9}" if k in ("emb", "box") else f"{'d_' + k:>6}"
                              for k in ("emb", "lvis", "coco", "box"))
    for view in [v.strip() for v in args.views.split(",")]:
        base_bits, grp_bits = (8, args.low_bits) if view == "drop" else (args.low_bits, 8)
        wb, ab = (base_bits, args.a_bits) if args.drop_kind == "w" else (8, base_bits)
        q = rc.build(YOLOWorld, args.model, coco, device, calib, "naive", fp=fp,
                     w_bits=wb, a_bits=ab, skip_head=False,
                     first_last_bits=args.first_last_bits)
        recal = [t.to(device) for t in calib[:args.recal_images]]
        expand = {int(x) for x in args.expand.split(",") if x.strip()}
        if args.conv_level:
            expand = set(range(len(q.model.model)))
        groups = groups_of(q.model, expand)
        if args.only_expanded:
            groups = [g for g in groups if g[0].split(".")[0].isdigit() and int(g[0].split(".")[0]) in expand
                      and ":" not in g[0]]
        base = measure(q, evals, refs, device, vm_lvis, vm_coco)
        print("\n" + "=" * 130)
        print(f" view={view} kind={args.drop_kind}: 기준 W{wb}A{ab}, 그룹 하나만 {'W' if args.drop_kind == 'w' else 'A'}{grp_bits} "
              f"(첫/마지막 {args.first_last_bits}bit 고정), eval {len(evals)}장, "
              f"anchor LVIS {base['n_lvis']} / COCO {base['n_coco']}")
        print("=" * 130)
        print(hdr + dhdr)
        tot = sum(c.conv.weight.numel() for _, cs in groups for c in cs) / 1e6
        print(fmt(f"base W{base_bits}", tot, base))
        rows = []
        for name, convs in groups:
            mp = sum(c.conv.weight.numel() for c in convs) / 1e6
            if args.drop_kind == "w":
                set_wbits(convs, grp_bits)
                m = measure(q, evals, refs, device, vm_lvis, vm_coco)
                set_wbits(convs, base_bits)
            else:
                set_abits(convs, grp_bits, q, recal)
                m = measure(q, evals, refs, device, vm_lvis, vm_coco)
                set_abits(convs, base_bits, q, recal)
            rows.append((name, mp, m))
            print(fmt(name, mp, m, base), flush=True)
        if args.rank_json and view == "drop":
            import json
            json.dump([dict(name=n, mparam=mp, **{k: m[k] - base[k] for k in ("emb", "lvis", "coco", "box", "cls")})
                       for n, mp, m in rows], open(args.rank_json, "w"), indent=1)
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
