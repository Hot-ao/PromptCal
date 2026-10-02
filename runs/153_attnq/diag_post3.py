# PTQ_POST_BUILD 훅: W4A5 attn_cls seed0 BRECQ 붕괴 -- conv weight를 FP로 되돌리며 이분 탐색(activation은 그대로 양자화).
from quant.adaround import AdaRoundQuantConv2d
q = models["brecq"]; dm = q.model
convs = [(n, mm) for n, mm in dm.named_modules() if isinstance(mm, AdaRoundQuantConv2d)]
def ap():
    return measure_ap(q, 'configs/coco_local.yaml', args.imgsz, args.device)[0][0]
def run_fp(sel):
    for n, mm in sel: mm.quant_weight = (lambda c=mm.conv: c.weight)
    r = ap()
    for n, mm in sel: del mm.quant_weight
    return r
print(f"[diag3] conv {len(convs)}개, 그대로 {ap():.2f}", flush=True)
cand = convs; TH = 26
while len(cand) > 1:
    h, t = cand[:len(cand) // 2], cand[len(cand) // 2:]
    rh = run_fp(h); print(f"[diag3] 앞 {len(h)}개({h[0][0]}~{h[-1][0]}) weight FP: {rh:.2f}", flush=True)
    if rh > TH: cand = h; continue
    rt = run_fp(t); print(f"[diag3] 뒤 {len(t)}개({t[0][0]}~{t[-1][0]}) weight FP: {rt:.2f}", flush=True)
    if rt > TH: cand = t; continue
    print("[diag3] 한쪽만 되돌려서는 회복 안 됨 -- 여러 conv가 함께 원인", flush=True); break
with torch.no_grad():
    for n, mm in cand:
        w = mm.conv.weight; wq = mm.quant_weight()
        rel = ((wq - w).norm() / w.norm()).item()
        print(f"[diag3] 후보 {n}: w_bits={mm.w_bits} weight 상대오차 {rel:.3f}  |W| max {w.abs().max().item():.3f}  |Wq| max {wq.abs().max().item():.3f}", flush=True)
    rel_all = sorted(((((mm.quant_weight() - mm.conv.weight).norm() / mm.conv.weight.norm()).item(), n) for n, mm in convs), reverse=True)[:8]
    print("[diag3] weight 상대오차 상위: " + ", ".join(f"{n} {r:.3f}" for r, n in rel_all), flush=True)
