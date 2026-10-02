# PTQ_POST_BUILD 훅: W4A5 attn_cls seed0 BRECQ 붕괴 conv 찾기 -- conv activation 양자화를 구간별로 끄며 이분 탐색.
from quant.adaround import AdaRoundQuantConv2d
q = models["brecq"]; dm = q.model
convs = [(n, mm) for n, mm in dm.named_modules() if isinstance(mm, AdaRoundQuantConv2d)]
def ap():
    return measure_ap(q, 'configs/coco_local.yaml', args.imgsz, args.device)[0][0]
def run_off(sel):
    for n, mm in sel: mm.act_quant_enabled = False
    r = ap()
    for n, mm in sel: mm.act_quant_enabled = True
    return r
print(f"[diag2] conv {len(convs)}개, 그대로 {ap():.2f}", flush=True)
r_all = run_off(convs); print(f"[diag2] conv activation 양자화 전부 끔: {r_all:.2f}", flush=True)
for n, mm in convs: mm._wbak = mm.quant_weight  # weight 쪽 확인용
cand = convs
if r_all > 25:
    while len(cand) > 1:
        h = cand[:len(cand) // 2]; t = cand[len(cand) // 2:]
        rh = run_off(h); print(f"[diag2] 앞 {len(h)}개({h[0][0]}~{h[-1][0]}) 끔: {rh:.2f}", flush=True)
        if rh > 25: cand = h; continue
        rt = run_off(t); print(f"[diag2] 뒤 {len(t)}개({t[0][0]}~{t[-1][0]}) 끔: {rt:.2f}", flush=True)
        if rt > 25: cand = t; continue
        print("[diag2] 한쪽만 꺼서는 회복 안 됨 -- 여러 conv가 함께 원인", flush=True); break
    for n, mm in cand:
        o = mm.a_obs; s = (o.scale * (mm.s_mult.clamp(0.1, 10).max() if getattr(mm, 's_mult', None) is not None else 1)).item()
        lo = (-o.zero_point * o.scale).item(); hi = ((2 ** o.bits - 1 - o.zero_point) * o.scale).item()
        print(f"[diag2] 후보 {n}: bits={o.bits} 범위 [{lo:.3f}, {hi:.3f}] 관측 min/max [{o.mm_min.item():.3f}, {o.mm_max.item():.3f}]", flush=True)
else:
    print("[diag2] activation이 원인이 아님 -- weight(AdaRound 반올림) 쪽", flush=True)
