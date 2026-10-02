# PTQ_POST_BUILD 훅: W4A5 attn_cls BRECQ 붕괴 원인 -- 양자화 지점을 하나씩 끄고 COCO AP.
from quant.attn_quant import QPoint, ConstQ, QuantLinear
q = models["brecq"]; dm = q.model
def ap():
    r = measure_ap(q, 'configs/coco_local.yaml', args.imgsz, args.device); return r[0][0]
def setq(pred, on):
    for n, mm in dm.named_modules():
        if getattr(mm, '_is_qpoint', False) and pred(n): mm.quantized = on
print(f"[diag] 그대로: {ap():.2f}", flush=True)
groups = {'head.cv4 x(영역 임베딩)': lambda n: 'cv4' in n and n.endswith('_aq_mods.x'),
          'head.cv4 w(텍스트)': lambda n: 'cv4' in n and n.endswith('_aq_mods.w'),
          'C2fAttn 게이트 전체': lambda n: '.attn.' in n,
          'ImagePoolingAttn 전체': lambda n: n.startswith('model.16')}
for k, pr in groups.items():
    setq(pr, False); print(f"[diag] {k} 끔: {ap():.2f}", flush=True); setq(pr, True)
for n, mm in dm.named_modules():
    if 'cv4' in n and n.endswith('_aq_mods.x'):
        o = mm.a_obs; print(f"[diag] {n} 범위 scale={o.scale.item():.4g} zp={o.zero_point.item():.0f} -> [{(-o.zero_point*o.scale).item():.3f}, {((255-o.zero_point)*o.scale).item():.3f}]  mm=[{o.mm_min.item():.3f},{o.mm_max.item():.3f}]", flush=True)
