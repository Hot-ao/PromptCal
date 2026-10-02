import glob as _g, os as _o
_x = preprocess(sorted(_g.glob(_o.path.join(args.coco_root, "train2017", "*.jpg")))[args.calib], args.imgsz, device).to(device)
for _c, _q in models.items():
    _mm = _q.model.model[-1].cv3[0][2]
    _st = {}
    _h = _mm.register_forward_pre_hook(lambda mod, i: _st.__setitem__('x', i[0].detach().float()))
    with torch.no_grad(): _q.model(_x)
    _h.remove()
    x = _st['x'] / _mm.mig if _mm.mig is not None else _st['x']
    o = _mm.a_obs
    d = _mm.lsq_act.delta.item() if getattr(_mm, 'use_lsq', False) else float('nan')
    print(f"[dbg3] {_c}: bits {o.bits}  a_obs.scale {o.scale.item():.4e} zp {o.zero_point.item():.1f}  lsq delta {d:.4e}  "
          f"입력(÷mig) min {x.min().item():.3f} max {x.max().item():.3f}  양자화 상한 {(2**o.bits-1-o.zero_point.item())*d:.3f} 하한 {(-o.zero_point.item())*d:.3f}"
          + (f"  mig min {_mm.mig.min().item():.3e} max {_mm.mig.max().item():.3e}" if _mm.mig is not None else ""), flush=True)
    with torch.no_grad():
        xq = _mm.quant_act(x)
    print(f"[dbg3] {_c}: 양자화 후 상대오차 {float((xq-x).norm()/x.norm()):.4e}", flush=True)
