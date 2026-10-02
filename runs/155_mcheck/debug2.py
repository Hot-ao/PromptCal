import glob as _g, os as _o
_p = sorted(_g.glob(_o.path.join(args.coco_root, "train2017", "*.jpg")))[args.calib]
_x = preprocess(_p, args.imgsz, device).to(device)
import torch.nn.functional as F
def _cap(m):
    st = {}; hs = []
    head = m.model.model[-1]
    for br in ('cv2', 'cv3'):
        for j, sub in enumerate(getattr(head, br)[0]):
            hs.append(sub.register_forward_hook(lambda mod, i, o, k=f'{br}.0.{j}': st.__setitem__(k, o.detach().float())))
    hs.append(head.cv4[0].register_forward_hook(lambda mod, i, o: st.__setitem__('cv4.0.in', i[0].detach().float())))
    for idx in (15, 22):
        hs.append(m.model.model[idx].register_forward_hook(lambda mod, i, o, k=f'blk{idx}': st.__setitem__(k, o.detach().float())))
    with torch.no_grad(): m.model(_x)
    for h in hs: h.remove()
    return st
_f = _cap(fp)
for _c, _q in models.items():
    _s = _cap(_q)
    for k in _f:
        a, b = _s[k], _f[k]
        rel = float((a - b).norm() / b.norm()); cos = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        print(f"[dbg2] {_c} {k}: rel {rel:.4f} cos {cos:.4f}", flush=True)
    from quant.adaround import AdaRoundQuantConv2d as _A
    for n, mm in _q.model.named_modules():
        if isinstance(mm, _A) and n.startswith('model.23.cv3.0'):
            print(f"[dbg2] {_c} {n}: mig={'있음' if mm.mig is not None else '없음'} soft={mm.soft} a_bits={mm.a_obs.bits} use_lsq={getattr(mm,'use_lsq',None)}", flush=True)
