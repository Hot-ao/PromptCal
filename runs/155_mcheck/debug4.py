import glob as _g, os as _o
import torch.nn.functional as F
_x = preprocess(sorted(_g.glob(_o.path.join(args.coco_root, "train2017", "*.jpg")))[args.calib], args.imgsz, device).to(device)
fph = fp.model.model[-1]
for _c, _q in [(k, v) for k, v in models.items() if k.startswith("brecq")]:
    qh = _q.model.model[-1]
    for br in ('cv3', 'cv2'):
        for l in range(3):
            mm, fc = getattr(qh, br)[l][2], getattr(fph, br)[l][2]
            st = {}
            h = mm.register_forward_pre_hook(lambda mod, i: st.__setitem__('x', i[0].detach().float()))
            with torch.no_grad(): _q.model(_x)
            h.remove()
            x = st['x']
            with torch.no_grad():
                ref = fc(x)                                              # FP conv, 같은 입력
                act = mm(x)                                              # 실제 양자화 conv
                xs = x / mm.mig if mm.mig is not None else x
                fpm = F.conv2d(xs, mm.conv.weight, mm.conv.bias)         # 이전된 FP weight
                wq = F.conv2d(xs, mm.quant_weight(), mm.conv.bias)       # 양자화 weight만
            r = lambda a: float((a - ref).norm() / ref.norm())
            print(f"[dbg4] {_c} {br}.{l}.2 mig={'O' if mm.mig is not None else 'X'}: 실제 {r(act):.4f}  이전FP {r(fpm):.4f}  weight만양자화 {r(wq):.4f}"
                  f"  |W·s|max {mm.conv.weight.abs().max().item():.3f} |Wq|max {mm.quant_weight().abs().max().item():.3f} w_bits {mm.w_bits}", flush=True)
