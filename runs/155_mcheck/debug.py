import glob as _g, os as _o, diag_w4_sensitivity as dg
_paths = sorted(_g.glob(_o.path.join(args.coco_root, "train2017", "*.jpg")))[args.calib:args.calib + 30]
_evals = [preprocess(p, args.imgsz, "cpu") for p in _paths]
dg.HS = dg.HeadSim(dg._find_head(fp.model))
_vm = dg.VocabMetric(dg.encode_text_bank(YOLOWorld, args.model, coco, device))
with torch.no_grad():
    _refs = dg.reference(fp, _evals, device, _vm, _vm, 0.25, "coco")
    _order = list(models) + list(models)[::-1] + ['fp']
    for _c in _order:
        _q = fp if _c == 'fp' else models[_c]
        _m = dg.measure(_q, _evals, _refs, device, _vm, _vm)
        print(f"[dbg] {_c}: flip {_m['coco']:.2f}  emb {_m['emb']:.4e}", flush=True)
    # 마지막으로 FP 기준을 다시 계산해 처음 것과 같은지(fp 상태 오염 여부)
    _refs2 = dg.reference(fp, _evals[:3], device, _vm, _vm, 0.25, "coco")
    print("[dbg] fp 기준 재계산 차이:", max(float((a['xn'].float() - b['xn'].float()).abs().max()) for p1, p2 in zip(_refs[:3], _refs2) for a, b in zip(p1, p2)), flush=True)
