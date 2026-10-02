# --post-build 스크립트: M 채택 검사 후보 지표를 빌드된 모델마다 잰다(누수 없음).
#   이미지: train2017 중 calibration에 쓰지 않은 n장(정렬 순서로 calib 다음부터). 어휘: COCO만. val/LVIS 미사용.
#   지표: COCO flip(FP가 확신하는 anchor에서 top-1 변경 %), 임베딩 방향 오차(emb), COCO logit 상대오차(cls), box 상대오차.
import glob as _g, os as _o, diag_w4_sensitivity as dg
_n = int(_o.environ.get("MCHECK_N", "200"))
_paths = sorted(_g.glob(_o.path.join(args.coco_root, "train2017", "*.jpg")))[args.calib:args.calib + _n]
_evals = [preprocess(p, args.imgsz, "cpu") for p in _paths]
dg.HS = dg.HeadSim(dg._find_head(fp.model))
_vm = dg.VocabMetric(dg.encode_text_bank(YOLOWorld, args.model, coco, device))
with torch.no_grad():
    _refs = dg.reference(fp, _evals, device, _vm, _vm, 0.25, "coco")
    for _c, _q in models.items():
        _m = dg.measure(_q, _evals, _refs, device, _vm, _vm)
        print(f"[mcheck] {args.model} W{args.w_bits}A{args.a_bits} seed{args.seed} {_c}: COCO flip {_m['coco']:.2f}%  "
              f"emb {_m['emb']:.4e}  cls {_m['cls']:.4e}  box {_m['box']:.4e}  (anchor {_m['n_coco']}, 이미지 {_n}장, calib 밖)", flush=True)
