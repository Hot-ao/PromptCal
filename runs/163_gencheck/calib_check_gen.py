# --post-build 스크립트: M 채택 검사를 COCO 어휘와 일반 어휘(WordNet 명사 4,000개, COCO·LVIS 이름 제외)로 함께 잰다.
#   이미지: train2017에서 calibration 다음 200장(calib 밖). val2017·LVIS는 쓰지 않는다(누수 없음).
#   일반 어휘 flip: FP가 일반 어휘 기준으로 확신(sigmoid > 0.25)하는 anchor에서 top-1(4,000개 중)이 바뀐 비율.
import glob as _g, os as _o, diag_w4_sensitivity as dg
from quant.vocab_metric import load_vocab as _lv
_paths = sorted(_g.glob(_o.path.join(args.coco_root, "train2017", "*.jpg")))[args.calib:args.calib + 200]
_evals = [preprocess(p, args.imgsz, "cpu") for p in _paths]
dg.HS = dg.HeadSim(dg._find_head(fp.model))
_vm_c = dg.VocabMetric(dg.encode_text_bank(YOLOWorld, args.model, coco, device))
_gen, _ = _lv("configs/vocab_generic.txt", coco, None)
_vm_g = dg.VocabMetric(dg.encode_text_bank(YOLOWorld, args.model, _gen, device))
with torch.no_grad():
    _refs = dg.reference(fp, _evals, device, _vm_g, _vm_c, 0.25, "coco")
    for _c, _q in models.items():
        _m = dg.measure(_q, _evals, _refs, device, _vm_g, _vm_c)
        print(f"[gencheck] {args.model} W{args.w_bits}A{args.a_bits} seed{args.seed} {_c}: COCO flip {_m['coco']:.2f}% (anchor {_m['n_coco']}) | "
              f"일반 어휘 flip {_m['lvis']:.2f}% (anchor {_m['n_lvis']}, 어휘 {len(_gen)}개)", flush=True)
