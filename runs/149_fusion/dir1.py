"""09-30 설계 방향 1(게이트 교환) 싼 확인. 학습 없음(naive), train2017 이미지만(LVIS는 분석용 지표로만 사용)."""
import sys, glob, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
import diag_w4_sensitivity as dg
from quant.quant_model import wrap_convs, set_first_last_bits, set_conv_wbits, calibrate
from quant.fusion_quant import gate_commute, gate_commute_all, apply_branch_quant
from quant.fake_quant import QuantConv2d
from quant.pdquant import _find_head
from quant.vocab_metric import VocabMetric, encode_text_bank
from ultralytics import YOLOWorld
gpu, cap, n_eval = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
dev = f'cuda:{gpu}'
if cap > 0:
    torch.cuda.set_per_process_memory_fraction(cap * 1024**3 / torch.cuda.get_device_properties(dev).total_memory, dev)
torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
root = '/data/taeho/coco_datasets'; coco, lvis = rc.load_names('coco'), rc.load_names('lvis')
paths = sorted(glob.glob(f'{root}/train2017/*.jpg'))
calib = [rc.preprocess(p, 640, dev).to(dev) for p in paths[:128]]
evals = [rc.preprocess(p, 640, 'cpu') for p in paths[1000:1000 + n_eval]]   # calibration과 겹치지 않는 train 이미지
fp = YOLOWorld('yolov8s-world.pt'); fp.set_classes(coco); fp.fuse(); fp.model.to(dev).eval()
dg.HS = dg.HeadSim(_find_head(fp.model))
vl = VocabMetric(encode_text_bank(YOLOWorld, 'yolov8s-world.pt', lvis, dev))
vc = VocabMetric(encode_text_bank(YOLOWorld, 'yolov8s-world.pt', coco, dev))
refs = dg.reference(fp, evals, dev, vl, vc, 0.25, 'coco')
c12 = fp.model.model[12]
print(f"[info] C2fAttn(12): c={c12.c}, nh={c12.attn.nh}, cv2 in={c12.cv2.conv.in_channels}", flush=True)

def make(wb, ab, gate=None, low=(), branch=None):
    """gate: None | [block idx] | 'all'.  low: W4로 둘 conv 경로(나머지 wb).  branch: {경로: groups}"""
    m = YOLOWorld('yolov8s-world.pt'); m.set_classes(coco); m.fuse()
    if gate == 'all': gate_commute_all(m.model)
    elif gate: [gate_commute(m.model.model[i]) for i in gate]
    wrap_convs(m.model, wb, ab); set_first_last_bits(m.model, 8)
    if low: set_conv_wbits(m.model, list(low), 4)
    for path, groups in (branch or {}).items():
        idx, _, rest = path.partition('.')
        mod = m.model.model[int(idx)].get_submodule(rest)
        mod = mod if isinstance(mod, QuantConv2d) else mod.conv
        apply_branch_quant(mod, groups)
    m.model.to(dev).eval()
    calibrate(m.model, calib, device=dev, act_observer='mse')
    return m

# FP 등가성: 게이트 교환은 FP로 같은 함수여야 한다
g = YOLOWorld('yolov8s-world.pt'); g.set_classes(coco); g.fuse(); gate_commute_all(g.model); g.model.to(dev).eval()
with torch.no_grad():
    x = calib[0]; a_ = fp.model(x)[0]; b_ = g.model(x)[0]
print(f"[FP 등가] 게이트 교환 전체: max|diff|/max|ref| = {float((a_ - b_).abs().max() / a_.abs().max()):.2e}", flush=True)
del g
c = c12.c
side = [f"12.cv2_side.{h}" for h in range(c12.attn.nh)]
V = [
    ("W8A8 전체 (기준)",                       dict(wb=8, ab=8)),
    ("12.cv2 W4 (기존)",                        dict(wb=8, ab=8, low=['12.cv2'])),
    ("12.cv2 W4 + 분기별 scale",                dict(wb=8, ab=8, low=['12.cv2'], branch={'12.cv2': [(i*c, (i+1)*c) for i in range(4)]})),
    ("12 게이트 교환, main+side W4",            dict(wb=8, ab=8, gate=[12], low=['12.cv2_main'] + side)),
    ("12 게이트 교환 + main 분기별 scale",      dict(wb=8, ab=8, gate=[12], low=['12.cv2_main'] + side,
                                                  branch={'12.cv2_main': [(i*c, (i+1)*c) for i in range(3)]})),
    ("W4A8 전체 (기존)",                        dict(wb=4, ab=8)),
    ("W4A8 + 게이트 교환(C2fAttn 4곳)",         dict(wb=4, ab=8, gate='all')),
    ("W4A6 전체 (기존)",                        dict(wb=4, ab=6)),
    ("W4A6 + 게이트 교환(C2fAttn 4곳)",         dict(wb=4, ab=6, gate='all')),
]
print(f"\n{'변형':<34} | {'emb_err':>8} | {'COCO flip':>9} | {'LVIS flip':>9}")
for name, kw in V:
    q = make(**kw)
    r = dg.measure(q, evals, refs, dev, vl, vc)
    print(f"{name:<34} | {r['emb']:>8.4f} | {r['coco']:>8.2f}% | {r['lvis']:>8.2f}%", flush=True)
    del q; torch.cuda.empty_cache()
