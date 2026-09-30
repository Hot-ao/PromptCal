"""09-30 설계 방향 2(분기 인식 concat 양자화) 싼 확인. 학습 없음(naive). activation 저비트에서 붕괴가 풀리는지."""
import sys, glob, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
import diag_w4_sensitivity as dg
from quant.quant_model import wrap_convs, set_first_last_bits, calibrate
from quant.fusion_quant import gate_commute_all, apply_branch_quant, branch_groups_from_trace
from quant.channel_graph import trace_channel_producers
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
evals = [rc.preprocess(p, 640, 'cpu') for p in paths[1000:1000 + n_eval]]
fp = YOLOWorld('yolov8s-world.pt'); fp.set_classes(coco); fp.fuse(); fp.model.to(dev).eval()
dg.HS = dg.HeadSim(_find_head(fp.model))
vl = VocabMetric(encode_text_bank(YOLOWorld, 'yolov8s-world.pt', lvis, dev))
vc = VocabMetric(encode_text_bank(YOLOWorld, 'yolov8s-world.pt', coco, dev))
refs = dg.reference(fp, evals, dev, vl, vc, 0.25, 'coco')

def make(wb, ab, branch=None, gate=False):
    """branch: None | 'aw'(activation+weight) | 'a'(activation만)"""
    m = YOLOWorld('yolov8s-world.pt'); m.set_classes(coco); m.fuse()
    if gate: gate_commute_all(m.model)
    wrap_convs(m.model, wb, ab); set_first_last_bits(m.model, 8)
    m.model.to(dev).eval()
    n_app = 0
    if branch:
        deps, _, cons = trace_channel_producers(m.model, calib[0], dev)
        for ci, (name, qc) in enumerate(cons):
            groups = branch_groups_from_trace(deps[ci], qc.conv.in_channels)
            if len(groups) > 1:
                apply_branch_quant(qc, groups, act=True, weight=(branch == 'aw')); n_app += 1
    calibrate(m.model, calib, device=dev, act_observer='mse')
    return m, n_app

print(f"{'설정':<6} {'변형':<26} | {'적용 conv':>8} | {'emb_err':>8} | {'COCO flip':>9} | {'LVIS flip':>9}", flush=True)
for wb, ab in ((8, 4), (4, 4), (4, 5), (4, 6)):
    for vname, kw in (("기존", {}), ("분기별 act scale", dict(branch='a')), ("분기별 act+weight scale", dict(branch='aw')),
                      ("분기별 act+weight + 게이트 교환", dict(branch='aw', gate=True))):
        q, n_app = make(wb, ab, **kw)
        r = dg.measure(q, evals, refs, dev, vl, vc)
        print(f"W{wb}A{ab}  {vname:<26} | {n_app:>8} | {r['emb']:>8.4f} | {r['coco']:>8.2f}% | {r['lvis']:>8.2f}%", flush=True)
        del q; torch.cuda.empty_cache()
