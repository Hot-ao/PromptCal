"""seed1 naive W8A8(m)에서 activation 범위를 seed0 값으로 바꾸는 conv 집합을 이분 탐색해, COCO AP 붕괴를 만드는 conv를 찾는다."""
import sys, glob, json, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
from quant.fake_quant import QuantConv2d
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = 'yolov8m-world.pt'
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); calib = [rc.preprocess(p, 640, dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
obs = json.load(open('runs/152_m_instab/obs_yolov8m-world.json'))
torch.manual_seed(1); torch.cuda.manual_seed_all(1)
q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False)
convs = {n: m for n, m in q.model.named_modules() if isinstance(m, QuantConv2d)}
def setr(m, r):
    lo, hi = r['lo'], r['hi']; mn, mx = min(lo, 0.), max(hi, 0.); sc = max((mx - mn) / (2 ** m.a_obs.bits - 1), 1e-8)
    m.a_obs.scale = torch.tensor(sc, device=dev); m.a_obs.zero_point = torch.tensor(float(round(-mn / sc)), device=dev)
def ap_with(swap):
    for n, m in convs.items(): setr(m, obs['0' if n in swap else '1'][n])
    return rc.measure_ap(q, 'configs/coco_local.yaml', 640, dev)[0][0]
cand = [n for n in convs if abs((obs['0'][n]['hi'] - obs['0'][n]['lo']) / max(obs['1'][n]['hi'] - obs['1'][n]['lo'], 1e-9) - 1) > 0.02]
print(f"범위가 다른 conv {len(cand)}개", flush=True)
while len(cand) > 1:
    a = cand[:len(cand) // 2]; b = cand[len(cand) // 2:]
    apa = ap_with(set(a)); print(f"  앞쪽 {len(a)}개를 seed0 범위로: AP {apa:.2f}", flush=True)
    cand = a if apa > 35 else b
    if apa <= 35:
        apb = ap_with(set(b)); print(f"  뒤쪽 {len(b)}개를 seed0 범위로: AP {apb:.2f}", flush=True)
        if apb <= 35: print("  -> 단일 conv로 설명 안 됨(여러 conv가 함께 원인)", flush=True); break
n = cand[0]
print(f"원인 conv: {n}  seed1 {obs['1'][n]['lo']:.3f}~{obs['1'][n]['hi']:.3f}  seed0 {obs['0'][n]['lo']:.3f}~{obs['0'][n]['hi']:.3f}  관측 min/max {obs['1'][n]['mmlo']:.3f}/{obs['1'][n]['mmhi']:.3f}")
print(f"그 conv 하나만 seed0 범위로: AP {ap_with({n}):.2f}")
