"""m-World naive W8A8가 seed마다 COCO AP 40 vs 24로 흔들리는 원인 진단.
naive에서 seed가 바꾸는 것은 ActObserver(mse)의 무작위 표본 추출뿐이다 -> conv별 activation 범위를 seed 0/1로 비교."""
import sys, glob, json, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
from quant.fake_quant import QuantConv2d
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = sys.argv[2]
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco')
calib = [rc.preprocess(p, 640, dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
res = {}
for seed in (0, 1, 2):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False)
    res[seed] = {n: dict(lo=float(m.a_obs.min_val), hi=float(m.a_obs.max_val), mmlo=float(m.a_obs.mm_min), mmhi=float(m.a_obs.mm_max))
                 for n, m in q.model.named_modules() if isinstance(m, QuantConv2d)}
    del q; torch.cuda.empty_cache()
json.dump(res, open(f'runs/152_m_instab/obs_{model[:-3]}.json', 'w'), indent=1)
rows = []
for n in res[0]:
    his = [res[s][n]['hi'] for s in res]; los = [res[s][n]['lo'] for s in res]
    rng = [h - l for h, l in zip(his, los)]
    rows.append((max(rng) / max(min(rng), 1e-9), n, rng, res[0][n]['mmhi'], res[0][n]['mmlo']))
rows.sort(reverse=True)
print(f"{model}: seed 0/1/2 간 activation 범위(hi-lo) 최대/최소 비 상위 12")
for r, n, rng, mh, ml in rows[:12]:
    print(f"  x{r:6.2f}  {n:40s} 범위 {[round(x, 2) for x in rng]}  (관측 min/max {ml:.1f}/{mh:.1f})")
