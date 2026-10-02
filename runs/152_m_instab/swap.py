"""m naive W8A8: seed1(나쁜 쪽) 모델에서 conv 하나씩 activation 범위를 seed0 값으로 바꿔, 어느 conv가 AP 붕괴를 만드는지 찾는다.
지표: COCO flip / emb_err (train2017 1000~1099, 학습 없음)."""
import sys, glob, json, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc, diag_w4_sensitivity as dg
from quant.fake_quant import QuantConv2d
from quant.pdquant import _find_head
from quant.vocab_metric import VocabMetric, encode_text_bank
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = 'yolov8m-world.pt'
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); paths = sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))
calib = [rc.preprocess(p, 640, dev) for p in paths[:256]]; evals = [rc.preprocess(p, 640, 'cpu') for p in paths[1000:1100]]
fp = rc.build(YOLOWorld, model, coco, dev, calib, 'fp'); dg.HS = dg.HeadSim(_find_head(fp.model))
vc = VocabMetric(encode_text_bank(YOLOWorld, model, coco, dev)); refs = dg.reference(fp, evals, dev, vc, vc, 0.25, 'coco')
obs = json.load(open('runs/152_m_instab/obs_yolov8m-world.json'))
def setrange(o, lo, hi):
    o.min_val.fill_(lo); o.max_val.fill_(hi)
    qmax = 2 ** o.bits - 1
    mn = min(lo, 0.0); mx = max(hi, 0.0); sc = max((mx - mn) / qmax, 1e-8)
    o.scale = torch.tensor(sc, device=o.min_val.device); o.zero_point = torch.tensor(float(round(-mn / sc)), device=o.min_val.device)
torch.manual_seed(1); torch.cuda.manual_seed_all(1)
q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False)
convs = {n: m for n, m in q.model.named_modules() if isinstance(m, QuantConv2d)}
def meas(): r = dg.measure(q, evals, refs, dev, vc, vc); return r['box'], r['coco']
b1 = meas(); print(f"seed1 원본: box_err {b1[0]:.4f}, COCO flip {b1[1]:.2f}%", flush=True)
for n, m in convs.items(): setrange(m.a_obs, obs['0'][n]['lo'], obs['0'][n]['hi'])
b0 = meas(); print(f"seed1 모델의 범위를 전부 seed0 값으로: box_err {b0[0]:.4f}, COCO flip {b0[1]:.2f}%", flush=True)
for n, m in convs.items(): setrange(m.a_obs, obs['1'][n]['lo'], obs['1'][n]['hi'])
rows = []
for n, m in convs.items():
    r0, r1 = obs['0'][n], obs['1'][n]
    if abs((r0['hi'] - r0['lo']) / max(r1['hi'] - r1['lo'], 1e-9) - 1) < 0.05: continue
    setrange(m.a_obs, r0['lo'], r0['hi']); f, e = meas(); setrange(m.a_obs, r1['lo'], r1['hi'])
    rows.append((f - b1[0], n, f, e, (r1['lo'], r1['hi']), (r0['lo'], r0['hi'])))
rows.sort()
print("conv 하나만 seed0 범위로 바꿨을 때 box_err 변화 (음수 = 회복), 상위 10")
for d, n, f, e, r1, r0 in rows[:10]:
    print(f"  {d:+.4f}  {n:40s} box_err {f:.4f}  seed1 범위 {r1[0]:.2f}~{r1[1]:.2f} -> seed0 {r0[0]:.2f}~{r0[1]:.2f}", flush=True)
