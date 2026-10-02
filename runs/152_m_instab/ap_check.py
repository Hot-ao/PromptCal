"""m naive W8A8 seed0/1의 COCO AP를 같은 스크립트에서 다시 재고, 실제 붕괴인지 확인. seed1에서 activation 범위를 seed0 값으로 바꾼 것도 잰다."""
import sys, glob, json, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
from quant.fake_quant import QuantConv2d
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = 'yolov8m-world.pt'
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); calib = [rc.preprocess(p, 640, dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
data = 'configs/coco_ultra.yaml' if len(sys.argv) < 3 else sys.argv[2]
obs = json.load(open('runs/152_m_instab/obs_yolov8m-world.json'))
for seed in (0, 1):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False)
    ap = rc.measure_ap(q, data, 640, dev); ap = ap[0] if isinstance(ap[0], tuple) else ap
    print(f"seed{seed} naive W8A8 COCO AP {ap[0]:.2f}", flush=True)
    if seed == 1:
        for n, m in q.model.named_modules():
            if isinstance(m, QuantConv2d):
                o = m.a_obs; lo, hi = obs['0'][n]['lo'], obs['0'][n]['hi']; mn, mx = min(lo, 0.), max(hi, 0.); sc = max((mx - mn) / 255, 1e-8)
                o.scale = torch.tensor(sc, device=dev); o.zero_point = torch.tensor(float(round(-mn / sc)), device=dev)
        print(f"seed1 모델 + seed0 activation 범위: COCO AP {(lambda a: a[0][0] if isinstance(a[0], tuple) else a[0])(rc.measure_ap(q, data, 640, dev)):.2f}", flush=True)
    del q; torch.cuda.empty_cache()
