"""마지막 레이어 activation 16bit 프로토콜이 각 모델의 naive W8A8 COCO AP에 주는 영향 (seed 0,1).
변형: 기존 / cv3 마지막 3개 A16 / cv2+cv3 마지막 6개 A16."""
import sys, glob, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
_orig = rc.set_conv_abits
rc.set_conv_abits = lambda dm, names, bits=8: _orig(dm, names, 16)
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = sys.argv[2]
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); calib = [rc.preprocess(p, 640, dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
_m = YOLOWorld(model); H = len(_m.model.model) - 1; del _m
C3 = tuple(f'{H}.cv3.{l}.2' for l in range(3)); C2 = tuple(f'{H}.cv2.{l}.2' for l in range(3))
for seed in (0, 1):
    out = []
    for name, convs in (('기존', ()), ('cv3 A16', C3), ('cv2+cv3 A16', C2 + C3)):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False, hi_abit_convs=convs)
        out.append(f"{name} {rc.measure_ap(q, 'configs/coco_local.yaml', 640, dev)[0][0]:.2f}"); del q; torch.cuda.empty_cache()
    print(f"{model} seed{seed}: " + " | ".join(out), flush=True)
