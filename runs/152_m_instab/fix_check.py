"""원인 conv(head cv3 마지막 1x1의 입력)를 activation 16bit로 두면 seed와 무관하게 안정적인가. head cv3 마지막 conv 3개 모두 A16으로."""
import sys, glob, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
from quant.quant_model import set_conv_abits
_orig = rc.set_conv_abits
rc.set_conv_abits = lambda dm, names, bits=8: _orig(dm, names, 16)
from quant.fake_quant import QuantConv2d
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = sys.argv[2]
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); calib = [rc.preprocess(p, 640, dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
for seed in (0, 1):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False,
                 hi_abit_convs=())  # 기준
    base = rc.measure_ap(q, 'configs/coco_local.yaml', 640, dev)[0][0]; del q
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m = YOLOWorld(model)
    q = rc.build(YOLOWorld, model, coco, dev, calib, 'naive', w_bits=8, a_bits=8, skip_head=False,
                 hi_abit_convs=('23.cv3.0.2', '23.cv3.1.2', '23.cv3.2.2'))
    for n, mm in q.model.named_modules():
        if isinstance(mm, QuantConv2d) and mm.a_obs.bits != 16 and n.startswith('model.23.cv3') and n.endswith('.2'):
            print('경고: A16 미적용', n)
    print(f"{model} seed{seed}: naive W8A8 {base:.2f} -> cv3 마지막 conv 입력 A16 {rc.measure_ap(q, 'configs/coco_local.yaml', 640, dev)[0][0]:.2f}", flush=True)
    del q; torch.cuda.empty_cache()
