"""naive W8A8 / W4A8(첫·마지막 8bit) s-World seed0: attention 양자화 수준별 COCO AP(구현 정상 여부 확인)."""
import sys, glob, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); calib = [rc.preprocess(p, 640, dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
for wb, ab, fl in ((8, 8, 0), (4, 8, 8)):
    for aq in ("none", "attn", "attn_cls"):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        q = rc.build(YOLOWorld, 'yolov8s-world.pt', coco, dev, calib, 'naive', w_bits=wb, a_bits=ab, skip_head=False,
                     first_last_bits=fl, attn_quant=aq)
        ap = rc.measure_ap(q, 'configs/coco_local.yaml', 640, dev); ap = ap[0] if isinstance(ap[0], tuple) else ap
        print(f"naive W{wb}A{ab} attn={aq}: COCO AP {ap[0]:.2f}", flush=True)
        del q; torch.cuda.empty_cache()
