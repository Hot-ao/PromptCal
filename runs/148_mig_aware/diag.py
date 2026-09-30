"""09-30 (A안 싼 검증): 이전(M)을 적용한 모델에서 보호 진단을 다시 하면 보호 목록이 바뀌는가.
기준 모델: W8·A{a}(첫/마지막 8bit), 이전 없음 / 공유 제약 이전 적용. conv 하나씩 weight만 W4 -> COCO flip 증가량.
누수 없음(train2017, COCO 어휘만). GPU를 본 실험과 함께 쓰므로 이 프로세스 메모리를 상한으로 묶는다."""
import sys, glob, os, json, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
import diag_w4_sensitivity as dg
from quant.pdquant import _find_head
from quant.vocab_metric import VocabMetric, encode_text_bank
from ultralytics import YOLOWorld
gpu, cap_gb, n_eval = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
dev = f'cuda:{gpu}'
torch.cuda.set_per_process_memory_fraction(cap_gb * 1024**3 / torch.cuda.get_device_properties(dev).total_memory, dev)
torch.manual_seed(0); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
root = '/data/taeho/coco_datasets'; coco = rc.load_names('coco')
paths = sorted(glob.glob(f'{root}/train2017/*.jpg'))
calib = [rc.preprocess(p, 640, dev) for p in paths[:256]]
evals = [rc.preprocess(p, 640, 'cpu') for p in paths[:n_eval]]
fp = rc.build(YOLOWorld, 'yolov8s-world.pt', coco, dev, calib, 'fp')
dg.HS = dg.HeadSim(_find_head(fp.model))
vc = VocabMetric(encode_text_bank(YOLOWorld, 'yolov8s-world.pt', coco, dev))
refs = dg.reference(fp, evals, dev, vc, vc, 0.25, 'coco')
D = 'runs/148_mig_aware'
for a in (8, 6, 5):
    for mig in (False, True):
        tag = f"A{a}_{'mig' if mig else 'nomig'}"
        q = rc.build(YOLOWorld, 'yolov8s-world.pt', coco, dev, calib, 'naive', fp=fp, w_bits=8, a_bits=a,
                     skip_head=False, first_last_bits=8,
                     mig_alphas=(0, .25, .5, .75, 1) if mig else (), mig_tied=mig)
        groups = dg.groups_of(q.model, set(range(len(q.model.model))))
        base = dg.measure(q, evals, refs, dev, vc, vc)
        rows = []
        for name, convs in groups:
            dg.set_wbits(convs, 4); m = dg.measure(q, evals, refs, dev, vc, vc); dg.set_wbits(convs, 8)
            rows.append(dict(name=name, mparam=sum(c.conv.weight.numel() for c in convs) / 1e6,
                             **{k: m[k] - base[k] for k in ('emb', 'coco', 'cls')}))
        json.dump(rows, open(f'{D}/rank_{tag}.json', 'w'), indent=1)
        sel, used, tot = dg.select_protected(rows, 0.015, 'coco')
        top = sorted(rows, key=lambda r: -r['coco'])[:8]
        print(f"[{tag}] base COCO flip {base['coco']:.2f}%  선택 {sel} ({used:.3f}M)", flush=True)
        print("   상위 8: " + ", ".join(f"{r['name'].replace('.conv','')}({r['coco']:+.1f})" for r in top), flush=True)
        del q; torch.cuda.empty_cache()
