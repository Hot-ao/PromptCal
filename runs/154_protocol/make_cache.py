"""확정 후보 프로토콜(A16 + attn_cls)의 보호 진단 캐시를 미리 만든다(run_comparison과 같은 경로·인자)."""
import sys, os, json
sys.path.insert(0, 'pipeline')
from diag_w4_sensitivity import rank_convs_leakfree
model, dev = sys.argv[1], sys.argv[2]
out = f"configs/protect_cache/{os.path.splitext(os.path.basename(model))[0]}_n200_c256_img640_la16_aqattn_cls.json"
rows = rank_convs_leakfree(model, '/data/taeho/coco_datasets', dev, n_eval=200, n_calib=256, imgsz=640,
                           first_last_bits=8, last_abits=16, attn_quant='attn_cls')
json.dump(rows, open(out, 'w'), indent=1); print("저장", out, len(rows))
