"""m naive W8A8에서 head cv3 마지막 1x1 conv(3개)에만 입력 채널 scale 이전을 적용해, A8 그대로 seed 불안정성이 사라지는지 확인.
이 conv들의 생산자는 cv3[l][1] 하나이고 소비자도 이 conv 하나라, conv별 독립 s가 곧 배포 제약을 만족한다."""
import sys, glob, torch
sys.path.insert(0, 'pipeline')
import run_comparison as rc
from quant.quant_model import wrap_convs, calibrate
from quant.fake_quant import QuantConv2d, quantize_weight_per_channel
from quant.migrate import _capture_inputs, _act_quant_mse, _conv
from ultralytics import YOLOWorld
dev = f"cuda:{sys.argv[1]}"; model = sys.argv[2]
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
coco = rc.load_names('coco'); calib = [rc.preprocess(p, 640, dev).to(dev) for p in sorted(glob.glob('/data/taeho/coco_datasets/train2017/*.jpg'))[:256]]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

@torch.no_grad()
def migrate_last(m):
    head = m.model.model[-1]
    targets = [lvl[-1] for lvl in head.cv3]
    for qc in m.model.modules():
        if isinstance(qc, QuantConv2d): qc.calibrating, qc.quantized = False, False
    for li, qc in enumerate(targets):
        xs = _capture_inputs(m.model, qc, calib[:8], dev)
        W = qc.conv.weight.detach().float()
        xmax = torch.stack([x.abs().amax(dim=(0, 2, 3)) for x in xs]).amax(0).float().clamp(min=1e-5)
        wmax = W.abs().amax(dim=(0, 2, 3)).clamp(min=1e-8)
        def err(s):
            Wq = quantize_weight_per_channel(W if s is None else W * s.view(1, -1, 1, 1), qc.w_bits); e = r = 0.0
            for x in xs:
                x = x.float(); ref = _conv(qc, x, W); xi = x if s is None else x / s.view(1, -1, 1, 1)
                yq = _conv(qc, _act_quant_mse(xi, qc.a_obs.bits), Wq); e += float((torch.nn.functional.normalize(yq, dim=1) - torch.nn.functional.normalize(ref, dim=1)).pow(2).sum(1).mean()); r += 1.0
            return e / r
        best = (err(None), None, None)
        for a in ALPHAS:
            s = xmax.pow(a) / wmax.pow(1 - a); s = s / s.log().mean().exp(); e = err(s)
            if e < best[0]: best = (e, a, s)
        if best[2] is not None: qc.apply_migration(best[2])
        xr = torch.stack([x.abs().amax(dim=(0, 2, 3)) for x in xs]).amax(0)
        print(f"  cv3[{li}] 마지막 conv: a={best[1]}  입력 채널 최대/중앙 {float(xr.max() / xr.median()):.1f}배  방향오차 {best[0]:.5f}", flush=True)
        del xs

for seed in (0, 1, 2):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m = YOLOWorld(model); m.set_classes(coco); m.fuse(); wrap_convs(m.model, 8, 8); m.model.to(dev).eval()
    migrate_last(m)
    calibrate(m.model, calib, device=dev, act_observer='mse')
    print(f"{model} seed{seed}: naive W8A8 + cv3 마지막 conv 이전(방향 오차 기준, A8 유지) COCO AP {rc.measure_ap(m, 'configs/coco_local.yaml', 640, dev)[0][0]:.2f}", flush=True)
    del m; torch.cuda.empty_cache()
