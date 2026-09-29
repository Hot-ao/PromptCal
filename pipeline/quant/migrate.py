"""
입력 채널별 scale 이전(migration) 탐색 (09-29).

근거: runs/128 W4 민감도 진단에서 W4 손상이 C2f concat을 받는 1x1 conv(12.cv2 등)에 몰려 있고,
그 conv들이 입력 채널별 weight 크기 편차가 가장 큰 conv와 겹친다(12.cv2 13.3배). 출력 채널별
weight 양자화는 입력 채널 방향 편차를 흡수하지 못하므로, 편차를 activation 쪽(A8은 여유가 있음)으로
옮긴다.

  conv(x, W) = conv(x / s, W * diag(s))       (FP로는 항등)
  양자화:     conv(Q_a(x / s), Q_w(W * diag(s)))   Q_a: per-tensor, Q_w: 출력 채널별(기존 그대로)

s는 SmoothQuant 형태 s_j = max|x_j|^a / max|W_:j|^(1-a) 로 두고, conv마다 a를 grid search로
고른다(AWQ처럼 그 conv 출력의 재구성 오차 최소). 후보에는 항상 "이전 없음(s=1)"을 포함해 어떤
conv도 기존보다 나빠지지 않게 한다. 입력은 FP 모델 기준(앞 conv들의 이전은 FP에서 항등이라
순서와 무관).

주의(배포): 한 텐서를 여러 conv가 소비하면(C2f split 등) 비용 0 배포에는 그 소비자들이 같은 s를
공유해야 한다. 이 1차 구현은 conv마다 독립 s -- 공유 제약이 없는 상한이다.

무작위성을 쓰지 않는다(표본 추출은 stride) -- 전역 RNG 스트림 불변.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .fake_quant import QuantConv2d, quantize_weight_per_channel


@torch.no_grad()
def _act_quant_mse(x: torch.Tensor, bits: int, p: float = 2.4, n_cand: int = 80, max_elems: int = 2_000_000):
    """ActObserver(method='mse')와 같은 탐색을 RNG 없이(stride 표본) 수행하고 x를 fake-quant."""
    v = x.detach().flatten().float()
    if v.numel() > max_elems:
        v = v[:: v.numel() // max_elems + 1]
    x_min = torch.minimum(v.min(), torch.zeros((), device=v.device))
    x_max = torch.maximum(v.max(), torch.zeros((), device=v.device))
    qmax = 2 ** bits - 1
    best, rng = None, (x_min, x_max)
    for i in range(n_cand):
        f = 1.0 - i * 0.01
        mn, mx = x_min * f, x_max * f
        sc = ((mx - mn) / qmax).clamp(min=1e-8)
        zp = torch.round(-mn / sc)
        vq = (torch.clamp(torch.round(v / sc) + zp, 0, qmax) - zp) * sc
        score = (vq - v).abs().pow(p).mean()
        if best is None or score < best:
            best, rng = score, (mn, mx)
    mn, mx = rng
    sc = ((mx - mn) / qmax).clamp(min=1e-8)
    zp = torch.round(-mn / sc)
    return (torch.clamp(torch.round(x / sc) + zp, 0, qmax) - zp) * sc


def _conv(c, x, w):
    k = c.conv
    return F.conv2d(x, w, k.bias, k.stride, k.padding, k.dilation, k.groups)


@torch.no_grad()
def _capture_inputs(model_module, qc, images, device):
    buf = []
    h = qc.register_forward_pre_hook(lambda _m, inp: buf.append(inp[0].detach()))
    for t in images:
        model_module(t.to(device))
    h.remove()
    return buf


@torch.no_grad()
def search_and_apply(model_module, images, device, alphas, w_bits_of=None, verbose=True):
    """model_module 하위 QuantConv2d(groups==1)마다 a를 고르고 apply_migration. calibrate() 전 호출.
    반환: [(이름, 고른 a 또는 None, 이전 없음 대비 오차 비율)]."""
    named = [(n, m) for n, m in model_module.named_modules()
             if isinstance(m, QuantConv2d) and m.conv.groups == 1]
    for _, m in named:
        m.calibrating, m.quantized = False, False          # FP forward로 입력 수집
    report = []
    for name, qc in named:
        xs = _capture_inputs(model_module, qc, images, device)
        W = qc.conv.weight.detach().float()
        xmax = torch.stack([x.abs().amax(dim=(0, 2, 3)) for x in xs]).amax(0).float().clamp(min=1e-5)
        wmax = W.abs().amax(dim=(0, 2, 3)).clamp(min=1e-8)
        wb, ab = qc.w_bits, qc.a_obs.bits

        def err(s):
            Wq = quantize_weight_per_channel(W if s is None else W * s.view(1, -1, 1, 1), wb)
            e = 0.0
            for x in xs:
                ref = _conv(qc, x.float(), W)
                xi = x.float() if s is None else x.float() / s.view(1, -1, 1, 1)
                e += float((_conv(qc, _act_quant_mse(xi, ab), Wq) - ref).pow(2).mean())
            return e

        base = err(None)
        best_a, best_e, best_s = None, base, None
        for a in alphas:
            s = xmax.pow(a) / wmax.pow(1.0 - a)
            s = s / s.log().mean().exp()                     # 기하평균 1로 정규화(수치 안정용, 결과 불변)
            e = err(s)
            if e < best_e:
                best_a, best_e, best_s = a, e, s
        if best_s is not None:
            qc.apply_migration(best_s)
        ratio = best_e / max(base, 1e-20)
        report.append((name, best_a, ratio))
        if verbose:
            print(f"  [mig] {name:32s} a={'-' if best_a is None else f'{best_a:.2f}':>5}  "
                  f"err {base:.3e} -> {best_e:.3e} (x{ratio:.3f})", flush=True)
        del xs
    return report


@torch.no_grad()
def select_hi_cols(model_module, images, device, frac: float, verbose=True):
    """09-29: 채널 단위 혼합 정밀도. W4 conv(groups==1)마다 입력 채널 점수
    score_j = max|W_:j| * max|x_j| 상위 ceil(frac*Cin)개 열을 hi_bits(8)로 지정한다.
    근거(12.cv2 분석): C2fAttn concat의 attention 분기 소수 채널이 activation과 weight 양쪽에서 동시에
    커서 scale 이전으로는 못 옮기고, 상위 2개 열만 W8로 둬도 출력 오차가 11배 준다.
    migration 뒤에 부르면 이전된 W*s와 x/s 기준으로 점수를 매긴다. calibrate() 전 호출. RNG 미사용."""
    import math
    named = [(n, m) for n, m in model_module.named_modules()
             if isinstance(m, QuantConv2d) and m.conv.groups == 1 and m.w_bits < m.hi_bits]
    for _, m in named:
        m.calibrating, m.quantized = False, False
    n_hi_total = n_total = 0
    for name, qc in named:
        xs = _capture_inputs(model_module, qc, images, device)
        if qc.mig is not None:
            xs = [x / qc.mig for x in xs]
        xmax = torch.stack([x.abs().amax(dim=(0, 2, 3)) for x in xs]).amax(0).float()
        wmax = qc.conv.weight.detach().float().abs().amax(dim=(0, 2, 3))
        score = xmax * wmax
        ci = score.numel()
        k = max(1, math.ceil(frac * ci))
        mask = torch.zeros(ci, dtype=torch.bool, device=score.device)
        mask[score.argsort(descending=True)[:k]] = True
        qc.set_hi_cols(mask)
        n_hi_total += k; n_total += ci
        if verbose:
            top = score.sort(descending=True).values
            print(f"  [hi-col] {name:32s} {k:>3}/{ci} 열 W{qc.hi_bits}  "
                  f"점수 최대/중앙 {float(top[0] / top.median().clamp(min=1e-12)):6.1f}", flush=True)
        del xs
    if verbose:
        print(f"  [hi-col] 합계 {n_hi_total}/{n_total} 입력 채널")
