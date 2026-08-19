"""
AdaRound (D-1): 학습 가능한 weight 반올림.

naive는 round(w/s)로 고정 반올림하지만, AdaRound는 floor(w/s) + h(alpha)로 두어
'올릴지/내릴지'를 layer 출력 재구성 오차로 학습한다. detector weight는 고정, 반올림
변수 alpha만 최적화. activation 양자화는 기존 ActObserver 재사용.

핵심 예측(우리 논문): AdaRound는 AP(reconstruction)를 개선하나 region-prompt 의사결정
(flip/substitution)은 여전히 못 지킨다 = "reconstruction ≠ decision preservation".
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA, ZETA = -0.1, 1.1   # rectified sigmoid 범위


def h_alpha(alpha):
    return torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)


class AdaRoundQuantConv2d(nn.Module):
    """기존 QuantConv2d를 AdaRound 반올림으로 확장."""
    def __init__(self, qconv):
        super().__init__()
        self.conv = qconv.conv               # 원본 Conv2d (weight 고정)
        self.w_bits = qconv.w_bits
        self.a_obs = qconv.a_obs             # 보정된 activation observer 재사용
        self.quantized = True
        self.soft = True                     # 최적화 중 soft, 추론 시 hard
        self.ste = False                     # PD-Quant end-to-end 시 activation STE

        w = self.conv.weight.detach()
        qmax = 2 ** (self.w_bits - 1) - 1
        dims = list(range(1, w.dim()))
        amax = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
        self.register_buffer("w_scale", amax / qmax)
        self.register_buffer("w_floor", torch.floor(w / self.w_scale))
        # alpha 초기화: h(alpha) ≈ 소수부(초기 soft = 원래 weight)
        rest = (w / self.w_scale) - self.w_floor
        p = ((rest - GAMMA) / (ZETA - GAMMA)).clamp(1e-4, 1 - 1e-4)
        self.alpha = nn.Parameter(-torch.log((1 - p) / p))

    def quant_weight(self):
        qmax = 2 ** (self.w_bits - 1) - 1
        if self.soft:
            w_int = self.w_floor + h_alpha(self.alpha)
        else:
            w_int = self.w_floor + (h_alpha(self.alpha) >= 0.5).float()
        w_int = torch.clamp(w_int, -(qmax + 1), qmax)
        return w_int * self.w_scale

    def forward(self, x):
        if self.quantized and self.a_obs.ready:
            x = self.a_obs.quantize_ste(x) if self.ste else self.a_obs.quantize(x)
        wq = self.quant_weight()
        return F.conv2d(x, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

    def reg_loss(self, beta, reduction="sum"):
        r = 1 - (2 * h_alpha(self.alpha) - 1).abs() ** beta
        return r.mean() if reduction == "mean" else r.sum()


def convert_to_adaround(model_module):
    """model 하위 QuantConv2d를 AdaRoundQuantConv2d로 교체(in-place). 교체 개수 반환."""
    from .fake_quant import QuantConv2d
    count = 0
    for name, child in list(model_module.named_children()):
        if isinstance(child, QuantConv2d):
            setattr(model_module, name, AdaRoundQuantConv2d(child))
            count += 1
        else:
            count += convert_to_adaround(child)
    return count


def list_adaround_convs(model_module):
    return [m for m in model_module.modules() if isinstance(m, AdaRoundQuantConv2d)]


@torch.no_grad()
def _capture_layer_input(fp_module, target_conv_weight_id, calib_tensors, device, fp_convs, idx):
    """fp 모델에서 idx번째 conv의 입력을 캡처(streaming). fp16 CPU 캐시로 반환."""
    buf = []
    fc = fp_convs[idx]
    h = fc.register_forward_hook(lambda m, inp, out: buf.append(inp[0].detach().half().cpu()))
    for t in calib_tensors:
        fp_module(t.to(device))
        # buf에 이번 이미지 입력 1개 쌓임
    h.remove()
    return buf  # list of [1,C,H,W] fp16 cpu


def optimize_adaround(quant_module, fp_module, calib_tensors, device,
                      iters=2000, lr=1e-2, reg_weight=1e-3, batch=1,
                      qdrop_prob=0.0, verbose=True):
    """
    layer-wise AdaRound 최적화. 각 AdaRound conv를 독립적으로 최적화.
      target = fp_conv(fp_input)  (FP 출력)
      pred   = adaround_conv(act_input)  (양자화 weight)
      loss   = MSE(pred, target) + reg_weight * reg(alpha)

    qdrop_prob > 0 이면 QDrop: 최적화 중 activation을 원소별로 확률 qdrop_prob로만
    양자화(나머지는 FP)해서 activation 양자화 노이즈에 강건한 반올림을 학습.
    추론(AdaRoundQuantConv2d.forward)은 항상 양자화 → drop 없음.

    주의: 계산 무거움. calib 이미지 수 x layer 수만큼 fp forward.
    """
    import torch.nn as nn

    # quant/fp 트리를 나란히 순회해 위치가 정확히 대응되는 (AdaRound conv, fp Conv2d) 짝 생성.
    # wrap이 같은 위치의 nn.Conv2d를 AdaRound로 교체했으므로 두 트리 구조가 동일 → 1:1 매칭.
    def pairs(qm, fm):
        for (qn, qc), (fn, fc) in zip(qm.named_children(), fm.named_children()):
            if isinstance(qc, AdaRoundQuantConv2d):
                yield qc, fc          # fc: 대응 nn.Conv2d
            else:
                yield from pairs(qc, fc)

    conv_pairs = list(pairs(quant_module, fp_module))
    if verbose:
        print(f"[adaround] {len(conv_pairs)} conv 최적화 시작 (트리 나란히 매칭)")

    calib_list = list(calib_tensors)

    for i, (ac, fc) in enumerate(conv_pairs):
        # 대응 fp conv(fc)에 hook을 걸어 이 layer의 fp 입력을 streaming 캐시
        buf = []
        hh = fc.register_forward_hook(
            lambda m, inp, out: buf.append(inp[0].detach().half().cpu()))
        with torch.no_grad():
            for t in calib_list:
                fp_module(t.to(device))
        hh.remove()

        inputs = buf
        if len(inputs) == 0:
            ac.soft = False
            if verbose:
                print(f"  [{i+1}/{len(conv_pairs)}] 입력 미포착 스킵 (hard 유지)")
            continue
        fp_w = ac.conv.weight.detach()
        stride, pad = ac.conv.stride, ac.conv.padding
        dil, grp = ac.conv.dilation, ac.conv.groups
        bias = ac.conv.bias

        ac.soft = True
        opt = torch.optim.Adam([ac.alpha], lr=lr)
        for it in range(iters):
            xin = inputs[it % len(inputs)].to(device).float()
            with torch.no_grad():
                target = F.conv2d(xin, fp_w, bias, stride, pad, dil, grp)
                xq = ac.a_obs.quantize(xin) if ac.a_obs.ready else xin
                if qdrop_prob > 0:
                    # QDrop: 원소별 확률 qdrop_prob로 양자화, 나머지는 FP
                    m = (torch.rand_like(xin) < qdrop_prob)
                    xq = torch.where(m, xq, xin)
            opt.zero_grad()
            wq = ac.quant_weight()
            pred = F.conv2d(xq, wq, bias, stride, pad, dil, grp)
            beta = max(2.0, 20.0 * (1 - it / iters))   # annealing
            loss = (pred - target).pow(2).mean() + reg_weight * ac.reg_loss(beta)
            loss.backward()
            opt.step()

        ac.soft = False   # 추론은 hard 반올림
        if verbose:
            h = h_alpha(ac.alpha).detach()
            conv01 = float(((h < 0.05) | (h > 0.95)).float().mean()) * 100
            print(f"  [{i+1}/{len(conv_pairs)}] layer opt done, "
                  f"h→0/1 수렴 {conv01:.0f}%")

    if verbose:
        print("[adaround] 전체 layer 최적화 완료 (hard 반올림 모드)")
