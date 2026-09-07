"""
Fake-quantization 모듈 (naive min-max). int8 실행이 아니라, quantize->dequantize로
정밀도만 int8로 떨어뜨려 "양자화가 유발할 손상"을 시뮬레이션한다(측정 목적).

- Weight: per-output-channel symmetric int8. 가중치는 고정이라 정적으로 scale 계산.
- Activation: per-tensor asymmetric int8. calibration 이미지로 min/max 수집 후 freeze.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_weight_per_channel(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """per-output-channel(dim0) symmetric int8 fake-quant."""
    qmax = 2 ** (bits - 1) - 1            # 127
    dims = list(range(1, w.dim()))
    amax = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    wq = torch.clamp(torch.round(w / scale), -(qmax + 1), qmax) * scale
    return wq


class ActObserver(nn.Module):
    """activation per-tensor asymmetric int8 관측/양자화기."""
    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.ready = False

    @torch.no_grad()
    def observe(self, x: torch.Tensor):
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        qmin, qmax = 0, 2 ** self.bits - 1
        mn = torch.minimum(self.min_val, torch.zeros_like(self.min_val))
        mx = torch.maximum(self.max_val, torch.zeros_like(self.max_val))
        scale = ((mx - mn) / (qmax - qmin)).clamp(min=1e-8)
        zp = torch.round(qmin - mn / scale)
        self.scale = scale
        self.zero_point = zp
        self.ready = True

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        qmin, qmax = 0, 2 ** self.bits - 1
        xq = torch.clamp(torch.round(x / self.scale) + self.zero_point, qmin, qmax)
        return (xq - self.zero_point) * self.scale

    def quantize_ste(self, x: torch.Tensor) -> torch.Tensor:
        """STE 버전: forward는 양자화값, backward는 항등(gradient 통과).
        PD-Quant end-to-end 최적화에서 cv4→초기 layer로 grad를 흘리기 위함."""
        qmin, qmax = 0, 2 ** self.bits - 1
        xq = torch.clamp(torch.round(x / self.scale) + self.zero_point, qmin, qmax)
        xdq = (xq - self.zero_point) * self.scale
        return x + (xdq - x).detach()


class QuantConv2d(nn.Module):
    """기존 Conv2d를 감싸 weight+input activation을 fake-quant한다."""
    def __init__(self, conv: nn.Conv2d, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        self.conv = conv
        self.w_bits = w_bits
        self.a_obs = ActObserver(a_bits)
        self.calibrating = False
        self.quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.a_obs.observe(x)
            xq = x                                   # 관측만, 통과
        elif self.quantized and self.a_obs.ready:
            xq = self.a_obs.quantize(x)
        else:
            xq = x
        w = self.conv.weight
        wq = quantize_weight_per_channel(w, self.w_bits) if self.quantized else w
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)
