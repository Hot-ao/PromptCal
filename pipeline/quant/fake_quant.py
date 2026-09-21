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
    def __init__(self, bits: int = 8, method: str = "minmax"):
        super().__init__()
        self.bits = bits
        self.method = method                 # "minmax" | "mse"
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self._mse_min_sum = 0.0
        self._mse_max_sum = 0.0
        self._mse_n = 0
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.ready = False

    @torch.no_grad()
    def _mse_range(self, x: torch.Tensor, p: float = 2.4, n_cand: int = 80, max_elems: int = 2_000_000):
        """공식 BRECQ scale_method='mse' 탐색: [min,max]를 (1-0.01*i)배로 줄여가며
        L_p(p=2.4) 양자화 오차가 최소인 범위를 고른다(i=0..79)."""
        x = x.detach().flatten().float()
        if x.numel() > max_elems:
            x = x[:: x.numel() // max_elems + 1]
        x_min = torch.minimum(x.min(), torch.zeros((), device=x.device))
        x_max = torch.maximum(x.max(), torch.zeros((), device=x.device))
        qmax = 2 ** self.bits - 1
        best, best_rng = None, (x_min, x_max)
        for i in range(n_cand):
            f = 1.0 - i * 0.01
            mn, mx = x_min * f, x_max * f
            scale = ((mx - mn) / qmax).clamp(min=1e-8)
            zp = torch.round(-mn / scale)
            xq = (torch.clamp(torch.round(x / scale) + zp, 0, qmax) - zp) * scale
            score = (xq - x).abs().pow(p).mean()
            if best is None or score < best:
                best, best_rng = score, (mn, mx)
        return best_rng

    @torch.no_grad()
    def observe(self, x: torch.Tensor):
        if self.method == "mse":
            # 공식은 큰 배치 하나에서 탐색하지만 메모리상 이미지별로 탐색 후 평균(AvgMSEObserver 방식)
            mn, mx = self._mse_range(x)
            self._mse_min_sum = self._mse_min_sum + mn
            self._mse_max_sum = self._mse_max_sum + mx
            self._mse_n += 1
            return
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        if self.method == "mse" and self._mse_n > 0:
            self.min_val = torch.as_tensor(self._mse_min_sum / self._mse_n)
            self.max_val = torch.as_tensor(self._mse_max_sum / self._mse_n)
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
