"""
모델 래핑 / 캘리브레이션 유틸.

- wrap_convs: vision 경로의 모든 Conv2d를 QuantConv2d로 교체(in-place).
  DFL(고정 가중치)은 건너뜀. text encoder는 model.model에 conv로 존재하지 않으므로
  자동 제외(offline 임베딩).
- calibrate: calibration 이미지로 activation min/max 수집 후 freeze, quantized 모드로 전환.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .fake_quant import QuantConv2d


def wrap_convs(module: nn.Module, w_bits: int = 8, a_bits: int = 8,
               skip_names=None) -> int:
    """module 하위 Conv2d를 QuantConv2d로 교체. 교체 개수 반환. DFL은 스킵.
    skip_names: 이 이름의 하위 트리는 통째로 제외(예: {'projections'} — v1의 vision-text
    정렬 모듈. 민감해서 양자화 시 AP 대폭 하락 → baseline 공정성 위해 제외 가능)."""
    skip_names = skip_names or set()
    if type(module).__name__ == "DFL":
        return 0
    count = 0
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue                                   # 하위 트리 통째로 제외
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d(child, w_bits, a_bits))
            count += 1
        else:
            count += wrap_convs(child, w_bits, a_bits, skip_names)
    return count


def set_mode(module: nn.Module, calibrating: bool = False, quantized: bool = False):
    for m in module.modules():
        if isinstance(m, QuantConv2d):
            m.calibrating = calibrating
            m.quantized = quantized


@torch.no_grad()
def calibrate(model_module: nn.Module, calib_tensors, device: str = "cuda:0"):
    """calib_tensors: 전처리된 [1,3,H,W] 텐서들의 iterable."""
    set_mode(model_module, calibrating=True, quantized=False)
    n = 0
    for t in calib_tensors:
        model_module(t.to(device))
        n += 1
    for m in model_module.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.freeze()
    set_mode(model_module, calibrating=False, quantized=True)
    return n
