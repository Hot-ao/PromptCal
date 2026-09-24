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


# 09-23 (버그 수정): 항상 제외하는 하위 트리. `set_classes()`가 WorldModel.clip_model에
# CLIP(ViT-B/32) 전체를 캐싱해두는데, 그 vision tower의 patch-embed conv
# (clip_model.model.visual.conv1, 3->768 k32, weight 2,359,296)가 여기 재귀에 걸려
# QuantConv2d로 감싸지고 있었다. detector forward에서는 절대 호출되지 않는 모듈이라
#   - optimize_adaround가 매 run "1/53 conv 입력 미포착"으로 경고하며 건너뛰었고,
#   - quantized_weight_mib()에 2.25 MiB가 허수로 더해져 모델 크기가 22.8% 과대계상됐다
#     (12.13 MiB로 보고 -> 실제 detector는 9.88 MiB).
# 호출되지도 RNG를 소비하지도 않으므로 이 수정 전후 결과는 모델 크기를 빼면
# bit-identical이다(09-23 실측 확인). scripts/ 아래 호출처가 50곳 가까이 되므로
# 호출처마다 skip_names로 넘기는 대신 DFL과 같은 방식으로 여기서 막는다.
ALWAYS_SKIP_NAMES = frozenset({"clip_model"})


def wrap_convs(module: nn.Module, w_bits: int = 8, a_bits: int = 8,
               skip_names=None, skip_modules=None) -> int:
    """module 하위 Conv2d를 QuantConv2d로 교체. 교체 개수 반환. DFL과 clip_model은 스킵.
    skip_names: 이 이름의 하위 트리는 통째로 제외(예: {'projections'} — v1의 vision-text
    정렬 모듈. 민감해서 양자화 시 AP 대폭 하락 → baseline 공정성 위해 제외 가능).
    ALWAYS_SKIP_NAMES(=clip_model)는 skip_names와 무관하게 항상 제외된다(위 주석 참고)."""
    skip_names = set(skip_names or ()) | ALWAYS_SKIP_NAMES
    skip_modules = skip_modules or []
    if type(module).__name__ == "DFL":
        return 0
    count = 0
    for name, child in list(module.named_children()):
        if name in skip_names or any(child is sm for sm in skip_modules):
            continue                                   # 하위 트리 통째로 제외
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QuantConv2d(child, w_bits, a_bits))
            count += 1
        else:
            count += wrap_convs(child, w_bits, a_bits, skip_names, skip_modules)
    return count


def set_mode(module: nn.Module, calibrating: bool = False, quantized: bool = False):
    for m in module.modules():
        if isinstance(m, QuantConv2d):
            m.calibrating = calibrating
            m.quantized = quantized


@torch.no_grad()
def calibrate(model_module: nn.Module, calib_tensors, device: str = "cuda:0", act_observer: str = "minmax"):
    """calib_tensors: 전처리된 [1,3,H,W] 텐서들의 iterable."""
    for m in model_module.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.method = act_observer
    set_mode(model_module, calibrating=True, quantized=False)
    n = 0
    for t in calib_tensors:
        model_module(t.to(device))
        n += 1
    for m in model_module.modules():
        if isinstance(m, QuantConv2d):
            m.a_obs.freeze()
            m.freeze_weight_quant()
    set_mode(model_module, calibrating=False, quantized=True)
    return n
