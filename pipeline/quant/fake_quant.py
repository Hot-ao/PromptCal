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


@torch.no_grad()
def mse_weight_scale_asym_channelwise(w: torch.Tensor, bits: int = 8, n_cand: int = 80, p: float = 2.4):
    """공식 BRECQ/QDrop UniformAffineQuantizer(scale_method='mse', channel_wise=True)와 동일:
    dim0(출력 채널)별로 x_min/x_max를 (1-0.01*i)배로 줄여가며(0-클램프 없음, 공식 그대로)
    L_p(p=2.4) 양자화 오차가 최소인 (delta, zero_point)를 찾는다. i=0..n_cand-1 전 채널을
    한 번에 벡터화(각 후보 i에 대해 전 채널을 동시에 채점 -- 채널마다 파이썬 루프 안 돔).
    반환: (delta[Cout,1,1,1], zero_point[Cout,1,1,1]) -- 비대칭, [0, 2**bits-1] 정수 범위 기준."""
    dims = list(range(1, w.dim()))
    x_min = w.amin(dim=dims, keepdim=True)
    x_max = w.amax(dim=dims, keepdim=True)
    n_levels = 2 ** bits - 1
    best_score = best_delta = best_zp = None
    for i in range(n_cand):
        f = 1.0 - i * 0.01
        new_min, new_max = x_min * f, x_max * f
        delta = ((new_max - new_min) / n_levels).clamp(min=1e-8)
        zp = torch.round(-new_min / delta)
        w_int = torch.clamp(torch.round(w / delta) + zp, 0, n_levels)
        w_q = (w_int - zp) * delta
        score = (w_q - w).abs().pow(p).mean(dim=dims, keepdim=True)
        if best_score is None:
            best_score, best_delta, best_zp = score, delta, zp
        else:
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best_delta = torch.where(better, delta, best_delta)
            best_zp = torch.where(better, zp, best_zp)
    return best_delta, best_zp


@torch.no_grad()
def mse_weight_scale_symmetric_perlayer(w: torch.Tensor, bits: int = 8, n_cand: int = 80, p: float = 2.4):
    """AdaRound 원 논문(Nagel et al. ICML'20) Table 6 본문 설정: 대칭, 레이어 전체 스칼라
    scale s를 ‖W - W̄‖_F^2 최소화로 결정(공식 BRECQ 탐색과 통일해 L_p(2.4) grid search로 근사).
    반환: scale(스칼라 텐서, w와 broadcast 가능).

    09-21 확인: 이 모델(YOLOv8s-world)에는 채널 간 weight 최대값이 최대 12배(head 제외)~64배
    (head 포함) 차이 나는 conv가 있어서, 레이어 전체 스칼라 하나로는 정식 스케일(calib=256,
    iters=1000)에서도 COCO_AP -1.62, Heval_flip 5.3%->21.5%로 실측 손상이 크다(`runs/89_weight_mse`).
    그래서 기본 채택은 아래 mse_weight_scale_symmetric_channelwise(논문 Table 7 각주 "일부는
    더 유리한 per-channel을 썼다"가 이미 허용하는 변형)로 바뀌었다 -- 이 함수는 순수 논문
    본문 재현이 필요한 ablation용으로 남겨둔다."""
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().max()
    best_score = best_scale = None
    for i in range(n_cand):
        cand_amax = amax * (1.0 - i * 0.01)
        scale = (cand_amax / qmax).clamp(min=1e-8)
        w_int = torch.clamp(torch.round(w / scale), -(qmax + 1), qmax)
        w_q = w_int * scale
        score = (w_q - w).abs().pow(p).mean()
        if best_score is None or score < best_score:
            best_score, best_scale = score, scale
    return best_scale


@torch.no_grad()
def mse_weight_scale_symmetric_channelwise(w: torch.Tensor, bits: int = 8, n_cand: int = 80, p: float = 2.4):
    """09-21 확정 기본값(AdaRound 모드): 대칭이지만 채널별(dim0) scale -- 원 논문 Table 7
    각주("일부 결과는 더 유리한 per-channel quantization을 썼다")가 허용하는 변형.
    위 mse_weight_scale_symmetric_perlayer와 탐색 방식은 동일(대칭, L_p 2.4), 채널마다
    독립적으로 최적 scale을 고르는 것만 다르다(BRECQ/QDrop의 채널별 비대칭 탐색과
    granularity를 맞춰서 "대칭 vs 비대칭"만 AdaRound/BRECQ 간 단일 변수로 남긴다).
    반환: scale[Cout,1,1,1] (w와 broadcast 가능)."""
    dims = list(range(1, w.dim()))
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(dim=dims, keepdim=True)
    best_score = best_scale = None
    for i in range(n_cand):
        cand_amax = amax * (1.0 - i * 0.01)
        scale = (cand_amax / qmax).clamp(min=1e-8)
        w_int = torch.clamp(torch.round(w / scale), -(qmax + 1), qmax)
        w_q = w_int * scale
        score = (w_q - w).abs().pow(p).mean(dim=dims, keepdim=True)
        if best_score is None:
            best_score, best_scale = score, scale
        else:
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best_scale = torch.where(better, scale, best_scale)
    return best_scale


def quantize_weight_per_channel(w: torch.Tensor, bits: int = 8, method: str = "mse") -> torch.Tensor:
    """per-output-channel(dim0) int8 fake-quant. method='mse'(기본, BRECQ/QDrop 공식과 동일
    channel-wise 비대칭 MSE 탐색) 또는 'maxabs_sym'(이전 동작 -- 대칭, max-abs, ablation용).
    주의: 'mse'는 80-후보 탐색이라 무겁다 -- forward마다 부르지 말 것(QuantConv2d는
    freeze_weight_quant()로 1회 계산해 캐싱한다)."""
    if method == "maxabs_sym":
        qmax = 2 ** (bits - 1) - 1
        dims = list(range(1, w.dim()))
        amax = w.abs().amax(dim=dims, keepdim=True).clamp(min=1e-8)
        scale = amax / qmax
        return torch.clamp(torch.round(w / scale), -(qmax + 1), qmax) * scale
    delta, zp = mse_weight_scale_asym_channelwise(w, bits)
    n_levels = 2 ** bits - 1
    w_int = torch.clamp(torch.round(w / delta) + zp, 0, n_levels)
    return (w_int - zp) * delta


class ActObserver(nn.Module):
    """activation per-tensor asymmetric int8 관측/양자화기."""
    def __init__(self, bits: int = 8, method: str = "minmax"):
        super().__init__()
        self.bits = bits
        self.method = method                 # "minmax" | "mse"
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self._mse_buf = None                 # method="mse": calibration 전체에서 모은 표본(1회 탐색용)
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
            # 09-21 수정: 이전엔 이미지마다 80-후보 탐색을 따로 돌려 평균(calib=256이면
            # 탐색 256회, 게다가 이미지별 결과 평균은 공식 semantics도 아님) -- 공식
            # MSEObserver는 calibration 배치 전체를 모아 탐색을 "한 번"만 한다(BRECQ
            # main_imagenet.py: model(cali_data[:256].cuda()) 한 번의 forward로 관측).
            # 여기서는 표본만 모아두고(상한 max_elems), freeze()에서 1회 탐색한다.
            v = x.detach().flatten().float()
            if v.numel() > 200_000:          # 레이어당 표본 상한(메모리/속도), 무작위 부분추출
                idx = torch.randint(0, v.numel(), (200_000,), device=v.device)
                v = v[idx]
            self._mse_buf = v if self._mse_buf is None else torch.cat([self._mse_buf, v])
            if self._mse_buf.numel() > 2_000_000:
                idx = torch.randint(0, self._mse_buf.numel(), (2_000_000,), device=self._mse_buf.device)
                self._mse_buf = self._mse_buf[idx]
            return
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        if self.method == "mse" and self._mse_buf is not None:
            mn, mx = self._mse_range(self._mse_buf)
            self.min_val, self.max_val = torch.as_tensor(mn), torch.as_tensor(mx)
            self._mse_buf = None
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
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.register_buffer("w_zero_point", torch.tensor(0.0))
        self._w_quant_ready = False

    @torch.no_grad()
    def freeze_weight_quant(self):
        """09-21: naive/Combined도 BRECQ/QDrop 공식과 같은 채널별 비대칭 MSE 출발점을
        쓰도록(claim17 이후 weight-scale 정합성 수정) 1회만 탐색해 캐싱한다 -- 80-후보
        탐색을 매 forward(probe 5000장+LVIS 4809장)마다 반복하면 감당 못 할 만큼 느려진다."""
        self.w_scale, self.w_zero_point = mse_weight_scale_asym_channelwise(
            self.conv.weight.detach(), self.w_bits)
        self._w_quant_ready = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self.a_obs.observe(x)
            xq = x                                   # 관측만, 통과
        elif self.quantized and self.a_obs.ready:
            xq = self.a_obs.quantize(x)
        else:
            xq = x
        w = self.conv.weight
        if self.quantized:
            if not self._w_quant_ready:
                self.freeze_weight_quant()
            n_levels = 2 ** self.w_bits - 1
            w_int = torch.clamp(torch.round(w / self.w_scale) + self.w_zero_point, 0, n_levels)
            wq = (w_int - self.w_zero_point) * self.w_scale
        else:
            wq = w
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)
