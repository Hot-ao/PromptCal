"""
AdaRound (D-1): 학습 가능한 weight 반올림.

naive는 round(w/s)로 고정 반올림하지만, AdaRound는 floor(w/s) + h(alpha)로 두어
'올릴지/내릴지'를 layer 출력 재구성 오차로 학습한다. detector weight는 고정, 반올림
변수 alpha만 최적화. activation 양자화는 기존 ActObserver 재사용.

핵심 예측(우리 논문): AdaRound는 AP(reconstruction)를 개선하나 region-prompt 의사결정
(flip/substitution)은 여전히 못 지킨다 = "reconstruction ≠ decision preservation".
"""

from __future__ import annotations
import ctypes
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def free_cpu_mem():
    """glibc malloc_trim으로 free된 CPU 메모리를 OS에 반환.
    09-09 발견: optimize_adaround/optimize_brecq가 layer/block마다 256장짜리
    fp16 CPU 버퍼(fp_buf/q_buf, in_buf/out_buf)를 만들고 버리는 걸 반복하는데,
    Python 레퍼런스카운트로 즉시 해제돼도 glibc가 그 메모리를 arena에 들고
    있고 OS에 바로 안 돌려줘서 RSS가 파편화로 누적된다(실측: 단일 job이
    naive~qdrop 빌드 도중 RSS 68GB까지 상승 -- 그 시점엔 probe+calib+model
    몇 개뿐이라 논리적 필요량은 30GB 미만이어야 함). malloc_trim은 계산
    결과에 전혀 영향을 안 주는 순수 메모리 정리라 안전하다."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


GAMMA, ZETA = -0.1, 1.1   # rectified sigmoid 범위

# AdaRound/BRECQ 공식 구현 기본값 (Nagel et al. ICML'20 / Li et al. ICLR'21)
DEFAULT_LR = 1e-3          # alpha용 Adam lr (BRECQ layer/block_recon: Adam(opt_params) 기본값)
DEFAULT_REG_WEIGHT = 0.01  # BRECQ main_imagenet.py --weight 기본값
DEFAULT_WARMUP = 0.2       # 앞 20% iteration은 round_loss=0 (공식 warmup)
# activation step size용 lr (09-18 재검토). 처음엔 공식 BRECQ의 delta(scale 절댓값) lr을
# s_mult(1.0 기준 배율) 단위로 환산해서 썼는데, 인용한 소스값(4e-5)부터 틀렸다 -- 실제
# main_imagenet.py 실험은 --lr 4e-4(+iters_a 5000)를 쓴다. 이 모델에서 a_obs.scale 중앙값을
# 실측하면 0.0576(ImageNet post-ReLU6 가정 0.024의 2.4배)이라 4e-4/0.0576 ≈ 7e-3으로 환산되지만,
# 이 절대값 변환 자체가 불필요한 불확실성을 더한다는 지적이 맞다 -- s_mult는 delta 절대
# 스케일에 의존하지 않는 파라미터화라, Combined의 s_mult(promptcal.py, Adam lr=1e-2)와 정확히
# 맞추는 게 "손잡이 유무만 단일 변수" 비교에 더 방어 가능하다. 다만 scheduler 없이 1e-2만
# 쓰면(=Combined와 완전히 동일) 1000 iters 기준 실측(conv 2개, stem 3→32 / mid 64→64)에서
# 마지막 10% 구간에도 값이 계속 진동(각각 변동폭 0.0517/0.0052, 2e-3+cosine 대비 10배 이상)해서
# 수렴하지 않는다 -- CosineAnnealing은 유지하고 peak lr만 Combined와 맞춘다.
DEFAULT_ACT_LR = 1e-2


def h_alpha(alpha):
    return torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)


def lp_rec_loss(pred, target):
    """재구성 오차. BRECQ 공식 lp_loss(p=2, reduction='none')와 동일한 정규화:
    채널축은 sum, 나머지(batch/spatial)는 mean.

    09-18 수정: 이전에는 그냥 .pow(2).mean()이었는데, 그건 공식 대비 정확히
    C_out배 작다. reg_loss는 alpha 전 원소 sum이므로, 이 차이가 그대로
    "재구성 대비 rounding 정규화가 C_out/10배 과도"(reg_weight도 공식 0.01의
    1/10인 1e-3이었음)로 이어져서 alpha가 초기값(=round-to-nearest)에서 거의
    못 움직이고 정규화에 의해 0/1로 포화됐다. 합성 conv(Cin=128,Cout=256,3x3)
    재현 실험에서 nearest 대비 반올림 flip이 0.069%(수정 전) vs 0.797%(공식
    정규화), 출력 MSE 개선이 0.08% vs 0.85%로 약 10배 차이 -- 즉 수정 전
    AdaRound는 사실상 naive rounding과 같은 모델을 만들고 있었다."""
    d = (pred - target).abs().pow(2)
    return d.sum(1).mean() if d.dim() > 1 else d.mean()


def temp_decay(it, iters, warmup=DEFAULT_WARMUP, start_b=20.0, end_b=2.0):
    """AdaRound/BRECQ 공식 LinearTempDecay와 동일. warmup 구간에는 start_b(=20)로
    고정하고, 그 이후 end_b(=2)까지 선형 감소. beta가 크면 |2h-1|^beta의 gradient가
    거의 0이라 정규화 압력이 약하고, 작아질수록 h를 0/1로 밀어낸다."""
    start = warmup * iters
    if it < start:
        return start_b
    rel = (it - start) / max(iters - start, 1e-8)
    return end_b + (start_b - end_b) * max(0.0, 1.0 - rel)


class AdaRoundQuantConv2d(nn.Module):
    """기존 QuantConv2d를 AdaRound 반올림으로 확장."""
    def __init__(self, qconv, channelwise_smult=True):
        super().__init__()
        self.conv = qconv.conv               # 원본 Conv2d (weight 고정)
        self.w_bits = qconv.w_bits
        self.a_obs = qconv.a_obs             # 보정된 activation observer 재사용
        self.quantized = True
        self.soft = True                     # 최적화 중 soft, 추론 시 hard
        self.ste = False                     # PD-Quant end-to-end 시 activation STE
        # QDrop(Wei et al. ICLR'22): 재구성 중에만 >0. activation 양자화를 원소별
        # 확률 qdrop_prob로만 적용하고 나머지는 양자화 전 값을 그대로 흘린다
        # (공식 구현의 UniformAffineQuantizer.prob와 동일 -- prob = "양자화값을
        # 유지할 확률"). 추론 시에는 반드시 0.0이어야 한다.
        self.qdrop_prob = 0.0
        # 방향 C: learnable activation scale multiplier (초기 1.0, 연속 최적화용).
        # 09-07 재설계: conv당 스칼라 하나(구버전)는 class-selective 조정이
        # 구조적으로 불가능해서(§7.5~7.8) LVIS 등 calibration 밖 vocabulary에
        # 무차별 적용되는 전역 편향으로 수렴함이 확인됨(정규화로도 트레이드오프만
        # 이동, 해결 안 됨). in_channels별 벡터로 바꿔 채널마다 다른 조정이
        # 가능하게 함 -- text embedding이 채널마다 다른 가중치를 갖는다는 사실을
        # 활용해, 특정 채널만 조정하고 나머지는 안 건드리는 것이 원리적으로 가능해짐.
        # torch.ones(N)은 기본 CPU 텐서라 명시적으로 device를 맞춰야 함 -- 스칼라
        # 버전(torch.tensor(1.0))은 0-dim이라 cuda 텐서와 섞여도 PyTorch가 암묵적으로
        # 허용해줬지만(스칼라 특례), 벡터는 그 특례가 없어 device mismatch로 즉시 드러남.
        # 09-15 claim4 대조 실험: baseline(naive/AdaRound/QDrop/BRECQ)은 전부
        # activation을 per-tensor(ActObserver의 scale이 0-dim)로 양자화하는데
        # Combined만 s_mult로 per-channel 자유도를 쓰는 게 unisolated confound라는
        # 지적 -- 배포 시에도 표준 INT8 커널은 per-channel activation dequant를
        # 지원 안 해서 s_mult를 그대로 못 쓴다는 문제도 겹침. channelwise_smult=False로
        # 두면 s_mult를 다시 conv당 스칼라(0-dim)로 만들어 두 baseline과 동일한
        # granularity로 맞춰서 "차원(dimensionality) 자유도"만 격리해 비교할 수 있다.
        self.channelwise_smult = channelwise_smult
        if channelwise_smult:
            self.s_mult = nn.Parameter(torch.ones(self.conv.in_channels, device=self.conv.weight.device))
        else:
            self.s_mult = nn.Parameter(torch.tensor(1.0, device=self.conv.weight.device))
        self.use_smult = False               # True일 때만 s_mult 적용(scale 학습 모드)

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
        self._hard_weight_cache = None       # soft=False일 때 quant_weight() 캐시(아래 참고)

    def _apply(self, fn):
        """09-18 (사용자 지적) _hard_weight_cache는 buffer/parameter가 아닌 일반
        속성이라 nn.Module의 .to()/.half()/.cuda() 등이 자동으로 안 따라간다 --
        run_comparison.py의 switch_vocab()이 model.to("cpu") 후 다시 .to(device)를
        하는데, 그 사이에 forward가 한 번이라도 들어가면 device mismatch로
        터질 수 있다(지금은 안 터지지만 잠재 버그). fn을 캐시에도 적용해서
        나머지 파라미터/버퍼와 항상 같은 device/dtype을 유지한다."""
        super()._apply(fn)
        if self._hard_weight_cache is not None:
            self._hard_weight_cache = fn(self._hard_weight_cache)
        return self

    def quant_weight(self):
        """09-18 (사용자 지적) hard(soft=False) 경로 캐싱: alpha가 학습 중(soft=True)일
        때는 매번 새로 계산해야 하지만, 학습이 끝나 soft=False로 고정되면(Combined
        2단계 내내, 또는 iters=0이라 애초에 학습을 안 한 경우) h_alpha(alpha)와 그
        임계값 판정은 상수다 -- 그런데도 매 forward마다 weight 크기 그대로
        재계산되고 있었다(probe 5000장 + LVIS 4809장 전부). soft가 다시 True가 되면
        캐시를 무효화해서 정확성은 그대로 유지한다."""
        qmax = 2 ** (self.w_bits - 1) - 1
        if self.soft:
            self._hard_weight_cache = None
            w_int = self.w_floor + h_alpha(self.alpha)
            w_int = torch.clamp(w_int, -(qmax + 1), qmax)
            return w_int * self.w_scale
        if self._hard_weight_cache is None:
            w_int = self.w_floor + (h_alpha(self.alpha) >= 0.5).float()
            w_int = torch.clamp(w_int, -(qmax + 1), qmax)
            self._hard_weight_cache = w_int * self.w_scale
        return self._hard_weight_cache

    def quant_act(self, x):
        """activation 양자화 한 경로로 통합(use_smult > ste > plain 우선순위).
        qdrop_prob > 0이면 QDrop: 원소별 확률 qdrop_prob로만 양자화값을 쓰고
        나머지는 양자화 전 x를 그대로 흘린다."""
        if self.use_smult:
            xq = self._quantize_smult(x)
        elif self.ste:
            xq = self.a_obs.quantize_ste(x)
        else:
            xq = self.a_obs.quantize(x)
        if self.qdrop_prob > 0:
            keep = torch.rand_like(x) < self.qdrop_prob
            xq = torch.where(keep, xq, x)
        return xq

    def forward(self, x):
        if self.quantized and self.a_obs.ready:
            x = self.quant_act(x)
        wq = self.quant_weight()
        return F.conv2d(x, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)

    def _quantize_smult(self, x):
        """learnable scale multiplier로 activation 양자화. s_mult에 grad가 흐르도록,
        round만 STE로 통과시키고 scale 곱셈은 graph에 유지. s_mult가 in_channels별
        벡터이므로 [1,C,1,1]로 view해서 채널마다 다른 배율을 브로드캐스트한다."""
        qmin, qmax = 0, 2 ** self.a_obs.bits - 1
        mult = self.s_mult.clamp(min=0.1, max=10.0).view(1, -1, 1, 1)
        scale = self.a_obs.scale * mult
        zp = self.a_obs.zero_point
        x_s = x / scale
        # round를 STE로: forward=round, backward=identity (scale 경로는 유지)
        x_r = x_s + (torch.round(x_s) - x_s).detach()
        x_c = torch.clamp(x_r + zp, qmin, qmax)
        return (x_c - zp) * scale                     # scale이 graph에 남아 s_mult grad 흐름

    def reg_loss(self, beta, reduction="sum"):
        r = 1 - (2 * h_alpha(self.alpha) - 1).abs() ** beta
        return r.mean() if reduction == "mean" else r.sum()

    def flip_rate(self):
        """round-to-nearest 대비 반올림 결정이 바뀐 비율(%). AdaRound가 실제로
        뭔가 학습했는지 보는 진단 -- h→0/1 수렴률은 정규화만으로도 100%가 되므로
        (초기 h가 이미 소수부 = nearest 결정) 학습 여부를 구분하지 못한다."""
        with torch.no_grad():
            rest = (self.conv.weight.detach() / self.w_scale) - self.w_floor
            nearest = (rest >= 0.5)
            learned = (h_alpha(self.alpha) >= 0.5)
            return float((learned != nearest).float().mean()) * 100


def convert_to_adaround(model_module, channelwise_smult=True):
    """model 하위 QuantConv2d를 AdaRoundQuantConv2d로 교체(in-place). 교체 개수 반환.
    channelwise_smult=False면 s_mult을 per-tensor 스칼라로 생성(claim4 대조 실험용,
    §AdaRoundQuantConv2d 주석 참고)."""
    from .fake_quant import QuantConv2d
    count = 0
    for name, child in list(model_module.named_children()):
        if isinstance(child, QuantConv2d):
            setattr(model_module, name, AdaRoundQuantConv2d(child, channelwise_smult=channelwise_smult))
            count += 1
        else:
            count += convert_to_adaround(child, channelwise_smult=channelwise_smult)
    return count


def list_adaround_convs(model_module):
    return [m for m in model_module.modules() if isinstance(m, AdaRoundQuantConv2d)]


def optimize_adaround(quant_module, fp_module, calib_tensors, device,
                      iters=2000, lr=DEFAULT_LR, reg_weight=DEFAULT_REG_WEIGHT, batch=1,
                      qdrop_prob=0.0, verbose=True, learn_act_scale=False,
                      act_lr=DEFAULT_ACT_LR, warmup=DEFAULT_WARMUP):
    """
    layer-wise AdaRound 최적화(순차/누적 오차 반영, 09-06 수정). 각 conv를
    네트워크 순서대로 처리하며:
      target = fp_conv(fp_input)      (이상적 목표 -- 항상 순수 FP forward)
      pred   = adaround_conv(q_input) (q_input은 quant_module 자체의 현재 상태로
                                        흘린 실제 입력 -- 이미 처리된 앞쪽 layer는
                                        hardened, 아직 처리 안 된 뒤쪽은 soft/init)
      loss   = lp_rec_loss(pred, target) + reg_weight * reg(alpha)

    이전 버전은 target과 pred 둘 다 fp_module에서 캡처한 입력(xin)을 그대로
    썼다 -- 즉 모든 layer가 "앞쪽이 전부 완벽한 FP"라고 가정하고 독립적으로
    재구성됐고, 앞선 layer들의 실제 양자화 오차가 뒤쪽 layer 학습에 전혀
    반영되지 않았다(AdaRound/BRECQ 원 절차의 핵심인 누적 오차 보상이 빠짐).
    이번 버전은 pred의 입력만 quant_module 쪽에서 별도로 다시 캡처해서 이
    누적 오차를 반영한다. target 기준은 그대로 FP다(asymmetric reconstruction).

    qdrop_prob > 0 이면 QDrop: q_input 기준으로 원소별 확률 qdrop_prob로만
    양자화(나머지는 q_input 그대로, 즉 "이 layer만 양자화 안 했다면"의 반사실)
    해서 activation 양자화 노이즈에 강건한 반올림을 학습.
    추론(AdaRoundQuantConv2d.forward)은 항상 양자화 → drop 없음.
    주의: QDrop 원 논문은 block-wise 재구성 위에서 drop을 건다. 논문에 충실한
    QDrop은 brecq.optimize_brecq(qdrop_prob=...)를 쓸 것. 여기 qdrop_prob은
    "layer-wise + drop" ablation용으로 남겨둔다.

    주의: 계산 더 무거움. layer마다 FP forward 1회 + quant forward 1회
    (calib 이미지 수만큼)가 추가로 필요.

    09-18 공식 구현 정합성 수정:
      - 재구성 손실을 lp_rec_loss(BRECQ lp_loss p=2와 동일 정규화)로 교체.
        기존 .pow(2).mean()은 공식 대비 C_out배 작아서 정규화가 압도했다.
      - lr 1e-2 → 1e-3, reg_weight 1e-3 → 0.01 (둘 다 공식 기본값).
      - warmup 도입: 앞 20% iteration은 round_loss를 아예 끈다(공식 동작).
        beta annealing도 warmup 이후부터 20→2로 감소(LinearTempDecay와 동일).
      - 배치 샘플링을 it % n(결정적 순환)에서 무작위 추출로 교체하고,
        선언만 되고 쓰이지 않던 batch 인자를 실제로 쓴다(공식은 batch=32이지만
        detection activation은 메모리가 커서 기본값은 1로 둔다).
      - learn_act_scale의 s_mult를 alpha와 같은 optimizer/lr에 묶지 않고
        별도 Adam(act_lr) + CosineAnnealingLR로 분리(공식 BRECQ와 동일 구조).

    learn_act_scale (09-17, claim: "AdaRound/QDrop/BRECQ엔 activation scale을
    학습하는 손잡이가 원 논문과 달리 아예 빠져있다" 대응): 원 BRECQ 논문(Li et al.,
    ICLR 2021)은 weight rounding(AdaRound)과 activation step size(LSQ, Esser et
    al. 2020)를 같은 reconstruction 목적함수로 공동 최적화한다. 우리 구현은
    지금까지 activation scale을 calibration 시점 min-max로 고정해두고 alpha만
    학습했다 -- 원 논문 대비 축소 구현이었다. True면 이미 AdaRoundQuantConv2d에
    있는 s_mult/_quantize_smult(PromptCal-C 방향에서 쓰던 것과 동일 메커니즘,
    STE로 scale까지 grad가 흐름)를 켜서 alpha와 s_mult를 같은 MSE loss로 동시
    최적화한다 -- 새 메커니즘이 아니라 기존 s_mult를 baseline 쪽에도 재사용하는
    것. 호출 전에 convert_to_adaround(channelwise_smult=False)로 만들어야
    Combined(per-tensor 확정 설계, claim4)와 granularity가 맞아 "손잡이 유무"만
    단일 변수로 비교된다. 기본 False(기존 동작과 완전히 동일).
    (공식 BRECQ는 weight rounding → activation scale 2-pass인데 여기는 1-pass
     공동 최적화다. 이 차이는 남아있다.)
    """
    # quant/fp 트리를 나란히 순회해 위치가 정확히 대응되는 (AdaRound conv, fp Conv2d) 짝 생성.
    # wrap이 같은 위치의 nn.Conv2d를 AdaRound로 교체했으므로 두 트리 구조가 동일 → 1:1 매칭.
    # named_children() 순회 순서가 forward 실행 순서와 (대체로) 일치한다고 가정 --
    # 분기/concat이 있는 블록에서는 근사치이지만, "전부 완벽한 FP" 가정보다는 낫다.
    def pairs(qm, fm):
        for (qn, qc), (fn, fc) in zip(qm.named_children(), fm.named_children()):
            if isinstance(qc, AdaRoundQuantConv2d):
                yield qc, fc          # fc: 대응 nn.Conv2d
            else:
                yield from pairs(qc, fc)

    conv_pairs = list(pairs(quant_module, fp_module))
    if verbose:
        print(f"[adaround] {len(conv_pairs)} conv 순차 최적화 시작 (누적 오차 반영, 트리 나란히 매칭)")

    calib_list = list(calib_tensors)
    n_skipped = 0
    flips = []

    for i, (ac, fc) in enumerate(conv_pairs):
        # (1) 이 layer의 FP 입력 -- target 계산 기준(이상적, 변경 없음)
        fp_buf = []
        hh_fp = fc.register_forward_hook(
            lambda m, inp, out: fp_buf.append(inp[0].detach().half().cpu()))
        with torch.no_grad():
            for t in calib_list:
                fp_module(t.to(device))
        hh_fp.remove()

        # (2) 이 layer로 실제 흘러들어오는 입력 -- quant_module 현재 상태(앞쪽
        #     layer는 이미 hardened, 뒤쪽은 아직 처리 전 soft/init)로 forward해서
        #     캡처. 누적 오차가 여기 반영된다.
        q_buf = []
        hh_q = ac.register_forward_hook(
            lambda m, inp, out: q_buf.append(inp[0].detach().half().cpu()))
        with torch.no_grad():
            for t in calib_list:
                quant_module(t.to(device))
        hh_q.remove()

        n = min(len(fp_buf), len(q_buf))
        if n == 0:
            ac.soft = False
            n_skipped += 1
            if verbose:
                print(f"  [{i+1}/{len(conv_pairs)}] 입력 미포착 스킵 (hard 유지)")
            del fp_buf, q_buf
            free_cpu_mem()
            continue
        fp_w = ac.conv.weight.detach()
        stride, pad = ac.conv.stride, ac.conv.padding
        dil, grp = ac.conv.dilation, ac.conv.groups
        bias = ac.conv.bias

        ac.soft = True
        opt = torch.optim.Adam([ac.alpha], lr=lr)
        opt_a = sched_a = None
        if learn_act_scale:
            ac.use_smult = True
            opt_a = torch.optim.Adam([ac.s_mult], lr=act_lr)
            sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=iters, eta_min=0.0)
        nb = max(1, min(batch, n))
        for it in range(iters):
            # 공식 구현과 동일하게 매 iteration 무작위 추출(it % n은 결정적 순환이라
            # beta annealing 스케줄과 위상이 고정되는 편향이 생긴다).
            sel = torch.randint(0, n, (nb,)).tolist()
            fp_in = torch.cat([fp_buf[k] for k in sel]).to(device).float()
            q_in = torch.cat([q_buf[k] for k in sel]).to(device).float()
            with torch.no_grad():
                target = F.conv2d(fp_in, fp_w, bias, stride, pad, dil, grp)
            opt.zero_grad()
            if opt_a is not None:
                opt_a.zero_grad()
            if learn_act_scale:
                # s_mult에 grad가 흘러야 하므로 no_grad 밖에서 _quantize_smult 호출
                # (STE라 round 자체는 여전히 backward에서 identity로 통과).
                xq = ac._quantize_smult(q_in)
            else:
                with torch.no_grad():
                    xq = ac.a_obs.quantize(q_in) if ac.a_obs.ready else q_in
            if qdrop_prob > 0:
                # QDrop: 원소별 확률 qdrop_prob로 양자화, 나머지는 q_in 그대로.
                # learn_act_scale일 때도 xq의 grad 경로(s_mult)는 torch.where로 보존됨.
                with torch.no_grad():
                    m = (torch.rand_like(q_in) < qdrop_prob)
                xq = torch.where(m, xq, q_in)
            wq = ac.quant_weight()
            pred = F.conv2d(xq, wq, bias, stride, pad, dil, grp)
            loss = lp_rec_loss(pred, target)
            if it >= warmup * iters:        # 공식 warmup: 앞 구간은 round_loss=0
                loss = loss + reg_weight * ac.reg_loss(temp_decay(it, iters, warmup))
            loss.backward()
            opt.step()
            if opt_a is not None:
                opt_a.step()
                sched_a.step()

        ac.soft = False   # 확정(hard) -- 다음 layer의 quant_module capture에 반영됨
        fl = ac.flip_rate()
        flips.append(fl)
        if verbose:
            h = h_alpha(ac.alpha).detach()
            conv01 = float(((h < 0.05) | (h > 0.95)).float().mean()) * 100
            print(f"  [{i+1}/{len(conv_pairs)}] layer opt done, "
                  f"h→0/1 수렴 {conv01:.0f}%, nearest 대비 flip {fl:.3f}%")
        del fp_buf, q_buf
        free_cpu_mem()

    if n_skipped:
        # verbose=False(run_comparison 기본)여도 조용히 넘어가면 안 되는 정보.
        print(f"[adaround][warn] {n_skipped}/{len(conv_pairs)} conv 입력 미포착으로 학습 없이 hard 확정됨")
    if flips:
        mean_flip = sum(flips) / len(flips)
        print(f"[adaround] 완료 (hard 반올림). nearest 대비 평균 flip {mean_flip:.3f}% "
              f"(max {max(flips):.3f}%) -- 0에 가까우면 naive rounding과 사실상 동일")
