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


def h_alpha(alpha):
    return torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)


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
            if self.use_smult:
                x = self._quantize_smult(x)          # 방향 C: learnable scale로 STE 양자화
            elif self.ste:
                x = self.a_obs.quantize_ste(x)
            else:
                x = self.a_obs.quantize(x)
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
    layer-wise AdaRound 최적화(순차/누적 오차 반영, 09-06 수정). 각 conv를
    네트워크 순서대로 처리하며:
      target = fp_conv(fp_input)      (이상적 목표 -- 항상 순수 FP forward)
      pred   = adaround_conv(q_input) (q_input은 quant_module 자체의 현재 상태로
                                        흘린 실제 입력 -- 이미 처리된 앞쪽 layer는
                                        hardened, 아직 처리 안 된 뒤쪽은 soft/init)
      loss   = MSE(pred, target) + reg_weight * reg(alpha)

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

    주의: 계산 더 무거움. layer마다 FP forward 1회 + quant forward 1회
    (calib 이미지 수만큼)가 추가로 필요.
    """
    import torch.nn as nn

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
            j = it % n
            fp_in = fp_buf[j].to(device).float()
            q_in = q_buf[j].to(device).float()
            with torch.no_grad():
                target = F.conv2d(fp_in, fp_w, bias, stride, pad, dil, grp)
                xq = ac.a_obs.quantize(q_in) if ac.a_obs.ready else q_in
                if qdrop_prob > 0:
                    # QDrop: 원소별 확률 qdrop_prob로 양자화, 나머지는 q_in 그대로
                    m = (torch.rand_like(q_in) < qdrop_prob)
                    xq = torch.where(m, xq, q_in)
            opt.zero_grad()
            wq = ac.quant_weight()
            pred = F.conv2d(xq, wq, bias, stride, pad, dil, grp)
            beta = max(2.0, 20.0 * (1 - it / iters))   # annealing
            loss = (pred - target).pow(2).mean() + reg_weight * ac.reg_loss(beta)
            loss.backward()
            opt.step()

        ac.soft = False   # 확정(hard) -- 다음 layer의 quant_module capture에 반영됨
        if verbose:
            h = h_alpha(ac.alpha).detach()
            conv01 = float(((h < 0.05) | (h > 0.95)).float().mean()) * 100
            print(f"  [{i+1}/{len(conv_pairs)}] layer opt done, "
                  f"h→0/1 수렴 {conv01:.0f}%")
        del fp_buf, q_buf
        free_cpu_mem()

    if verbose:
        print("[adaround] 전체 layer 최적화 완료 (hard 반올림 모드)")
