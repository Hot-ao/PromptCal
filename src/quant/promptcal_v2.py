"""
PromptCal-PTQ v2 — Decision-Preserving Scale Calibration.

기존(v1, promptcal.py)의 두 실패 축을 둘 다 버린다:
  1) 최적화 대상 alpha(weight rounding) — soft→hard 전환에서 '안 굳어서' held-out을 악화.
  2) 목적함수 margin(연속 값의 MSE) — margin은 줄어도 argmax 결정은 안 지켜짐(의심 C).

대신:
  1) 최적화 대상 = activation quant scale (연속, LSQ식 학습 가능 step). 굳을 필요가 없다.
     weight rounding은 round-to-nearest로 고정(학습 안 함).
  2) 목적함수 = held-out fallback '결정(argmax)'을 직접 보존하는 분류 손실.
     측정 지표(heldout_flip)가 보는 것과 '같은 양'을 최적화 → 의심 C를 정면으로 없앤다.

기존 코드는 건드리지 않는다(가산 모듈). QuantConv2d/ActObserver/_CV4Capture/_find_head/
SimilarityHarness/wrap_convs/calibrate를 그대로 재사용한다.

핵심 신규:
  - LearnableActScale: ActObserver의 frozen scale에 곱하는 학습 가능한 배수(로그 파라미터).
    LSQ식 gradient(round는 STE, scale에는 quant residual이 흐름)로 scale에 grad가 끊기지 않음.
  - enable_scale_training / disable_scale_training: 학습 forward에서만 STE 경로 사용.
  - decision_loss: FP의 held-out fallback top-1을 pseudo-label로 quant가 그 결정을 맞추게.
  - optimize_promptcal_v2: cv4 유사도 행렬에서 결정 손실을 계산, activation scale만 업데이트.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import QuantConv2d, ActObserver
from .pdquant import _find_head, _CV4Capture


# ---------------------------------------------------------------------------
# 1) 학습 가능한 activation scale (LSQ식). ActObserver에 곱셈 배수로 얹는다.
# ---------------------------------------------------------------------------
def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """forward=round, backward=identity."""
    return x + (torch.round(x) - x).detach()


class LearnableActScale(nn.Module):
    """
    frozen ActObserver(min/max로 잡은 scale·zero_point)를 감싸, scale에 배수 m을 학습.
    s_eff = obs.scale * exp(log_m),  m 초기값 1.0 (log_m=0 → 시작은 naive와 동일).

    학습 forward는 LSQ식으로 구현: x_hat = (round_ste(clamp(x/s_eff)+zp) - zp) * s_eff.
    round만 STE로 통과시키므로 s_eff(=m)에는 quant residual gradient가 그대로 흐른다.
    (v1 방향 C에서 'scale이 detach돼 grad가 끊기던' 버그를 구조적으로 차단.)

    zero_point는 freeze 시점 값으로 고정한다. m은 1.0 근방의 미세 조정이라 zp 재계산 없이
    유효하다(ablation에서 zp 동반 조정은 별도 검토).
    """
    def __init__(self, obs: ActObserver):
        super().__init__()
        self.obs = obs
        self.log_m = nn.Parameter(torch.zeros(()))   # exp(0)=1 → 시작은 naive scale
        self.qmin = 0
        self.qmax = 2 ** obs.bits - 1

    @property
    def s_eff(self) -> torch.Tensor:
        return self.obs.scale * self.log_m.exp()

    def quantize_train(self, x: torch.Tensor) -> torch.Tensor:
        s = self.s_eff
        zp = self.obs.zero_point
        x_s = x / s + zp
        x_clamp = torch.clamp(x_s, self.qmin, self.qmax)
        x_q = _round_ste(x_clamp)
        return (x_q - zp) * s

    @torch.no_grad()
    def quantize_hard(self, x: torch.Tensor) -> torch.Tensor:
        """추론용: 학습된 s_eff로 hard 양자화(grad 불필요). 학습한 배수가 추론에 반영됨."""
        s = self.s_eff
        zp = self.obs.zero_point
        x_q = torch.clamp(torch.round(x / s) + zp, self.qmin, self.qmax)
        return (x_q - zp) * s


class LearnableScaleQuantConv2d(QuantConv2d):
    """
    QuantConv2d를 상속. 평소(quantized)에는 부모와 동일(naive int8 추론)하지만,
    train_scale=True일 때 activation을 LearnableActScale의 STE 경로로 양자화해
    scale(m)에 gradient가 흐르게 한다. weight는 부모의 round-to-nearest 그대로(학습 안 함).
    """
    def __init__(self, qconv: QuantConv2d):
        # qconv의 상태(conv, a_obs, bits, quantized)를 물려받아 감싼다.
        nn.Module.__init__(self)
        self.conv = qconv.conv
        self.w_bits = qconv.w_bits
        self.a_obs = qconv.a_obs
        self.calibrating = qconv.calibrating
        self.quantized = qconv.quantized
        self.train_scale = False
        self.act_scale = LearnableActScale(self.a_obs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .fake_quant import quantize_weight_per_channel
        if self.calibrating:
            self.a_obs.observe(x)
            xq = x
        elif self.train_scale and self.a_obs.ready:
            xq = self.act_scale.quantize_train(x)          # STE, grad→scale
        elif self.quantized and self.a_obs.ready:
            xq = self.act_scale.quantize_hard(x)            # 학습된 s_eff로 추론(naive 아님)
        else:
            xq = x
        w = self.conv.weight
        wq = quantize_weight_per_channel(w, self.w_bits) if self.quantized else w
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


def convert_to_learnable_scale(model_module: nn.Module) -> int:
    """이미 wrap_convs+calibrate된 모델의 QuantConv2d를 LearnableScaleQuantConv2d로 교체."""
    count = 0
    for name, child in list(model_module.named_children()):
        if isinstance(child, QuantConv2d) and not isinstance(child, LearnableScaleQuantConv2d):
            setattr(model_module, name, LearnableScaleQuantConv2d(child))
            count += 1
        else:
            count += convert_to_learnable_scale(child)
    return count


def list_scale_convs(model_module):
    return [m for m in model_module.modules() if isinstance(m, LearnableScaleQuantConv2d)]


def enable_scale_training(model_module):
    for m in list_scale_convs(model_module):
        m.train_scale = True


def disable_scale_training(model_module):
    for m in list_scale_convs(model_module):
        m.train_scale = False


# ---------------------------------------------------------------------------
# 2) 결정 보존 목적함수 — held-out fallback argmax를 직접 겨냥.
# ---------------------------------------------------------------------------
def decision_loss(sim_q, sim_fp, removed_cols, keep_cols, conf_thres=0.25,
                  temp=1.0, margin_gate=0.0, return_count=False):
    """
    heldout_flip이 측정하는 것과 '같은 양'을 최적화한다.

    측정(heldout_flip): FP full-top1이 H_eval에 속한 confident anchor에 대해,
      H_eval을 제거한 뒤 남은 프롬프트 중 argmax가 FP↔quant 일치하는가.
    학습(여기): H_eval 대신 H_cal(removed_cols)을 '제거할 held-out'으로 삼아 동일 구조로.
      - anchor 선택: FP가 removed_cols 안의 프롬프트를 top-1으로 confident하게 고른 region.
      - 남은 집합 keep_cols(= S; H_eval은 loss에서 완전히 배제) 안에서 FP의 top-1을
        pseudo-label로 두고, quant가 그 top-1을 맞추도록 cross-entropy.

    → margin(값)이 아니라 argmax(결정)를 직접 맞추므로 '의심 C'를 정면 대응.

    sim_*        : [A, P]  pre-sigmoid 유사도(cv4). sim_q는 grad 유지, sim_fp는 no-grad.
    removed_cols : LongTensor. 학습에서 '제거'할 held-out 프롬프트(H_cal).
    keep_cols    : LongTensor. 결정을 겨루는 남은 프롬프트 집합(S). H_eval 미포함.
    conf_thres   : FP confident anchor 기준(sigmoid max).
    temp         : cross-entropy 온도. 작을수록 argmax를 더 뾰족하게 겨냥.
    margin_gate  : FP fallback margin이 이 값보다 큰 anchor만 사용(pseudo-label 잡음 억제).
                   0이면 게이트 없음.
    return_count : True면 (loss, 사용된 anchor 수)를 반환. anchor 0이면 빈 스텝 판별용.
    """
    def _ret(loss, n):
        return (loss, n) if return_count else loss

    # 1) '제거할 held-out'을 top-1으로 원하는 confident anchor 고르기 (FP 기준)
    prob = sim_fp.sigmoid()
    maxp, top1_full = prob.max(dim=-1)
    conf = maxp > conf_thres
    removed_mask = torch.zeros(sim_fp.shape[-1], dtype=torch.bool, device=sim_fp.device)
    removed_mask[removed_cols] = True
    want_removed = conf & removed_mask[top1_full]
    if want_removed.sum() == 0:
        return _ret(sim_q.sum() * 0.0, 0)            # grad 0, 빈 스텝

    aidx = want_removed.nonzero(as_tuple=True)[0]

    # 2) 남은 집합(keep_cols)만 뽑아 FP fallback 결정 = pseudo-label
    fp_keep = sim_fp[aidx][:, keep_cols]             # [n, |keep|]
    q_keep = sim_q[aidx][:, keep_cols]               # [n, |keep|] grad 유지
    y_fp = fp_keep.argmax(dim=-1)                    # FP fallback top-1 (pseudo-label)

    # 3) (선택) fallback margin 게이트 — FP top1-top2 간격이 큰 anchor만
    if margin_gate > 0:
        top2 = fp_keep.topk(min(2, fp_keep.shape[-1]), dim=-1).values
        fp_margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
        sel = fp_margin > margin_gate
        if sel.sum() == 0:
            return _ret(sim_q.sum() * 0.0, 0)
        q_keep, y_fp = q_keep[sel], y_fp[sel]

    # 4) quant가 그 fallback top-1을 고르도록 cross-entropy (결정 직접 보존)
    return _ret(F.cross_entropy(q_keep / temp, y_fp), int(y_fp.numel()))


# ---------------------------------------------------------------------------
# 3) 최적화 루프 — activation scale만 업데이트.
# ---------------------------------------------------------------------------
def _assemble_sim_from_cap(cap):
    parts = []
    for i in sorted(cap.buf):
        B, P, H, W = cap.buf[i].shape
        parts.append(cap.buf[i].reshape(B, P, H * W))
    return torch.cat(parts, dim=2)[0].transpose(0, 1)   # [A, P]


def optimize_promptcal_v2(quant_model, fp_model, calib_tensors, device,
                          removed_cols, keep_cols, iters=150, lr=3e-3,
                          conf_thres=0.25, temp=1.0, margin_gate=0.0,
                          train_convs=None, verbose=True):
    """
    결정 보존 목적함수로 activation scale(m)만 최적화.

    removed_cols : H_cal (학습에서 제거할 held-out). keep_cols : S (겨루는 남은 집합).
    ★ H_eval은 여기 어디에도 넣지 않는다(정직성). 최종 측정은 20_*의 heldout_flip에서만.

    train_convs : 학습(scale 업데이트)할 LearnableScaleQuantConv2d 부분집합. None이면 전체.
      · 지정 시 그 conv들의 log_m만 최적화. 나머지 conv는 STE로 gradient는 통과시키되
        log_m=0(=자기 calibration scale)로 '동결'. → head를 정밀도로 보호하고 학습에서
        빼면서, backbone/neck에만 PromptCal을 적용하는 실험에 사용.
      · 전체 conv가 STE 경로여야 gradient가 backbone까지 흐르므로 enable은 전체에 한다.

    안정화(초기 실험에서 단일 이미지 스텝이 노이즈 랜덤워크로 held-out을 악화시킨 문제 대응):
      - full-batch 누적: 매 스텝 calib 전체를 돌며 gradient를 모아 한 번 업데이트(빈 스텝 제거).
      - best-checkpoint: averaged decision loss가 최저였던 scale을 저장했다 마지막에 복원.
      - anchor 수/평균 loss를 로깅해 '목적함수가 실제로 최소화되는가'를 판정 가능하게.

    반환: (best_avg_loss, moved_log_m). quant_model in-place, 추론은 naive 경로로 되돌림.
    """
    scale_convs = list_scale_convs(quant_model)
    if len(scale_convs) == 0:
        raise RuntimeError("LearnableScaleQuantConv2d 없음. convert_to_learnable_scale 먼저 호출.")
    enable_scale_training(quant_model)   # 전체 STE → grad가 backbone까지 흐름

    # FP cv4 유사도 타깃 캐시 (grad 불필요)
    fp_head = _find_head(fp_model)
    fp_cap = _CV4Capture(fp_head)
    fp_sims = []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear(); fp_model(t.to(device))
            fp_sims.append(_assemble_sim_from_cap(fp_cap).detach())
    fp_cap.close()

    q_head = _find_head(quant_model)
    q_cap = _CV4Capture(q_head)
    if train_convs is None:
        train_set = scale_convs
    else:
        ids = {id(x) for x in train_convs}
        train_set = [c for c in scale_convs if id(c) in ids]
        if len(train_set) == 0:
            raise RuntimeError("train_convs가 scale conv와 하나도 안 겹침 — 대상 확인.")
    params = [c.act_scale.log_m for c in train_set]
    m_before = torch.tensor([p.detach().item() for p in params])
    opt = torch.optim.Adam(params, lr=lr)

    removed_cols = torch.as_tensor(removed_cols, device=device)
    keep_cols = torch.as_tensor(keep_cols, device=device)

    if verbose:
        print(f"[promptcal_v2] scale conv {len(scale_convs)}개 중 학습 {len(train_set)}개, "
              f"removed(H_cal) {len(removed_cols)}, keep(S) {len(keep_cols)}, "
              f"목적=decision CE(temp={temp}, gate={margin_gate}) | full-batch, lr={lr}")

    n = len(calib_tensors)
    # best 기준선: m=1(naive) 상태의 training CE를 먼저 측정 → best-checkpoint가
    # naive보다 나쁜 scale은 남기지 않도록 보장(안전장치).
    with torch.no_grad():
        base_loss, base_anchor = 0.0, 0
        for j in range(n):
            q_cap.clear(); quant_model(calib_tensors[j].to(device))
            sim_q = _assemble_sim_from_cap(q_cap)
            lj, cj = decision_loss(sim_q, fp_sims[j], removed_cols, keep_cols,
                                   conf_thres=conf_thres, temp=temp,
                                   margin_gate=margin_gate, return_count=True)
            if cj > 0:
                base_loss += float(lj) * cj; base_anchor += cj
    best_loss = base_loss / max(base_anchor, 1)
    best_state = [p.detach().clone() for p in params]
    if verbose:
        print(f"  [init] naive(m=1) avg_decision_CE={best_loss:.4f} (best 기준선)")

    for step in range(iters):
        opt.zero_grad()
        tot_loss, tot_anchor, used_imgs = 0.0, 0, 0
        for j in range(n):
            t = calib_tensors[j].to(device)
            sim_fp = fp_sims[j]
            q_cap.clear()
            quant_model(t)
            sim_q = _assemble_sim_from_cap(q_cap)
            loss_j, cnt = decision_loss(sim_q, sim_fp, removed_cols, keep_cols,
                                        conf_thres=conf_thres, temp=temp,
                                        margin_gate=margin_gate, return_count=True)
            if cnt > 0:
                (loss_j / n).backward()               # 누적(평균 gradient)
                tot_loss += float(loss_j.detach()) * cnt
                tot_anchor += cnt; used_imgs += 1
        if used_imgs == 0:
            continue                                   # 이 스텝은 신호 없음
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        avg = tot_loss / max(tot_anchor, 1)            # anchor 가중 평균 CE
        if avg < best_loss:
            best_loss = avg
            best_state = [p.detach().clone() for p in params]

        if verbose and (step + 1) % max(1, iters // 10) == 0:
            m_now = torch.tensor([p.detach().exp().item() for p in params])
            print(f"  [{step+1}/{iters}] avg_decision_CE={avg:.4f} (best {best_loss:.4f}) "
                  f"anchors/step≈{tot_anchor} imgs={used_imgs}/{n} "
                  f"m∈[{m_now.min():.3f},{m_now.max():.3f}]")

    # best-checkpoint 복원 (노이즈 드리프트로 더 나빠진 scale 버림)
    with torch.no_grad():
        for p, b in zip(params, best_state):
            p.copy_(b)

    q_cap.close()
    disable_scale_training(quant_model)
    m_after = torch.tensor([p.detach().item() for p in params])
    moved = (m_after - m_before).abs().sum().item()
    if verbose:
        print(f"[promptcal_v2] 완료 | best avg CE={best_loss:.4f}, "
              f"복원된 log_m 총 이동량={moved:.3f} (best 기준)")
    return best_loss, moved
