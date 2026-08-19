"""
PD-Quant (D-4): Prediction Difference 기반 PTQ.

기존 reconstruction 계열(AdaRound/QDrop)이 layer 출력을 local하게 맞추는 것과 달리,
PD-Quant는 '양자화 전후 최종 예측의 차이'를 global하게 최소화한다. 우리 detector
맥락에서 '예측' = cv4 region-prompt 유사도 행렬로 정의(옵션 A).

즉 layer별 MSE가 아니라, 전체 모델을 통과시켜 나온 유사도 행렬(FP vs 양자화)의 차이를
end-to-end로 최소화하도록 모든 layer의 반올림(alpha)을 동시에 최적화한다.
(원 논문의 Distribution Correction은 v1에서 생략 — PD loss 핵심만.)

핵심 질문(우리 논문): PD-Quant는 '예측'을 맞추지만 그 예측이 calibration vocabulary
안에 갇혀 있어 held-out에서 여전히 실패하는가? 실패하면 "예측을 봐도 calib vocab에
갇힌다"가 확정되어 우리 decision-preservation 방법의 필요성이 정당화됨.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .adaround import AdaRoundQuantConv2d, list_adaround_convs, h_alpha


def _find_head(model):
    core = getattr(model, "model", model)
    return core[-1] if hasattr(core, "__getitem__") else core.model[-1]


class _CV4Capture:
    """head.cv4의 레벨별 출력을 grad 유지한 채 캡처."""
    def __init__(self, head):
        self.head = head
        self.buf = {}
        self.handles = []
        for i, sub in enumerate(head.cv4):
            def make(idx):
                def hook(_m, _inp, out):
                    self.buf[idx] = out          # grad 유지 (detach 안 함)
                return hook
            self.handles.append(sub.register_forward_hook(make(i)))

    def clear(self): self.buf = {}
    def close(self):
        for h in self.handles: h.remove()
        self.handles = []


def optimize_pdquant(quant_model, fp_model, calib_tensors, device,
                     iters=2000, lr=1e-3, reg_weight=1e-3, verbose=True):
    """
    end-to-end PD 최적화. 모든 AdaRound conv의 alpha를 동시에 최적화하여
    양자화 유사도 행렬이 FP 유사도 행렬에 가까워지도록 함.

    주의: layer-wise AdaRound보다 훨씬 무거움 — 매 iter 전체 모델 forward+backward.
    """
    ada_convs = list_adaround_convs(quant_model)
    for ac in ada_convs:
        ac.soft = True      # 미분 가능 반올림
        ac.ste = True       # activation STE로 grad 통과

    # FP cv4 타깃 캐시 (레벨별, calib 이미지별)
    fp_head = _find_head(fp_model)
    fp_cap = _CV4Capture(fp_head)
    fp_targets = []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear()
            fp_model(t.to(device))
            fp_targets.append({i: fp_cap.buf[i].detach() for i in fp_cap.buf})
    fp_cap.close()
    if verbose:
        print(f"[pdquant] FP 타깃 캐시 {len(fp_targets)}장, "
              f"alpha {len(ada_convs)}개 동시 최적화 (end-to-end)")

    q_head = _find_head(quant_model)
    q_cap = _CV4Capture(q_head)
    alphas = [ac.alpha for ac in ada_convs]
    opt = torch.optim.Adam(alphas, lr=lr)

    n = len(calib_tensors)
    for it in range(iters):
        idx = it % n
        t = calib_tensors[idx].to(device)
        tgt = fp_targets[idx]

        q_cap.clear()
        opt.zero_grad()
        quant_model(t)                                  # grad 흐르는 forward
        # PD loss: 레벨별 유사도 맵 MSE 합
        pd = 0.0
        for i in q_cap.buf:
            pd = pd + (q_cap.buf[i] - tgt[i]).pow(2).mean()
        # 반올림 정규화 (h를 0/1로)
        beta = max(2.0, 20.0 * (1 - it / iters))
        reg = sum(ac.reg_loss(beta) for ac in ada_convs)
        loss = pd + reg_weight * reg
        loss.backward()
        opt.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            hconv = sum(float(((h_alpha(ac.alpha) < 0.05) |
                               (h_alpha(ac.alpha) > 0.95)).float().mean())
                        for ac in ada_convs) / len(ada_convs) * 100
            print(f"  [{it+1}/{iters}] pd={float(pd.detach()):.5f} h→0/1 {hconv:.0f}%")

    q_cap.close()
    for ac in ada_convs:
        ac.soft = False     # 추론은 hard 반올림
        ac.ste = False      # 추론은 일반 양자화
    if verbose:
        print("[pdquant] 최적화 완료 (hard 반올림 모드)")
