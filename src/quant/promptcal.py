"""
PromptCal-PTQ 최소 실험(국면 E, 첫 관문): held-out margin 보존.

가설: held-out prompt(H_cal)를 calibration에 포함하고, 그 prompt들의 top-k 경계 margin을
FP와 맞추도록 양자화 파라미터를 최적화하면, 완전히 안 본 prompt(H_eval)에서의 flip이
baseline(reconstruction 계열)보다 준다.

09_phase1(RankSafe) 실패와의 차이:
  - calibration 지표를 줄이는 게 아니라 held-out(H_cal) margin을 직접 보존 대상으로.
  - 다항 짬뽕(rank+topk+dist+box+hw)이 아니라 margin 보존 '하나'만.
  - 측정은 H_eval(학습에 안 쓴 prompt)에서만 → GT gate.

구현: AdaRound rounding + end-to-end. cv4 유사도 행렬을 S+H_cal 프롬프트로 뽑아
top-(k+1) 인접 margin을 FP와 맞춤. weight 고정, alpha(round)만 최적화.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .adaround import AdaRoundQuantConv2d, list_adaround_convs, h_alpha
from .pdquant import _find_head, _CV4Capture


def margin_loss(sim_q, sim_fp, k=5, boundary_w=3.0):
    """top-(k+1) 인접 pairwise margin을 FP와 맞춤. top-k 경계 margin에 가중.
    sim_*: [anchors, P] pre-sigmoid 유사도. confident anchor만 넣어 호출."""
    kk = min(k + 1, sim_fp.shape[-1])
    fp_top, _ = sim_fp.topk(kk, dim=-1)
    q_top, _ = sim_q.topk(kk, dim=-1)
    fp_m = fp_top[:, :-1] - fp_top[:, 1:]        # [A, kk-1]
    q_m = q_top[:, :-1] - q_top[:, 1:]
    w = torch.ones(kk - 1, device=sim_fp.device)
    w[-1] = boundary_w                            # k-1↔k 경계 강조
    return ((q_m - fp_m).pow(2) * w).mean()


def optimize_promptcal(quant_model, fp_model, calib_tensors, device,
                       prompt_idx, iters=1000, lr=1e-3, reg_weight=1.0,
                       k=5, conf_thres=0.25, verbose=True):
    """
    held-out margin 보존 최적화. AdaRound로 초기화된 모델을 refine.

    prompt_idx: calibration에 쓸 프롬프트 인덱스(S + H_cal). cv4 유사도 행렬에서
                이 열들만 뽑아 margin 보존. (모델은 이미 set_classes로 전체 P를 갖되,
                loss는 이 부분집합에서만 계산 — H_eval은 제외)
    """
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = True; ac.ste = True

    fp_head = _find_head(fp_model)
    fp_cap = _CV4Capture(fp_head)
    # FP 유사도 타깃 캐시 (레벨별 cv4 출력을 [A,P]로 조립)
    fp_sims = []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear(); fp_model(t.to(device))
            parts = []
            for i in sorted(fp_cap.buf):
                B, P, H, W = fp_cap.buf[i].shape
                parts.append(fp_cap.buf[i].reshape(B, P, H*W))
            sim = torch.cat(parts, dim=2)[0].transpose(0, 1)   # [A, P]
            fp_sims.append(sim.detach())
    fp_cap.close()

    q_head = _find_head(quant_model)
    q_cap = _CV4Capture(q_head)
    alphas = [ac.alpha for ac in ada]
    alpha_before = [a.detach().clone() for a in alphas]   # 진단: 변화량 추적
    opt = torch.optim.Adam(alphas, lr=lr)
    pidx = torch.tensor(prompt_idx, device=device)

    if verbose:
        print(f"[promptcal] {len(fp_sims)} calib, prompt subset {len(prompt_idx)}개, "
              f"alpha {len(ada)}개, margin 보존(k={k})")

    n = len(calib_tensors)
    warmup = int(0.4 * iters)      # 진단: 전반부는 reg 강하게 h를 0/1로 수렴시킴
    for it in range(iters):
        j = it % n
        t = calib_tensors[j].to(device)
        sim_fp = fp_sims[j]                          # [A, P]
        # confident anchor 선택 (FP 기준)
        prob = sim_fp.sigmoid(); maxp, _ = prob.max(-1)
        conf = maxp > conf_thres
        if conf.sum() == 0:
            continue
        aidx = conf.nonzero(as_tuple=True)[0]

        q_cap.clear(); opt.zero_grad()
        quant_model(t)
        parts = []
        for i in sorted(q_cap.buf):
            B, P, H, W = q_cap.buf[i].shape
            parts.append(q_cap.buf[i].reshape(B, P, H*W))
        sim_q = torch.cat(parts, dim=2)[0].transpose(0, 1)     # [A, P] grad 유지

        # S+H_cal 프롬프트 부분집합 + confident anchor에서만 margin 보존
        sq = sim_q[aidx][:, pidx]
        sf = sim_fp[aidx][:, pidx]
        ml = margin_loss(sq, sf, k=k)
        # rounding 수렴: 전반부 reg 강하게(h→0/1), 후반부 완화 + margin 주도
        if it < warmup:
            beta = max(2.0, 20.0 * (1 - it / warmup))
            rw = 10.0                                  # 강한 reg로 h 수렴 우선
        else:
            beta = 2.0
            rw = reg_weight                            # margin 주도
        reg = sum(ac.reg_loss(beta, reduction="mean") for ac in ada) / len(ada)
        loss = ml + rw * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(alphas, max_norm=1.0)
        opt.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            hc = sum(float(((h_alpha(a.alpha) < 0.05) | (h_alpha(a.alpha) > 0.95))
                     .float().mean()) for a in ada) / len(ada) * 100
            phase = "warmup" if it < warmup else "margin"
            print(f"  [{it+1}/{iters}] margin_loss={float(ml.detach()):.4f} h→0/1 {hc:.0f}% ({phase})")

    q_cap.close()
    for ac in ada:
        ac.soft = False; ac.ste = False
    if verbose:
        tot_change = sum(float((a.detach()-b).abs().sum()) for a,b in zip(alphas, alpha_before))
        print(f"[promptcal] 최적화 완료 (alpha 총 변화량={tot_change:.2f} — 0이면 안 움직인 것)")


def optimize_promptcal_scale(quant_model, fp_model, calib_tensors, device,
                             prompt_idx, iters=1000, lr=1e-2, k=5,
                             conf_thres=0.25, verbose=True):
    """
    방향 C: rounding(alpha) 대신 learnable activation scale(s_mult)을 최적화.

    동기: alpha는 이산적(0/1로 굳어야)이라 연속 목적함수(margin)와 미스매치 →
    세 번 실패(h가 중간에 껴서 hard 전환 오류). activation scale은 연속값이라
    margin과 결이 맞고, '굳혀야 하는' 문제가 없음.

    - rounding은 round-to-nearest로 고정(soft=False, alpha 최적화 안 함).
    - use_smult=True로 각 conv의 s_mult(초기 1.0)만 최적화.
    - 목적: S+H_cal 프롬프트의 top-k margin을 FP와 맞춤(margin_loss).
    """
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = False          # rounding 고정(round-to-nearest)
        ac.ste = False
        ac.use_smult = True      # learnable scale 모드
        ac.alpha.requires_grad_(False)

    fp_head = _find_head(fp_model); fp_cap = _CV4Capture(fp_head)
    fp_sims = []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear(); fp_model(t.to(device))
            parts = []
            for i in sorted(fp_cap.buf):
                B, P, H, W = fp_cap.buf[i].shape
                parts.append(fp_cap.buf[i].reshape(B, P, H*W))
            fp_sims.append(torch.cat(parts, dim=2)[0].transpose(0, 1).detach())
    fp_cap.close()

    q_head = _find_head(quant_model); q_cap = _CV4Capture(q_head)
    smults = [ac.s_mult for ac in ada]
    s0 = [s.detach().clone() for s in smults]
    opt = torch.optim.Adam(smults, lr=lr)
    pidx = torch.tensor(prompt_idx, device=device)

    if verbose:
        print(f"[promptcal-C] {len(fp_sims)} calib, prompt subset {len(prompt_idx)}개, "
              f"s_mult {len(ada)}개 최적화(scale), margin(k={k})")

    n = len(calib_tensors)
    for it in range(iters):
        j = it % n
        t = calib_tensors[j].to(device); sim_fp = fp_sims[j]
        prob = sim_fp.sigmoid(); mp, _ = prob.max(-1); conf = mp > conf_thres
        if conf.sum() == 0:
            continue
        aidx = conf.nonzero(as_tuple=True)[0]
        q_cap.clear(); opt.zero_grad()
        quant_model(t)
        parts = []
        for i in sorted(q_cap.buf):
            B, P, H, W = q_cap.buf[i].shape
            parts.append(q_cap.buf[i].reshape(B, P, H*W))
        sim_q = torch.cat(parts, dim=2)[0].transpose(0, 1)
        ml = margin_loss(sim_q[aidx][:, pidx], sim_fp[aidx][:, pidx], k=k)
        ml.backward()
        torch.nn.utils.clip_grad_norm_(smults, max_norm=1.0)
        opt.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            sd = sum(float((s.detach()-s0i).abs()) for s, s0i in zip(smults, s0)) / len(smults)
            smean = sum(float(s.detach()) for s in smults) / len(smults)
            print(f"  [{it+1}/{iters}] margin_loss={float(ml.detach()):.4f} "
                  f"s_mult 평균={smean:.3f} 변화={sd:.4f}")

    q_cap.close()
    if verbose:
        tot = sum(float((s.detach()-s0i).abs()) for s, s0i in zip(smults, s0))
        print(f"[promptcal-C] 완료 (s_mult 총 변화={tot:.3f})")
