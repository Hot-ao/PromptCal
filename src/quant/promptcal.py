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

def decision_loss(sim_q, sim_fp):
    """
    FP의 top-1 prompt를 pseudo-label로 사용하여
    quantized model의 top-1 decision을 FP와 일치시키는 loss.

    sim_*: [anchors, P] pre-sigmoid similarity
    """
    target = sim_fp.argmax(dim=-1).detach()  # FP top-1 = pseudo-GT
    return F.cross_entropy(sim_q, target)


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
                       decision_weight=0.1, k=5, conf_thres=0.25,
                       verbose=True, debug_eval=None, post_train_hook=None):
    """
    PromptCal: semantic decision-aware rounding optimization.

    핵심:
      1. FP의 top-1 prompt를 pseudo-label로 사용.
      2. 현재 quantized model에서 FP decision과 다른 anchor를 hard-example으로 선택.
      3. 선택된 anchor에 대해 FP-vs-quant decision을 직접 맞추도록 CE 최적화.
      4. margin은 보조 objective로만 사용.
      5. alpha를 0/1로 몰아가는 강한 regularization은 사용하지 않음.

    목적:
      soft alpha 상태에서 단순히 margin을 줄이는 것이 아니라,
      실제 discrete rounding 이후의 semantic decision을 보존하는 방향으로
      alpha가 움직이는지 확인하는 것.

    prompt_idx:
      S + H_cal.
      H_eval은 절대 optimization에 사용하지 않음.
    """
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = True
        ac.ste = True

    # ------------------------------------------------------------
    # 1. FP similarity cache
    # ------------------------------------------------------------
    fp_head = _find_head(fp_model)
    fp_cap = _CV4Capture(fp_head)
    fp_sims = []

    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear()
            fp_model(t.to(device))
            parts = []
            for i in sorted(fp_cap.buf):
                B, P, H, W = fp_cap.buf[i].shape
                parts.append(fp_cap.buf[i].reshape(B, P, H * W))
            sim = torch.cat(parts, dim=2)[0].transpose(0, 1)
            fp_sims.append(sim.detach())

    fp_cap.close()

    # ------------------------------------------------------------
    # 2. Quantized model capture
    # ------------------------------------------------------------
    q_head = _find_head(quant_model)
    q_cap = _CV4Capture(q_head)

    alphas = [ac.alpha for ac in ada]
    alpha_before = [a.detach().clone() for a in alphas]

    opt = torch.optim.Adam(alphas, lr=lr)
    pidx = torch.tensor(prompt_idx, device=device, dtype=torch.long)

    if verbose:
        print(f"[promptcal] {len(fp_sims)} calib, prompt subset "
              f"{len(prompt_idx)}개, alpha {len(ada)}개, "
              f"decision-aware rounding(k={k})")

    n = len(calib_tensors)

    for it in range(iters):
        j = it % n
        t = calib_tensors[j].to(device)
        sim_fp = fp_sims[j]

        # --------------------------------------------------------
        # 3. FP confident anchors
        # --------------------------------------------------------
        prob_fp = sim_fp.sigmoid()
        maxp_fp, fp_top1 = prob_fp.max(-1)
        conf = maxp_fp > conf_thres

        if conf.sum() == 0:
            continue

        aidx_all = conf.nonzero(as_tuple=True)[0]

        # --------------------------------------------------------
        # 4. Quantized forward
        # --------------------------------------------------------
        q_cap.clear()
        opt.zero_grad()

        quant_model(t)

        parts = []
        for i in sorted(q_cap.buf):
            B, P, H, W = q_cap.buf[i].shape
            parts.append(q_cap.buf[i].reshape(B, P, H * W))

        sim_q = torch.cat(parts, dim=2)[0].transpose(0, 1)

        # --------------------------------------------------------
        # 5. Calibration prompt subset
        # --------------------------------------------------------
        sq = sim_q[aidx_all][:, pidx]
        sf = sim_fp[aidx_all][:, pidx]

        # FP top-1 inside the calibration prompt subset
        fp_target = sf.argmax(dim=-1).detach()

        # Current quantized top-1
        q_prob = sq.sigmoid()
        q_top1 = q_prob.argmax(dim=-1).detach()

        # --------------------------------------------------------
        # 6. Hard semantic examples
        #
        # FP와 quant decision이 이미 같은 anchor보다
        # 현재 decision이 뒤집힌 anchor에 더 강한 gradient를 준다.
        # --------------------------------------------------------
        hard = q_top1 != fp_target

        if hard.any():
            sq_h = sq[hard]
            sf_h = sf[hard]
            target_h = fp_target[hard]

            decision = F.cross_entropy(sq_h, target_h)
        else:
            # 이미 decision이 모두 일치하면 전체 anchor를 사용하되
            # gradient가 지나치게 커지지 않도록 평균 CE만 사용.
            decision = F.cross_entropy(sq, fp_target)

        # --------------------------------------------------------
        # 7. Margin preservation
        #
        # decision loss가 주 objective.
        # margin은 decision boundary 주변의 안정성을 보조.
        # --------------------------------------------------------
        margin = margin_loss(sq, sf, k=k, boundary_w=3.0)

        # --------------------------------------------------------
        # 8. Alpha regularization
        #
        # 강한 h->0/1 forcing을 하지 않는다.
        # 현재 rounding 상태에서 너무 멀리 이동하는 것을 약하게 억제.
        # --------------------------------------------------------
        reg = sum(
            ac.reg_loss(beta=2.0, reduction="mean")
            for ac in ada
        ) / len(ada)

        loss = (
            decision_weight * decision
            + margin
            + reg_weight * reg
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(alphas, max_norm=1.0)
        opt.step()

        # --------------------------------------------------------
        # 9. Logging
        # --------------------------------------------------------
        if verbose and (it + 1) % max(1, iters // 10) == 0:
            with torch.no_grad():
                flip_ratio = float(hard.float().mean()) * 100.0
                hc = sum(
                    float(
                        (
                            (h_alpha(a.alpha) < 0.05)
                            | (h_alpha(a.alpha) > 0.95)
                        ).float().mean()
                    )
                    for a in ada
                ) / len(ada) * 100.0

            print(
                f"  [{it+1}/{iters}] "
                f"loss={float(loss.detach()):.4f} "
                f"decision={float(decision.detach()):.4f} "
                f"margin={float(margin.detach()):.4f} "
                f"reg={float(reg.detach()):.4f} "
                f"soft_flip={flip_ratio:.1f}% "
                f"h→0/1={hc:.0f}%"
            )
            if (it + 1) in [100, 200, 300, 500, 750, 1000, 1250, 1500]:
                torch.save(
                    {
                        "iter": it + 1,
                        "alphas": [a.detach().cpu() for a in alphas],
                        "h": [h_alpha(a.detach()).cpu() for a in alphas],
                    },
                    f"/tmp/promptcal_iter_{it+1}.pt"
                )
            

    q_cap.close()

    if post_train_hook is not None:
        # 아직 soft=True/ste=True인 상태(hardening 전)에서 콜백 실행.
        # Q1~Q4 진단: soft 상태의 H_cal margin/flip을 discrete 상태와 분리해서 측정하기 위함.
        post_train_hook(quant_model)

    for ac in ada:
        ac.soft = False
        ac.ste = False

    if verbose:
        tot_change = sum(
            float((a.detach() - b).abs().sum())
            for a, b in zip(alphas, alpha_before)
        )
        print(
            f"[promptcal] 최적화 완료 "
            f"(alpha 총 변화량={tot_change:.2f})"
        )        
        
def optimize_promptcal_scale(quant_model, fp_model, calib_tensors, device,
                             prompt_idx, iters=1000, lr=1e-2, k=5,
                             conf_thres=0.25, verbose=True, eval_hook=None):
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

        if eval_hook is not None and (it + 1) % max(1, iters // 10) == 0:
            eval_hook(it + 1, quant_model)

    q_cap.close()
    if verbose:
        tot = sum(float((s.detach()-s0i).abs()) for s, s0i in zip(smults, s0))
        print(f"[promptcal-C] 완료 (s_mult 총 변화={tot:.3f})")
