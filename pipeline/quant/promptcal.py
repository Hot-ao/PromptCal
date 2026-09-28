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

from .adaround import AdaRoundQuantConv2d, list_adaround_convs, h_alpha, temp_decay
from .pdquant import _find_head, _CV4Capture
from .semantic_calib import get_txt_feats, text_neighbor_order, _LevelCapture, utility_refinement_terms


class _BlockCapture:
    """DetectionModel.model(Sequential)의 top-level 블록 출력을 캡처.
    09-28 공동 최적화용 -- BRECQ가 재구성하는 것과 같은 층위(블록 출력)의 dense 신호를
    stage-2 목적함수에 직접 넣기 위함. 양자화 모델 쪽은 grad를 유지하고(detach 안 함),
    FP 쪽은 no_grad로 부른다. idxs: 양자화 conv가 있는 블록만(나머지는 grad 경로 없음)."""
    def __init__(self, detection_model, idxs):
        self.buf = {}
        self.handles = []
        blocks = list(detection_model.model)
        for i in idxs:
            def make(k):
                def hook(_m, _inp, out):
                    if torch.is_tensor(out):
                        self.buf[k] = out
                return hook
            self.handles.append(blocks[i].register_forward_hook(make(i)))

    def clear(self): self.buf = {}
    def close(self):
        for h in self.handles: h.remove()
        self.handles = []

def decision_loss(sim_q, sim_fp):
    """
    FP의 top-1 prompt를 pseudo-label로 사용하여
    quantized model의 top-1 decision을 FP와 일치시키는 loss.

    sim_*: [anchors, P] pre-sigmoid similarity
    """
    target = sim_fp.argmax(dim=-1).detach()  # FP top-1 = pseudo-GT
    return F.cross_entropy(sim_q, target)


def margin_loss(sim_q, sim_fp, k=5, boundary_w=3.0, identity_aware=False, one_sided=False):
    """top-(k+1) 인접 pairwise margin을 FP와 맞춤. top-k 경계 margin에 가중.
    sim_*: [anchors, P] pre-sigmoid 유사도. confident anchor만 넣어 호출.

    identity_aware=False(기본, 기존 동작): fp_top/q_top을 각자 독립적으로
    topk해서 정렬된 값끼리만 비교한다 -- 어느 class가 그 순위를 차지했는지는
    버려진다. 09-15 claim5 지적: top-1/top-2가 서로 값을 맞바꾸는(class identity가
    뒤집히는) flip이 일어나도 정렬 후 margin 패턴 자체는 동일하면 loss=0이라,
    이 loss가 막으려는 대상(flip)이 목적함수에 아예 안 보이는 경로가 있다.
    identity_aware=True: fp_top의 index로 sim_q를 gather해서 동일 class 위치의
    값끼리 비교(q_top이 더 이상 정렬돼있지 않음 -- Q에서 순서가 FP와 달라지면
    q_m이 음수가 되면서 실제로 벌점을 받는다).

    09-16 claim5-b 지적 및 수정: 순수 gather는 FP top-(k+1) "밖"에 있던 class가
    Q에서 값이 치솟아 실제 1등을 빼앗는 경우(intrusion)를 아예 못 본다 --
    fp_idx가 그 class의 열 자체를 가리키지 않기 때문. 반면 기존 정렬 버전은
    identity는 몰라도 Q 자신의 top-1 값 자체가 커지는 식으로 intrusion을
    부분적으로 잡아냈었다 -- 즉 identity_aware가 기존 버전의 상위호환이
    아니라 "swap은 잡고 intrusion은 놓치는" 다른 trade-off였다. 마지막
    열(boundary_w가 걸리는 자리, FP rank-(k+1))만 "FP top-k(rank 1..k) 밖에
    있는 class 중 Q에서 가장 높은 값"으로 바꿔서, swap(앞쪽 k개 열, identity
    고정)과 intrusion(마지막 열, FP top-k 밖 전체에서 최댓값)을 둘 다 하나의
    boundary 비교에 담는다. FP 쪽 마지막 값(fp_top[:, -1], FP rank-(k+1))과
    비교 대상이 정확히 대응되므로(둘 다 "top-k 경계 바로 바깥의 가장 위협적인
    값") 의미도 일관된다."""
    kk = min(k + 1, sim_fp.shape[-1])
    fp_top, fp_idx = sim_fp.topk(kk, dim=-1)
    if identity_aware:
        q_top = sim_q.gather(-1, fp_idx)
        if kk > 1:
            mask = torch.zeros_like(sim_q, dtype=torch.bool).scatter_(-1, fp_idx[:, :-1], True)
            q_out = sim_q.masked_fill(mask, float("-inf")).max(-1).values
            q_top = torch.cat([q_top[:, :-1], q_out[:, None]], dim=1)
    else:
        q_top, _ = sim_q.topk(kk, dim=-1)
    fp_m = fp_top[:, :-1] - fp_top[:, 1:]        # [A, kk-1]
    q_m = q_top[:, :-1] - q_top[:, 1:]
    w = torch.ones(kk - 1, device=sim_fp.device)
    w[-1] = boundary_w                            # k-1↔k 경계 강조
    if one_sided:
        # 09-27: margin이 FP보다 **좁아진 경우만** 벌점. 기존 대칭 형태
        # (q_m - fp_m)^2는 margin이 넓어진 것도 똑같이 벌한다 -- 그런데 margin이
        # 넓어지는 건 flip에서 멀어지는 것이므로 decision preservation 관점에서
        # 바람직하다. 실측(naive W8A8, calib 64, confident anchor):
        #   margin 항 5,325개 중 FP보다 넓어진 항 2,559개(48.1%),
        #   실제 flip(q_m<0) 622개(11.7%),
        #   **전체 벌점의 30.4%가 '넓어진 margin'을 억제하는 데 쓰였다.**
        # 즉 대칭 형태는 "FP의 margin을 정확히 복원하라"(= margin 공간의
        # reconstruction loss)이지 "뒤집지 마라"(= decision loss)가 아니다.
        # 이 프로젝트는 같은 통찰을 이미 neighbor_loss에서 확인해 단측
        # (asymmetric hinge)을 채택했는데(41번), margin_loss 본체에는 적용하지
        # 않았다 -- claim15의 "BRECQ 기반 위에서 margin_loss가 BRECQ 자신의
        # LSQ보다 못하다"도 이걸로 설명된다(둘 다 MSE인데 우리 쪽이 더 희박한
        # 사영 위의 MSE).
        return (F.relu(fp_m - q_m).pow(2) * w).mean()
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
            sd = sum(float((s.detach()-s0i).abs().mean()) for s, s0i in zip(smults, s0)) / len(smults)
            smean = sum(float(s.detach().mean()) for s in smults) / len(smults)
            print(f"  [{it+1}/{iters}] margin_loss={float(ml.detach()):.4f} "
                  f"s_mult 평균={smean:.3f} 변화={sd:.4f}")

        if eval_hook is not None and (it + 1) % max(1, iters // 10) == 0:
            eval_hook(it + 1, quant_model)

    q_cap.close()
    if verbose:
        tot = sum(float((s.detach()-s0i).abs().sum()) for s, s0i in zip(smults, s0))
        print(f"[promptcal-C] 완료 (s_mult 총 변화={tot:.3f})")


def optimize_promptcal_scale_neighbor(quant_model, fp_model, calib_tensors, device,
                                      prompt_idx, iters=1000, lr=1e-2, k=5,
                                      boundary_w=3.0, neighbor_k=5, neighbor_weight=1.0,
                                      asymmetric=False, scale_reg_weight=0.0,
                                      exclude_from_neighbors=None,
                                      cal_idx=None, cal_weight=1.0,
                                      conf_thres=0.25, verbose=True, eval_hook=None,
                                      identity_aware_margin=False, control_mse=False,
                                      neighbor_of_cal=False, aux_mse_weight=0.0,
                                      learn_alpha=False, alpha_lr=1e-2,
                                      alpha_reg_weight=1e-2, alpha_warmup=0.2,
                                      learn_alpha_bias=False, alpha_bias_lr=1e-2,
                                      alpha_bias_reg_weight=1e-3, alpha_bias_limit=0.5,
                                      utility_stage2_frac=0.0, thresh_w=1.0, box_w=0.5,
                                      det_thres=0.25, margin_thres=0.5,
                                      region_dir_weight=0.0, margin_one_sided=False,
                                      random_sample=False, per_group_anchors=False,
                                      local_recon_weight=0.0, block_recon_weight=0.0):
    """
    방향 C(연속 s_mult) + neighbor preservation (09-05 §40 진단 이후).

    동기: optimize_promptcal_scale은 S(pidx) 프롬프트의 top-k margin만 FP와
    맞춘다. 그런데 s_mult는 class-agnostic한 activation scale이라 S의 margin을
    맞추려는 움직임이 나머지 79개 출력 컬럼 전체에 공유되어 전파된다 -- 특히
    text embedding상 S와 가까운 class일수록 이 collateral shift를 크게 받는다
    (09-05 §40: H_eval이 S와 가까운 seed일수록 held-out 성능이 더 나빠짐을 관찰).

    이 함수는 margin_loss에 다음을 추가한다: S의 각 class에 대해 text embedding
    최근접 이웃(비-S class) neighbor_k개를 뽑아, 그 class들의 similarity가 FP
    대비 흔들리지 않도록 붙잡아둔다. GT/held-out identity 불필요 -- FP text
    encoder(get_txt_feats)만으로 "위험한 이웃"을 계산하므로 논문 §4.2 Prompt
    Selection("teacher target 및 semantically competitive prompts 선택, GT
    annotation 없이 teacher-derived signal 사용")과 동일한 제약을 따른다.

    30번(semantic_calib.py)과의 차이: 30번은 경쟁 프롬프트를 상대 margin 계산에만
    썼다(target-competitor 차이를 FP와 맞춤) -- competitor 자체의 절대 유사도가
    드리프트하는 건 막지 못했다. 이 함수는 그 절대값 드리프트를 직접 억제하는
    항을 margin_loss 하나에만 단독으로 추가한다(09-03 §24 원칙: 한 번에 하나씩).

    asymmetric=False(기본): symmetric MSE. 이웃의 유사도가 FP보다 오르든 내리든
    똑같이 벌점을 준다. 41번 실험(neighbor_weight 0.3/0.5/1.0 스윕)에서 seed마다
    반응이 뒤바뀌는 불안정한 패턴이 나왔다 -- 이웃 쪽으로의 흔들림이 우연히
    유익한 seed(0)에서도 무차별적으로 억제해버렸기 때문으로 추정.

    asymmetric=True: one-sided hinge, relu(sim_q - sim_fp)^2 만 사용. 이웃의
    유사도가 FP보다 "낮아지는" 방향(경쟁자가 약해짐 -> 원래 anchor의 target이
    안전해짐)은 전혀 벌점을 주지 않고, "높아지는" 방향(경쟁자가 FP보다 강해져서
    실제로 top-1을 빼앗을 위험이 커짐)만 억제한다. semantic_calib.py의
    utility_refinement_terms(threshold-crossing hinge)와 같은 스타일.

    scale_reg_weight (09-07, LVIS 일반성 검증 이후 추가): s_mult가 conv당
    스칼라 하나뿐이라 "이웃 컬럼만 조준"이 구조적으로 불가능하고, asymmetric
    hinge를 값싸게 만족시키는 가장 쉬운 해법이 "전체를 조금씩 낮추는" 전역
    편향으로 수렴한다는 게 확인됨(COCO-80/LVIS-이식/LVIS-native 세 시나리오
    모두에서 s_mult 평균이 1.0 미만으로 수렴 -- PromptCal_PTQ_progress_
    2026-09-07.md §7.3~7.7). 이 편향은 계산에 쓴 vocabulary뿐 아니라 배포
    시점의 임의의(학습이 전혀 모르는) vocabulary에도 무차별 적용되어 실제
    AP·UPIR을 해친다(s_mult=1로 되돌리면 AdaRound 수준으로 정확히 회복됨,
    `scripts/52_smult_ablation.py`로 인과관계 확정). scale_reg_weight>0이면
    `mean((s_mult-1)^2)`을 손실에 더해 "값싸게 전역적으로 줄이는" 지름길에
    비용을 매겨서, 최적화가 margin/neighbor 목적에 실제로 필요한 만큼만
    s_mult를 움직이도록 유도한다. 기본값 0.0(기존 동작 유지, opt-in).

    cal_idx/cal_weight (09-12, APr(rare) 열위·lost 그룹 분해 이후 추가): 지금까지
    S(prompt_idx, 40개)만 margin_loss로 직접 보호받고, H_cal(20개)은 "S의
    neighbor로 우연히 뽑히면" asymmetric hinge로 간접·수동적으로만 억제됐다.
    최적화 압력이 S에만 집중되면서 COCO-80 전체 lost/APr(rare-class)가
    희생된다는 정황(§9)에 대응해, cal_idx(H_cal 인덱스)를 넘기면 S와 동일한
    margin_loss를 H_cal 컬럼에도 별도로 적용해 S∪H_cal(60개, COCO-80의 75%)을
    직접 보호 대상으로 넓힌다. cal_idx가 주어지면 해당 인덱스는 neighbor 후보
    풀에서도 제외한다(이미 margin_loss로 직접 보호받으므로 이중 처리 방지).
    cal_weight는 S 쪽 margin_loss 대비 H_cal 쪽 margin_loss의 상대 가중치.
    cal_idx=None(기본)이면 기존 동작과 완전히 동일(opt-in).

    control_mse (09-17, claim: "baseline엔 activation scale을 학습하는 손잡이가
    아예 없다" 검증용 control 실험): True면 margin_loss/neighbor-hinge/cal/
    scale_reg를 전부 건너뛰고, train_cols(S∪H_cal) 전체에 대해 순수
    F.mse_loss(sim_q, sim_fp)만 최적화한다. s_mult 파라미터화·optimizer·iters·
    confident-anchor(aidx) 선정은 기존과 동일하게 유지 -- "손잡이가 있다는
    사실 자체"가 이득의 원인인지, margin_loss/identity-aware 설계가 추가로
    기여하는지를 분리하기 위한 단일 변수 ablation. k/boundary_w/neighbor_k/
    neighbor_weight/asymmetric/scale_reg_weight/cal_idx/cal_weight/
    identity_aware_margin은 control_mse=True일 때 전부 무시된다.
    control_mse=False(기본)이면 기존 동작과 완전히 동일.

    neighbor_of_cal (09-20, claim16 방향 2 -- margin_loss 보호 범위 확장):
    기존엔 S(prompt_idx)의 이웃만 neighbor_cols에 들어갔다. True면 H_cal
    (cal_idx)의 이웃도 같은 exclude_set(H_eval 포함)으로 걸러서 추가한다 --
    H_eval은 exclude_set에 이미 있으므로 이 확장은 held-out 불변식을 절대
    깨지 않는다(같은 exclude_set을 재사용). claim6에서 S의 이웃 풀이 이미
    H_cal 크기로 포화된다고 확인됐으니, 이건 "더 많이"가 아니라 "다른 각도
    에서" 보호 범위를 넓히는 것 -- H_cal 자신의 최근접 이웃(S와는 다를 수
    있음)까지 커버. 기본 False(기존 동작 유지, opt-in).

    aux_mse_weight (09-20, claim16 방향 3 -- margin_loss에 dense 신호 추가):
    margin_loss는 top-(k+1) 개 boundary만 보는 sparse한 신호다(BRECQ-stage1
    진단, claim15에서 margin_loss가 BRECQ 자신의 dense reconstruction
    objective보다 decision-preservation에 못한 것으로 확인됨). >0이면
    train_cols(S∪H_cal, H_eval 무관) 전체에 대한 F.mse_loss를 margin_loss에
    "더해서"(대체가 아니라 추가) 얹는다 -- control_mse가 margin_loss를
    통째로 대체하는 것과 다르다. 기본 0.0(기존 동작 유지, opt-in).
    """
    ada = list_adaround_convs(quant_model)
    # alpha_bias만 학습할 때는 soft를 켜지 않고 round_ste(hard forward + STE backward)를
    # 쓴다 -- soft를 켜면 alpha_bias와 무관하게 "최적화 대상 모델"이 바뀌어서
    # s_mult-only 실험과의 비교에 교란이 섞인다(adaround.py round_ste 주석 참고).
    # learn_alpha(원소별)는 기존 그대로 soft 경로를 쓴다. 둘 다 켜면 soft 우선.
    _use_ste = bool(learn_alpha_bias and not learn_alpha)
    for ac in ada:
        ac.soft = bool(learn_alpha)
        ac.round_ste = _use_ste
        ac.use_alpha_bias = bool(learn_alpha_bias)
        if learn_alpha_bias:
            ac.alpha_bias_limit = alpha_bias_limit
        ac.ste = False
        ac.use_smult = True
        ac.alpha.requires_grad_(bool(learn_alpha))
        ac.alpha_bias.requires_grad_(bool(learn_alpha_bias))

    # utility_stage2_frac > 0 (09-24): 논문 §4.3 Utility-Constrained Refinement.
    # margin/neighbor(=ranking 보존)만으로는 최종 탐지 성능이 보장되지 않는다는
    # 논문 motivation(§Reconstruction--Utility Misalignment)에 직접 대응하는 항으로,
    # 마지막 stage2_frac 구간에서만 켜진다:
    #   l_thresh -- FP가 det_thres를 넘었던 target의 quant 확률이 그 밑으로
    #               떨어지지 않게 하는 one-sided hinge (threshold crossing)
    #   l_box    -- 같은 anchor에서 cv2(box regression) 출력을 FP와 MSE로 맞춤
    # 구현은 semantic_calib.utility_refinement_terms를 그대로 쓴다(30번/구
    # optimize_..._utility와 동일). 기본 0.0 = 꺼짐 = 기존 동작과 bit-identical.
    # region_dir_weight > 0 (09-25): vocabulary-agnostic 정규화.
    # ContrastiveHead는 sim_j = x_hat · w_hat_j 이므로, region embedding의 단위 방향
    # x_hat이 보존되면 **프롬프트를 하나도 참조하지 않고** seen/unseen/in-span/out-of-span
    # 모든 vocabulary의 유사도가 함께 보존된다.
    #
    # 왜 이 형태인가: 초록은 "calibration vocabulary 과적합을 억제해 새 vocabulary에서도
    # 보존"을 약속하는데, 현재 목적함수의 모든 항(margin/neighbor/scale_reg)이
    # calibration vocabulary 위에서 계산된다 -- neighbor_loss는 claim16이 보였듯
    # COCO-80 폐쇄 구조에서 H_cal과 구조적으로 동일해 무력화됐다. 그렇다고 합성
    # 프롬프트(calibration 임베딩의 볼록결합)를 쓰는 건 수학적으로 퇴화한다:
    # sim_s = (a·s_raw)/sqrt(a^T G a)로 calibration 컬럼들의 결정적 함수라,
    # 그 컬럼이 보존되면 자동 보존 = claim16 aux_mse와 같은 것이 된다.
    # 프롬프트 쪽에서는 span 밖으로 못 나가므로, 한 단계 아래인 region 쪽을 잡는다.
    #
    # BRECQ의 feature reconstruction과 다른 점: BRECQ는 모든 블록의 원시 activation을
    # MSE로 맞추고, 여기서는 head 입력의 **단위 방향**만 (decision-relevant anchor에서)
    # 지킨다 -- 크기는 버린다. 순위를 정하는 건 방향이기 때문.
    # 기본 0.0 = 꺼짐 = 기존 동작과 bit-identical.
    # local_recon_weight > 0 (09-27): 논문 §4 Semantic Objective가 명시하는
    # "Local reconstruction + region-prompt semantic consistency"의 앞쪽 절반.
    # 현재 목적함수는 semantic consistency(margin/neighbor)만 있고 reconstruction
    # 항이 없어서, 2단계가 1단계(BRECQ) 결과에서 자유롭게 멀어진다 -- claim14의
    # "1단계 이득이 2단계 밖으로 샌다"가 그 증상이다.
    #
    # 근거가 하나 더 있다: 대칭 margin_loss가 단측보다 나았는데(runs/110), 대칭
    # 형태는 "FP margin으로 되돌려라"라서 **암묵적 앵커** 역할을 한다. 암묵적
    # 앵커가 도움이 되면 명시적 앵커는 더 나을 수 있다.
    #
    # 형태: cv4 입력(region feature)의 상대 제곱오차. 크기까지 포함한다는 점에서
    # region_dir(방향만)과 다르고, ||x_fp||^2로 정규화해 스케일 무관하게 만들어
    # margin_loss와 같은 수준(O(0.01~0.1))에서 비교되게 한다.
    # block_recon_weight > 0 (09-28): 진짜 공동 최적화.
    # 기존 구조는 순차다 -- BRECQ가 블록 재구성으로 alpha를 맞춘 뒤 hard 고정하고 stage 2가
    # s_mult만 ranking 목적함수로 움직인다. 그래서 stage 2가 stage 1의 결과를 **사후에
    # 교란**한다(W8A8에서도 Heval_flip 3.802→4.278 악화, 저비트에서 전면화 -- W4A8 0/3,
    # W8A6 0/7). 손상이 클수록 그 지점이 민감해 역효과가 커진다.
    #
    # 이 항은 BRECQ가 재구성하는 것과 **같은 층위(블록 출력)** 의 재구성 손실을 stage 2
    # 목적함수에 직접 넣어 교란이 아니라 협상이 되게 한다. 매 iteration FP 모델을 한 번 더
    # forward해 블록 출력을 얻는다(no_grad, 캐시 불필요).
    #
    # learn_alpha와 함께 쓰면 alpha도 (재구성 + ranking) 아래에서 같이 풀린다 = 논문 §4
    # Semantic Objective("Local reconstruction + region-prompt semantic consistency")의 형태.
    # claim18-a가 실패한 이유("자유도 수백만인데 감독이 sparse")를 이 dense 항이 메운다.
    # 형태는 블록별 상대 MSE의 평균 -- 스케일 무관해서 weight 1.0이 margin_loss와 같은
    # 수준(O(0.01~0.1))에서 경쟁한다.
    use_block_recon = block_recon_weight > 0
    use_local_recon = local_recon_weight > 0
    use_region_dir = region_dir_weight > 0
    use_utility = utility_stage2_frac > 0
    _need_region = use_region_dir or use_local_recon
    fp_head = _find_head(fp_model); fp_cap = _CV4Capture(fp_head, capture_input=_need_region)
    fp_cv2_cap = _LevelCapture(fp_head.cv2) if use_utility else None
    fp_sims, fp_cv2s, fp_regions = [], [], []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear()
            if fp_cv2_cap is not None:
                fp_cv2_cap.clear()
            fp_model(t.to(device))
            parts = []
            for i in sorted(fp_cap.buf):
                B, P, H, W = fp_cap.buf[i].shape
                parts.append(fp_cap.buf[i].reshape(B, P, H*W))
            fp_sims.append(torch.cat(parts, dim=2)[0].transpose(0, 1).detach())
            if _need_region:
                # 정규화하지 않은 원본을 캐시한다(region_dir은 사용 시점에 정규화).
                fp_regions.append(fp_cap.assemble_input().detach())
            if fp_cv2_cap is not None:
                fp_cv2s.append(fp_cv2_cap.assemble().detach())
    fp_cap.close()
    if fp_cv2_cap is not None:
        fp_cv2_cap.close()

    txt_feats = get_txt_feats(fp_model).to(device)
    neighbor_order = text_neighbor_order(txt_feats)
    pidx_set = set(prompt_idx)
    # 09-09 버그 수정: neighbor 후보 풀에서 S(pidx_set)만 제외하면, H_eval(순수
    # held-out이어야 할 프롬프트)이 text-embedding상 S와 가까울 경우 neighbor_cols에
    # 그대로 섞여 들어가 asymmetric hinge로 간접 학습돼버린다("H_eval은 최적화에
    # 한 번도 안 씀"이 깨짐 -- 실측 seed=0 기준 H_eval 20개 중 18개가 포함돼
    # 있었음). exclude_from_neighbors로 H_eval도 같이 제외해서 진짜 held-out을
    # 보장한다. None이면 기존 동작(버그 있는 채로) 유지 -- 하위 호환용.
    #
    # 09-12 버그 발견: cal_idx(H_cal)를 여기서도 제외하면(초안 -- "이미
    # margin_loss로 보호하니 이중 처리 방지" 의도였음) S+H_cal+H_eval이 정확히
    # COCO-80 80개 전부(40+20+20)라서 후보 풀이 통째로 0개가 되어 neighbor_cols가
    # 항상 빈 텐서가 되고, F.mse_loss/relu(...).pow(2).mean()이 빈 텐서에 대해
    # 조용히 nan을 반환한다(neighbor_weight*nan을 더해 loss 자체는 nan이 되지만,
    # 빈 인덱싱의 backward는 실제로 0 gradient만 주므로 s_mult가 nan으로 오염되진
    # 않음 -- 대신 neighbor-hinge 항 자체가 통째로 무효화된 채 조용히 실행됨,
    # 재현: neighbor 0개/neighbor_loss=nan 로그로 확인). cal_idx는 neighbor 후보
    # 풀에서 제외하지 않는다 -- cal_weight=0일 때 기존 동작과 완전히 동일해야
    # 한다는 불변식(cal_idx가 주어져도 가중치 0이면 결과가 바뀌면 안 됨)도 이걸로
    # 지켜진다.
    exclude_set = pidx_set | (set(exclude_from_neighbors) if exclude_from_neighbors else set())
    neighbor_set = set()
    for c in prompt_idx:
        order = neighbor_order[c].tolist()
        picked = [o for o in order if o not in exclude_set][:neighbor_k]
        neighbor_set.update(picked)
    if neighbor_of_cal and cal_idx:
        # H_cal 자신의 이웃도 추가(같은 exclude_set 재사용 -- H_eval은 이미
        # 그 안에 있으므로 held-out 불변식 안 깨짐). claim16 방향 2.
        for c in cal_idx:
            order = neighbor_order[c].tolist()
            picked = [o for o in order if o not in exclude_set][:neighbor_k]
            neighbor_set.update(picked)
    neighbor_cols = torch.tensor(sorted(neighbor_set), device=device, dtype=torch.long)

    q_head = _find_head(quant_model); q_cap = _CV4Capture(q_head, capture_input=_need_region)
    if use_block_recon:
        _bidx = [bi for bi, b in enumerate(list(quant_model.model)) if list_adaround_convs(b)]
        q_bcap = _BlockCapture(quant_model, _bidx); fp_bcap = _BlockCapture(fp_model, _bidx)
        print(f"[promptcal][joint] 블록 재구성 대상 {len(_bidx)}개, weight={block_recon_weight}")
    else:
        _bidx = []; q_bcap = fp_bcap = None
    q_cv2_cap = _LevelCapture(q_head.cv2) if use_utility else None
    stage2_start = int((1 - utility_stage2_frac) * iters) if use_utility else iters
    smults = [ac.s_mult for ac in ada]
    s0 = [s.detach().clone() for s in smults]
    opt = torch.optim.Adam(smults, lr=lr)
    # learn_alpha (09-24): alpha를 margin 목적함수 아래에서 s_mult와 공동 최적화.
    # s_mult(conv당 스칼라 52개)만으로는 BRECQ/QDrop(alpha 수백만 개 + LSQ delta 52개)과
    # 자유도 차이가 너무 크다는 진단에서 나온 방향. claim14가 실패한 건 1단계 AdaRound가
    # **MSE 목적함수**로 alpha를 움직여 보호 범위 밖으로 손상이 샜기 때문이므로,
    # 같은 alpha를 margin_loss 아래에서 직접 푸는 것은 별개의 시도다.
    # lr은 s_mult(1e-2)와 분리 -- 공식 AdaRound/BRECQ와 같은 1e-3.
    alphas = [ac.alpha for ac in ada] if learn_alpha else []
    opt_a = torch.optim.Adam(alphas, lr=alpha_lr) if learn_alpha else None
    # alpha_bias (09-28, claim21 대응): learn_alpha와 상호배타는 아니지만 용도가
    # 다르다 -- alpha는 원소별 자유도(자유도 폭발로 W4A8에서 붕괴, claim21),
    # alpha_bias는 conv당 스칼라 하나(s_mult와 같은 자릿수)로 자유도를 억제한
    # 대안. 위 adaround.py의 alpha_bias 주석 참고.
    alpha_biases = [ac.alpha_bias for ac in ada] if learn_alpha_bias else []
    opt_ab = torch.optim.Adam(alpha_biases, lr=alpha_bias_lr) if learn_alpha_bias else None
    pidx = torch.tensor(prompt_idx, device=device)
    cal_idx_list = list(cal_idx) if cal_idx else []
    cidx = torch.tensor(cal_idx_list, device=device, dtype=torch.long) if cal_idx_list else None
    # 09-15 버그 수정: confident anchor 선정을 80열(COCO-80) 전체 기준
    # sim_fp.sigmoid().max(-1)으로 하면, FP가 H_eval class를 1등으로 확신한
    # anchor까지 aidx에 섞여 들어가서 그 anchor의 S/H_cal 컬럼 값이 margin_loss에
    # 쓰인다. group_flip(masked Heval_flip)도 정확히 같은 "80열 전체 top-1 ∈
    # H_eval" 기준으로 anchor를 고르므로, 두 계산이 같은 anchor 풀을 공유해
    # 결합이 특히 크다(H_eval이 "최적화에 한 번도 안 쓰인다"는 원칙이 §5.4.1의
    # neighbor 리크와는 별개로 여기서도 새고 있었음). train_cols(S∪H_cal)만
    # 놓고 confidence를 재서 anchor 선정 자체를 H_eval과 무관하게 만든다.
    train_cols = pidx if cidx is None else torch.cat([pidx, cidx])

    if verbose:
        if control_mse:
            print(f"[promptcal-C+neighbor][CONTROL-MSE] {len(fp_sims)} calib, "
                  f"train_cols(S∪H_cal) {len(train_cols)}개, s_mult {len(ada)}개 -- "
                  f"margin_loss/neighbor-hinge 대신 순수 MSE reconstruction")
        else:
            cal_msg = f", cal {len(cal_idx_list)}개(cal_weight={cal_weight})" if cidx is not None else ""
            print(f"[promptcal-C+neighbor] {len(fp_sims)} calib, prompt subset {len(prompt_idx)}개, "
                  f"neighbor {len(neighbor_cols)}개(k={neighbor_k}), s_mult {len(ada)}개, "
                  f"margin(k={k}) neighbor_weight={neighbor_weight}{cal_msg}")

    n = len(calib_tensors)
    for it in range(iters):
        # random_sample (09-27): it % n은 결정적 순환이라 (a) iters=1500 / n=256이면
        # 이미지 0~219는 6번, 220~255는 5번 쓰여 calibration 이미지에 불균등 가중이
        # 걸리고 (b) Adam 모멘텀이 주기 n과 상호작용한다. AdaRound/BRECQ는 09-18에
        # 이미 무작위 추출로 고쳤는데(adaround.py "it % n은 결정적 순환이라 ... 편향이
        # 생긴다") promptcal.py만 빠져 있었다. 기본 False = 기존 동작.
        j = int(torch.randint(0, n, (1,)).item()) if random_sample else it % n
        t = calib_tensors[j].to(device); sim_fp = fp_sims[j]
        prob = sim_fp[:, train_cols].sigmoid(); mp, _ = prob.max(-1); conf = mp > conf_thres
        if conf.sum() == 0:
            continue
        aidx = conf.nonzero(as_tuple=True)[0]
        # per_group_anchors (09-27): anchor는 train_cols(S∪H_cal, 60개)로 선정하는데
        # margin은 S(40개)와 H_cal(20개)에서 따로 계산한다 -- 그래서 FP가 H_cal만
        # 확신하는 anchor에서 S-margin을, S만 확신하는 anchor에서 H_cal-margin을
        # 계산하게 된다(그 anchor의 해당 그룹 top-1은 저확신 class라 "margin을
        # 보존하라"가 의미 없는 신호다). claim1에서 anchor 선정 리크를 고치며
        # train_cols로 통일할 때 항별 대응이 깨진 것으로 보인다.
        # True면 각 margin 항을 그 그룹에서 confident한 anchor에서만 계산한다.
        # H_eval은 어느 쪽에도 안 들어가므로 held-out 불변식은 그대로다.
        if per_group_anchors:
            aidx_s = (sim_fp[:, pidx].sigmoid().max(-1).values > conf_thres).nonzero(as_tuple=True)[0]
            aidx_c = ((sim_fp[:, cidx].sigmoid().max(-1).values > conf_thres).nonzero(as_tuple=True)[0]
                      if cidx is not None else None)
        else:
            aidx_s, aidx_c = aidx, aidx
        q_cap.clear(); opt.zero_grad()
        if q_cv2_cap is not None:
            q_cv2_cap.clear()
        if q_bcap is not None:
            q_bcap.clear(); fp_bcap.clear()
            with torch.no_grad():
                fp_model(t)                       # FP 블록 출력 (매 iteration, no_grad)
        quant_model(t)
        parts = []
        for i in sorted(q_cap.buf):
            B, P, H, W = q_cap.buf[i].shape
            parts.append(q_cap.buf[i].reshape(B, P, H*W))
        sim_q = torch.cat(parts, dim=2)[0].transpose(0, 1)
        if control_mse:
            loss = F.mse_loss(sim_q[aidx][:, train_cols], sim_fp[aidx][:, train_cols])
        else:
            if len(aidx_s) == 0:
                ml = sim_q.sum() * 0.0
            else:
                ml = margin_loss(sim_q[aidx_s][:, pidx], sim_fp[aidx_s][:, pidx], k=k,
                                 boundary_w=boundary_w,
                                 identity_aware=identity_aware_margin, one_sided=margin_one_sided)
            if cidx is not None and cal_weight > 0 and aidx_c is not None and len(aidx_c) > 0:
                # 09-16: cal_weight=0이면 이 항의 기여가 0*ml_cal=0이라 결과는
                # 원래도 같았지만(버그 아님), cal_idx가 이제 항상 전달되므로
                # (claim5-a) cal_weight=0에서도 매 iter margin_loss를 불필요하게
                # 계산하고 있었다. 게이트를 걸어서 그 계산 자체를 스킵한다.
                ml_cal = margin_loss(sim_q[aidx_c][:, cidx], sim_fp[aidx_c][:, cidx], k=k,
                                     boundary_w=boundary_w, one_sided=margin_one_sided,
                                     identity_aware=identity_aware_margin)
                ml = ml + cal_weight * ml_cal
            if asymmetric:
                # 경쟁자가 FP보다 강해지는 방향(sim_q > sim_fp)만 억제. 약해지는
                # 방향은 벌점 없음 -- 우연히 유익한 흔들림(예: seed 0)을 보존.
                # neighbor_cols는 S의 text-embedding 이웃이므로 S에서 confident한
                # anchor가 대응된다(per_group_anchors=False면 aidx_s == aidx).
                an = aidx_s if len(aidx_s) > 0 else aidx
                diff = sim_q[an][:, neighbor_cols] - sim_fp[an][:, neighbor_cols]
                nl = F.relu(diff).pow(2).mean()
            else:
                an = aidx_s if len(aidx_s) > 0 else aidx
                nl = F.mse_loss(sim_q[an][:, neighbor_cols], sim_fp[an][:, neighbor_cols])
            loss = ml + neighbor_weight * nl
            if aux_mse_weight > 0:
                # claim16 방향 3: margin_loss(sparse top-k)에 dense 보조 신호를
                # "더한다"(대체 아님) -- train_cols는 S∪H_cal뿐이라 H_eval 무관.
                aux = F.mse_loss(sim_q[aidx][:, train_cols], sim_fp[aidx][:, train_cols])
                loss = loss + aux_mse_weight * aux
            if scale_reg_weight > 0:
                sr = sum((s - 1.0).pow(2).mean() for s in smults) / len(smults)
                loss = loss + scale_reg_weight * sr
        if use_block_recon:
            terms = []
            for bi in _bidx:
                if bi in q_bcap.buf and bi in fp_bcap.buf:
                    qb, fb = q_bcap.buf[bi], fp_bcap.buf[bi]
                    if qb.shape == fb.shape:
                        terms.append((qb - fb).pow(2).mean() / fb.pow(2).mean().clamp(min=1e-12))
            if terms:
                loss = loss + block_recon_weight * (sum(terms) / len(terms))
        if _need_region:
            rq_raw = q_cap.assemble_input()
            rf_raw = fp_regions[j]
            if use_local_recon:
                # 상대 제곱오차: ||x_q - x_fp||^2 / ||x_fp||^2 (anchor별) 평균.
                num = (rq_raw[aidx] - rf_raw[aidx]).pow(2).sum(-1)
                den = rf_raw[aidx].pow(2).sum(-1).clamp(min=1e-12)
                loss = loss + local_recon_weight * (num / den).mean()
            if use_region_dir:
                # 단위 방향만 보존(크기 무시). 프롬프트 컬럼을 전혀 참조하지 않는다.
                rq = F.normalize(rq_raw, dim=1, p=2); rf = F.normalize(rf_raw, dim=1, p=2)
                rd = (1.0 - (rq[aidx] * rf[aidx]).sum(-1)).mean()
                loss = loss + region_dir_weight * rd
        if use_utility and it >= stage2_start:
            l_thresh, l_box = utility_refinement_terms(
                sim_q, sim_fp, q_cv2_cap.assemble(), fp_cv2s[j], pidx,
                det_thres=det_thres, conf_thres=conf_thres, margin_thres=margin_thres)
            loss = loss + thresh_w * l_thresh + box_w * l_box
        if learn_alpha and it >= alpha_warmup * iters:
            # rounding 정규화: h(alpha)를 0/1로 몰아 hard 확정 때의 점프를 줄인다.
            # 공식 AdaRound와 동일하게 reduction="sum". 09-24에 reduction="mean"으로
            # (margin_loss가 mean 스케일이니 맞춰야 한다고 생각해서) 뒀다가 실측으로
            # 틀렸음을 확인했다 -- mean은 원소당 gradient를 1/N(수백만분의 1)로 나눠서
            # alpha_reg_weight를 5.0까지 올려도 h 수렴률이 10%(=초기 균일분포 그대로)
            # 에서 꿈쩍하지 않았다. 맞춰야 하는 건 손실 "값"의 크기가 아니라 alpha
            # 원소당 gradient 크기다. sum이면 loss 값은 커지지만 s_mult에는 영향이
            # 없고(reg는 s_mult와 무관), alpha 원소마다 O(alpha_reg_weight) 크기의
            # push를 받는다. control_mse 분기와 무관해야 하므로 루프 레벨에 둔다.
            beta = temp_decay(it, iters, alpha_warmup)
            rl = sum(ac.reg_loss(beta, reduction="sum") for ac in ada)
            loss = loss + alpha_reg_weight * rl
        if learn_alpha_bias:
            # L2로 alpha_bias를 0(=BRECQ 원래 반올림) 근처에 붙잡아둔다. 위
            # reg_loss(rectified-sigmoid, 0/1로 밀어냄)와는 목적이 다르다 --
            # bias_reg_loss는 "많이 안 움직이게" 억제하는 정규화다.
            rbl = sum(ac.bias_reg_loss() for ac in ada)
            loss = loss + alpha_bias_reg_weight * rbl
        if opt_a is not None:
            opt_a.zero_grad()
        if opt_ab is not None:
            opt_ab.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(smults, max_norm=1.0)
        if alpha_biases:
            torch.nn.utils.clip_grad_norm_(alpha_biases, max_norm=1.0)
        opt.step()
        if opt_a is not None:
            opt_a.step()
        if opt_ab is not None:
            opt_ab.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            sd = sum(float((s.detach()-s0i).abs().mean()) for s, s0i in zip(smults, s0)) / len(smults)
            smean = sum(float(s.detach().mean()) for s in smults) / len(smults)
            extra = f" scale_reg={float(sr.detach()):.5f}" if scale_reg_weight > 0 else ""
            print(f"  [{it+1}/{iters}] margin_loss={float(ml.detach()):.4f} "
                  f"neighbor_loss={float(nl.detach()):.4f} "
                  f"s_mult 평균={smean:.3f} 변화={sd:.4f}{extra}")

        if eval_hook is not None and (it + 1) % max(1, iters // 10) == 0:
            eval_hook(it + 1, quant_model)

    q_cap.close()
    if q_bcap is not None:
        q_bcap.close(); fp_bcap.close()
    if q_cv2_cap is not None:
        q_cv2_cap.close()
    if learn_alpha or learn_alpha_bias:
        for ac in ada:
            ac.soft = False                      # hard 확정(round 결정 고정)
            ac.round_ste = False                 # STE 해제 -- forward는 이미 hard였다
            ac._hard_weight_cache = None         # bias가 바뀌었으니 캐시 무효화
        fl = [ac.flip_rate() for ac in ada]
        print(f"[promptcal][{'learn_alpha' if learn_alpha else 'alpha_bias'}] hard 확정. "
              f"nearest 대비 평균 flip {sum(fl)/len(fl):.3f}% (max {max(fl):.3f}%) "
              f"-- flip이 0에 가까우면 안 움직인 것")
    if learn_alpha:
        hc = [float(((h_alpha(ac.alpha.detach()) < 0.05) |
                     (h_alpha(ac.alpha.detach()) > 0.95)).float().mean()) * 100 for ac in ada]
        print(f"[promptcal][learn_alpha] h->0/1 수렴 {sum(hc)/len(hc):.0f}%")
    if learn_alpha_bias:
        ab = [float(ac.eff_alpha_bias().detach().abs()) for ac in ada]
        raw = [float(ac.alpha_bias.detach().abs()) for ac in ada]
        n_clamp = sum(1 for r in raw if r > alpha_bias_limit + 1e-9)
        print(f"[promptcal][alpha_bias] |bias| 평균={sum(ab)/len(ab):.4f} "
              f"max={max(ab):.4f} (0=BRECQ 원래 반올림과 동일, 상한={alpha_bias_limit}), "
              f"상한에 걸린 conv {n_clamp}/{len(ada)}개 "
              f"-- 많이 걸리면 상한을 올리거나 reg를 키울 것")
    if verbose:
        tot = sum(float((s.detach()-s0i).abs().sum()) for s, s0i in zip(smults, s0))
        print(f"[promptcal-C+neighbor] 완료 (s_mult 총 변화={tot:.3f})")


def optimize_promptcal_scale_neighbor_utility(quant_model, fp_model, calib_tensors, device,
                                              prompt_idx, iters=1500, lr=1e-2, k=5,
                                              neighbor_k=5, neighbor_weight=1.0,
                                              stage2_frac=0.3, thresh_w=1.0, box_w=0.5,
                                              det_thres=0.25, margin_thres=0.5,
                                              conf_thres=0.25, verbose=True, eval_hook=None,
                                              exclude_from_neighbors=None):
    """
    optimize_promptcal_scale_neighbor(asymmetric 고정) + 논문 §4.3
    Utility-Constrained Refinement(semantic_calib.py의 utility_refinement_terms:
    threshold-crossing + box consistency) 재통합.

    09-06: 41번에서 확정한 asymmetric neighbor preservation은 AP를 6/6 seed에서
    개선시켰지만, 논문 §4.3(utility constraint)은 아직 한 번도 이 조합에
    재통합해서 검증하지 않았다. §6.4 Ablation Study("Semantic Calibration /
    Utility Refinement on/off")가 요구하는 비교이기도 하다.

    30번(semantic_calib.py)의 optimize_semantic_pcal과 동일하게 2단계로 나눈다:
      1단계(전체 iters의 (1-stage2_frac)): margin_loss + neighbor_weight*neighbor_loss만.
      2단계(마지막 stage2_frac): 여기에 thresh_w*l_thresh + box_w*l_box 추가.
        l_thresh: reliable anchor에서 FP가 det_thres를 넘었던 target의 quant
                  확률이 그 밑으로 떨어지지 않게 하는 one-sided hinge.
        l_box   : 같은 anchor에서 cv2(box regression) 출력을 FP와 MSE로 맞춤.
    neighbor_loss는 항상 asymmetric(one-sided hinge)로 고정 -- 41번에서 이미
    symmetric보다 낫다는 게 확인됨.
    """
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = False
        ac.ste = False
        ac.use_smult = True
        ac.alpha.requires_grad_(False)

    fp_head = _find_head(fp_model)
    fp_cap = _CV4Capture(fp_head)
    fp_cv2_cap = _LevelCapture(fp_head.cv2)
    fp_sims, fp_cv2s = [], []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cap.clear(); fp_cv2_cap.clear()
            fp_model(t.to(device))
            parts = []
            for i in sorted(fp_cap.buf):
                B, P, H, W = fp_cap.buf[i].shape
                parts.append(fp_cap.buf[i].reshape(B, P, H*W))
            fp_sims.append(torch.cat(parts, dim=2)[0].transpose(0, 1).detach())
            fp_cv2s.append(fp_cv2_cap.assemble().detach())
    fp_cap.close(); fp_cv2_cap.close()

    txt_feats = get_txt_feats(fp_model).to(device)
    neighbor_order = text_neighbor_order(txt_feats)
    pidx_set = set(prompt_idx)
    exclude_set = pidx_set | (set(exclude_from_neighbors) if exclude_from_neighbors else set())
    neighbor_set = set()
    for c in prompt_idx:
        order = neighbor_order[c].tolist()
        picked = [o for o in order if o not in exclude_set][:neighbor_k]
        neighbor_set.update(picked)
    neighbor_cols = torch.tensor(sorted(neighbor_set), device=device, dtype=torch.long)

    q_head = _find_head(quant_model)
    q_cap = _CV4Capture(q_head)
    q_cv2_cap = _LevelCapture(q_head.cv2)
    smults = [ac.s_mult for ac in ada]
    s0 = [s.detach().clone() for s in smults]
    opt = torch.optim.Adam(smults, lr=lr)
    pidx = torch.tensor(prompt_idx, device=device)

    n = len(calib_tensors)
    stage2_start = int((1 - stage2_frac) * iters)

    if verbose:
        print(f"[promptcal-C+neighbor+utility] {len(fp_sims)} calib, prompt subset "
              f"{len(prompt_idx)}개, neighbor {len(neighbor_cols)}개(k={neighbor_k}), "
              f"s_mult {len(ada)}개, margin(k={k}) neighbor_weight={neighbor_weight}, "
              f"stage2 시작={stage2_start}/{iters}")

    for it in range(iters):
        j = it % n
        t = calib_tensors[j].to(device); sim_fp = fp_sims[j]
        prob = sim_fp.sigmoid(); mp, _ = prob.max(-1); conf = mp > conf_thres
        if conf.sum() == 0:
            continue
        aidx = conf.nonzero(as_tuple=True)[0]
        q_cap.clear(); q_cv2_cap.clear(); opt.zero_grad()
        quant_model(t)
        parts = []
        for i in sorted(q_cap.buf):
            B, P, H, W = q_cap.buf[i].shape
            parts.append(q_cap.buf[i].reshape(B, P, H*W))
        sim_q = torch.cat(parts, dim=2)[0].transpose(0, 1)

        ml = margin_loss(sim_q[aidx][:, pidx], sim_fp[aidx][:, pidx], k=k)
        diff = sim_q[aidx][:, neighbor_cols] - sim_fp[aidx][:, neighbor_cols]
        nl = F.relu(diff).pow(2).mean()
        loss = ml + neighbor_weight * nl

        l_thresh = l_box = None
        if it >= stage2_start:
            cv2_q = q_cv2_cap.assemble()
            l_thresh, l_box = utility_refinement_terms(
                sim_q, sim_fp, cv2_q, fp_cv2s[j], pidx, det_thres=det_thres,
                conf_thres=conf_thres, margin_thres=margin_thres)
            loss = loss + thresh_w * l_thresh + box_w * l_box

        loss.backward()
        torch.nn.utils.clip_grad_norm_(smults, max_norm=1.0)
        opt.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            sd = sum(float((s.detach()-s0i).abs().mean()) for s, s0i in zip(smults, s0)) / len(smults)
            smean = sum(float(s.detach().mean()) for s in smults) / len(smults)
            phase = "utility" if it >= stage2_start else "calib"
            extra = (f" thresh={float(l_thresh.detach()):.4f} box={float(l_box.detach()):.4f}"
                    if l_thresh is not None else "")
            print(f"  [{it+1}/{iters}] ({phase}) margin_loss={float(ml.detach()):.4f} "
                  f"neighbor_loss={float(nl.detach()):.4f} "
                  f"s_mult 평균={smean:.3f} 변화={sd:.4f}{extra}")

        if eval_hook is not None and (it + 1) % max(1, iters // 10) == 0:
            eval_hook(it + 1, quant_model)

    q_cap.close(); q_cv2_cap.close()
    if verbose:
        tot = sum(float((s.detach()-s0i).abs().sum()) for s, s0i in zip(smults, s0))
        print(f"[promptcal-C+neighbor+utility] 완료 (s_mult 총 변화={tot:.3f})")
