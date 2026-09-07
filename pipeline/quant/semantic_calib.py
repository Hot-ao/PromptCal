"""
Semantic Calibration + Utility-Constrained Refinement (paper §Methodology).

v1/v2(promptcal.py, promptcal_v2.py)와의 핵심 차이:
  - v1/v2는 고정된 global prompt 부분집합(S+H_cal 60개)의 decision을 통째로 맞추려 했고,
    특정 60개 class identity에 rounding capacity를 재배분하는 부작용(negative transfer,
    scripts/29_diag_transfer.py로 확인한 Case B)이 있었다.
  - 여기서는 anchor마다 "target + text-embedding상 가장 가까운 경쟁자"로 국소 경쟁집합을
    동적으로 구성한다(Prompt Selection). 특정 class index를 암기하는 게 아니라 "가까운
    경쟁자와의 margin을 지켜라"라는 class-agnostic한 절차를 학습하므로, calibration에서
    본 것과 다른 class 조합에도 같은 절차가 적용될 수 있다는 것이 핵심 가설이다.
  - local reconstruction(cv3 region-embedding MSE, prompt 무관)을 semantic 항과 함께
    최적화해 capacity가 마음대로 재배분되는 것을 억제한다.

두 단계:
  1. Semantic Calibration : region filtering(confidence+margin) + 동적 경쟁 프롬프트
     선택 + (local reconstruction + margin) objective.
  2. Utility-Constrained Refinement : threshold-crossing(탐지 여부 자체가 바뀌는 것)과
     box consistency(cv2 회귀 출력)까지 추가로 제약. stage 1의 결과를 warm-start로 이어받아
     학습 후반부(stage2_frac)에 활성화.

최적화 대상은 v1과 동일하게 AdaRound rounding parameter(alpha)만 사용한다(가산 모듈,
weight는 고정). scale/clip 학습(LSQ)은 추후 확장 지점으로 남겨둔다.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .adaround import list_adaround_convs
from .pdquant import _find_head


class _LevelCapture:
    """cv2/cv3/cv4 등 레벨별 ModuleList의 출력을 캡처해 [anchors, C]로 조립."""

    def __init__(self, module_list):
        self.buf = {}
        self.handles = []
        for i, sub in enumerate(module_list):
            def make(idx):
                def hook(_m, _inp, out):
                    self.buf[idx] = out
                return hook
            self.handles.append(sub.register_forward_hook(make(i)))

    def clear(self):
        self.buf = {}

    def assemble(self):
        parts = []
        for i in sorted(self.buf):
            t = self.buf[i]
            B, C, H, W = t.shape
            parts.append(t.reshape(B, C, H * W))
        cat = torch.cat(parts, dim=2)                 # [B, C, A]
        return cat[0].transpose(0, 1).contiguous()     # [A, C]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_txt_feats(model_module):
    """WorldModel.txt_feats: [1, P, 512], L2-normalized (fuse 전/후 모두 존재).
    이건 FP text encoder 출력이라 양자화 대상이 아니고, 항상 FP 모델에서 뽑는다."""
    return model_module.txt_feats[0].detach()


def text_neighbor_order(txt_feats):
    """txt_feats: [P, 512] (L2-normalized). 반환: [P, P] 각 class별 유사도 내림차순
    인덱스(자기 자신은 맨 뒤로 밀림)."""
    sim = txt_feats @ txt_feats.T
    sim = sim.clone()
    sim.fill_diagonal_(-2.0)
    return sim.argsort(dim=-1, descending=True)


def semantic_calibration_loss(sim_q, sim_fp, cv3_q, cv3_fp, pidx, neighbor_order,
                              k_near=5, k_far=3, conf_thres=0.25, margin_thres=0.5,
                              near_w=3.0, far_w=1.0, rng=None):
    """
    Region Filtering + Prompt Selection + Semantic Objective, 한 이미지 forward에 대해.

    sim_q, sim_fp : [A, P_full] pre-sigmoid cv4 유사도. sim_q는 grad 유지.
    cv3_q, cv3_fp : [A, 512] region embedding(cv3 출력, cv4 입력). local reconstruction 대상.
    pidx          : LongTensor. calibration에 쓸 수 있는 prompt pool(S+H_cal). 이 밖은 절대 사용 안 함.
    neighbor_order: [P_full, P_full] text embedding 유사도 내림차순 인덱스(전체 class에 대해 1회 계산).

    반환: (l_recon, l_sem, n_reliable_anchor)
    """
    device = sim_q.device
    pidx_list = pidx.tolist()
    pidx_set = set(pidx_list)

    # ---------------- Region Filtering ----------------
    sub_fp = sim_fp[:, pidx]                            # [A, |pool|]
    prob = sub_fp.sigmoid()
    top2 = sub_fp.topk(min(2, sub_fp.shape[-1]), dim=-1).values
    maxp = prob.max(-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
    reliable = (maxp > conf_thres) & (margin > margin_thres)

    if reliable.sum() == 0:
        zero = sim_q.sum() * 0.0
        return zero, zero, 0

    aidx = reliable.nonzero(as_tuple=True)[0]
    local_argmax = sub_fp[aidx].argmax(-1)
    targets = pidx[local_argmax]                        # 실제 class index (0..P_full-1)

    # ---------------- local reconstruction (prompt 무관) ----------------
    l_recon = F.mse_loss(cv3_q[aidx], cv3_fp[aidx])

    # ---------------- Prompt Selection + Semantic Objective ----------------
    sem_terms = []
    for t in targets.unique().tolist():
        a_t = aidx[targets == t]
        order_t = [c for c in neighbor_order[t].tolist() if c in pidx_set and c != t]
        near = order_t[:k_near]
        far_pool = order_t[k_near:]
        far = []
        if far_pool and k_far > 0:
            n_far = min(k_far, len(far_pool))
            if rng is not None:
                far = list(rng.choice(far_pool, size=n_far, replace=False))
            else:
                far = far_pool[-n_far:]
        comp = near + far
        if not comp:
            continue
        comp_t = torch.tensor(comp, device=device, dtype=torch.long)
        w = torch.tensor([near_w] * len(near) + [far_w] * len(far), device=device)

        m_fp = sim_fp[a_t][:, [t]] - sim_fp[a_t][:, comp_t]     # [n_a, n_comp]
        m_q = sim_q[a_t][:, [t]] - sim_q[a_t][:, comp_t]
        sem_terms.append((((m_q - m_fp) ** 2) * w).mean() * len(a_t))

    l_sem = (sum(sem_terms) / len(aidx)) if sem_terms else sim_q.sum() * 0.0
    return l_recon, l_sem, int(aidx.numel())


def utility_refinement_terms(sim_q, sim_fp, cv2_q, cv2_fp, pidx, det_thres=0.25,
                             conf_thres=0.25, margin_thres=0.5):
    """
    Utility Constraints: threshold-crossing + box consistency.
    reliable anchor(§semantic_calibration_loss와 동일 기준)에서만 계산.
    """
    device = sim_q.device
    sub_fp = sim_fp[:, pidx]
    prob_fp = sub_fp.sigmoid()
    top2 = sub_fp.topk(min(2, sub_fp.shape[-1]), dim=-1).values
    maxp = prob_fp.max(-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[-1] > 1 else top2[:, 0]
    reliable = (maxp > conf_thres) & (margin > margin_thres)
    if reliable.sum() == 0:
        z = sim_q.sum() * 0.0
        return z, z

    aidx = reliable.nonzero(as_tuple=True)[0]
    local_argmax = sub_fp[aidx].argmax(-1)
    targets = pidx[local_argmax]

    # threshold crossing: FP가 det_thres 위였던 target column의 quant 확률이
    # 그 밑으로 떨어지지 않도록 one-sided hinge.
    q_prob_t = sim_q[aidx, targets].sigmoid()
    fp_positive = prob_fp[torch.arange(len(aidx), device=device), local_argmax] > det_thres
    if fp_positive.any():
        l_thresh = F.relu(det_thres - q_prob_t[fp_positive]).pow(2).mean()
    else:
        l_thresh = sim_q.sum() * 0.0

    # box consistency: reliable anchor에서 cv2(box regression 원시 출력) MSE.
    l_box = F.mse_loss(cv2_q[aidx], cv2_fp[aidx])
    return l_thresh, l_box


def optimize_semantic_pcal(quant_model, fp_model, calib_tensors, device, pidx,
                           iters=1500, lr=3e-3, reg_weight=0.1,
                           k_near=5, k_far=3, conf_thres=0.25, margin_thres=0.5,
                           near_w=3.0, far_w=1.0, recon_w=1.0,
                           stage2_frac=0.3, thresh_w=1.0, box_w=0.5, det_thres=0.25,
                           seed=0, verbose=True, post_train_hook=None):
    """Semantic Calibration(전반) -> Utility-Constrained Refinement(후반 stage2_frac)."""
    ada = list_adaround_convs(quant_model)
    for ac in ada:
        ac.soft = True
        ac.ste = True

    fp_head = _find_head(fp_model)
    fp_cv3, fp_cv4, fp_cv2 = _LevelCapture(fp_head.cv3), _LevelCapture(fp_head.cv4), _LevelCapture(fp_head.cv2)

    fp_sims, fp_cv3s, fp_cv2s = [], [], []
    with torch.no_grad():
        for t in calib_tensors:
            fp_cv3.clear(); fp_cv4.clear(); fp_cv2.clear()
            fp_model(t.to(device))
            fp_sims.append(fp_cv4.assemble().detach())
            fp_cv3s.append(fp_cv3.assemble().detach())
            fp_cv2s.append(fp_cv2.assemble().detach())
    fp_cv3.close(); fp_cv4.close(); fp_cv2.close()

    txt_feats = get_txt_feats(fp_model).to(device)
    neighbor_order = text_neighbor_order(txt_feats)

    q_head = _find_head(quant_model)
    q_cv3, q_cv4, q_cv2 = _LevelCapture(q_head.cv3), _LevelCapture(q_head.cv4), _LevelCapture(q_head.cv2)

    alphas = [ac.alpha for ac in ada]
    opt = torch.optim.Adam(alphas, lr=lr)
    pidx_t = torch.tensor(pidx, device=device, dtype=torch.long)
    rng = np.random.default_rng(seed)

    n = len(calib_tensors)
    stage2_start = int((1 - stage2_frac) * iters)

    if verbose:
        print(f"[semantic_pcal] {n} calib, pool {len(pidx)}개, alpha {len(ada)}개, "
              f"k_near={k_near} k_far={k_far}, stage2 시작={stage2_start}/{iters}")

    for it in range(iters):
        j = it % n
        t = calib_tensors[j].to(device)
        sim_fp, cv3_fp, cv2_fp = fp_sims[j], fp_cv3s[j], fp_cv2s[j]

        q_cv3.clear(); q_cv4.clear(); q_cv2.clear()
        opt.zero_grad()
        quant_model(t)
        sim_q = q_cv4.assemble()
        cv3_q = q_cv3.assemble()

        l_recon, l_sem, n_anchor = semantic_calibration_loss(
            sim_q, sim_fp, cv3_q, cv3_fp, pidx_t, neighbor_order,
            k_near=k_near, k_far=k_far, conf_thres=conf_thres,
            margin_thres=margin_thres, near_w=near_w, far_w=far_w, rng=rng)

        beta = max(2.0, 20.0 * (1 - it / max(iters, 1)))
        reg = sum(ac.reg_loss(beta, reduction="mean") for ac in ada) / len(ada)

        loss = recon_w * l_recon + l_sem + reg_weight * reg
        l_thresh = l_box = None
        if it >= stage2_start:
            cv2_q = q_cv2.assemble()
            l_thresh, l_box = utility_refinement_terms(
                sim_q, sim_fp, cv2_q, cv2_fp, pidx_t, det_thres=det_thres,
                conf_thres=conf_thres, margin_thres=margin_thres)
            loss = loss + thresh_w * l_thresh + box_w * l_box

        loss.backward()
        torch.nn.utils.clip_grad_norm_(alphas, max_norm=1.0)
        opt.step()

        if verbose and (it + 1) % max(1, iters // 10) == 0:
            from .adaround import h_alpha
            hc = sum(float(((h_alpha(a) < 0.05) | (h_alpha(a) > 0.95)).float().mean())
                    for a in alphas) / len(alphas) * 100
            phase = "utility" if it >= stage2_start else "calib"
            extra = (f" thresh={float(l_thresh.detach()):.4f} box={float(l_box.detach()):.4f}"
                    if l_thresh is not None else "")
            print(f"  [{it+1}/{iters}] ({phase}) recon={float(l_recon.detach()):.4f} "
                  f"sem={float(l_sem.detach()):.4f} reg={float(reg.detach()):.4f} "
                  f"n_anchor={n_anchor} h→0/1={hc:.0f}%{extra}")

    q_cv3.close(); q_cv4.close(); q_cv2.close()

    if post_train_hook is not None:
        post_train_hook(quant_model)

    for ac in ada:
        ac.soft = False
        ac.ste = False

    if verbose:
        print("[semantic_pcal] 완료 (hard rounding 확정)")
