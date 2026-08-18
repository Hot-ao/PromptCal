"""
의미 지표 — FP 유사도 행렬과 양자화 유사도 행렬을 받아 계산한다.
전부 GT 라벨 없이(순위 기반) 계산되는 게 핵심. GT rank만 GT 프롬프트 인덱스를 옵션으로 받는다.

입력 규약:
  sim_fp, sim_q : [num_regions, num_prompts] (같은 region 순서/vocabulary 공유)
region 정렬은 harness가 동일 이미지·동일 앵커로 뽑았다는 전제에 의존한다.
"""

from __future__ import annotations
import torch


def top1_flip_rate(sim_fp: torch.Tensor, sim_q: torch.Tensor,
                   region_mask: torch.Tensor | None = None) -> float:
    """argmax(top-1 프롬프트)가 양자화로 바뀐 region의 비율.
    region_mask: confident region만 볼 때 bool 마스크(선택)."""
    a = sim_fp.argmax(dim=-1)
    b = sim_q.argmax(dim=-1)
    flip = (a != b)
    if region_mask is not None:
        flip = flip[region_mask]
    return flip.float().mean().item() if flip.numel() else 0.0


def gt_rank_shift(sim_fp: torch.Tensor, sim_q: torch.Tensor,
                  gt_prompt_idx: torch.Tensor) -> dict:
    """각 region의 GT 프롬프트가 FP→INT8에서 몇 위 밀렸는지.
    gt_prompt_idx: [num_regions] 각 region의 정답 프롬프트 열 인덱스.
    반환: mean rank(FP), mean rank(INT8), mean Δrank."""
    def rank_of(sim):
        order = sim.argsort(dim=-1, descending=True)   # [R, P]
        # gt가 정렬 순서에서 몇 번째인지
        pos = (order == gt_prompt_idx.unsqueeze(1)).float().argmax(dim=-1)
        return pos  # 0 = 1위
    r_fp = rank_of(sim_fp).float()
    r_q = rank_of(sim_q).float()
    return {
        "rank_fp": r_fp.mean().item(),
        "rank_q": r_q.mean().item(),
        "delta_rank": (r_q - r_fp).mean().item(),
    }


def boundary_inversion_rate(sim_fp: torch.Tensor, sim_q: torch.Tensor,
                            margin: float = 0.05) -> float:
    """FP에서 top-1과 top-2 유사도 차가 margin 이내였던(경계상의) region 중,
    양자화 후 top-1/top-2 순위가 뒤집힌 비율. '의사결정이 불안정한 지점'의 취약성."""
    top2_fp = sim_fp.topk(2, dim=-1)
    near = (top2_fp.values[:, 0] - top2_fp.values[:, 1]) <= margin
    if near.sum() == 0:
        return 0.0
    winner_fp = top2_fp.indices[near, 0]
    winner_q = sim_q[near].argmax(dim=-1)
    return (winner_fp != winner_q).float().mean().item()


# TODO(Sec 3.3): calibration-seen vs held-out prompt로 분리해 위 지표를 각각 계산.
# TODO(Sec 3.4): reconstruction 지표(MSE/cosine)와 위 rank 지표의 상관을 산점도로.
