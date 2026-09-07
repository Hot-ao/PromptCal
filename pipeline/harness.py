"""
SimilarityHarness — 이 코드베이스의 코어. (2단계 검증 완료 버전)

역할: 하나의 detector에 대해, region-prompt 유사도 행렬을 캡처하면서 추론을 돌린다.
이 하네스를 FP32 모델과 양자화 모델에 각각 씌워서 나온 두 행렬을 비교하는 것이
Motivation(Sec 3), Main results, Ablation 실험 전부의 공통 연산이다.

캡처 방식 (2단계 진단으로 확정):
  WorldDetect.cv4 = 레벨별 contrastive head. 각 레벨이 [B, num_prompts, H, W]의
  pre-sigmoid 유사도 맵을 낸다. 이를 flatten+concat하면 [anchors, num_prompts]
  region-prompt 유사도 행렬이 된다. head 출력 튜플을 파싱하는 것보다 모호함이 없고
  (검증됨: 전역 argmax == 모델 정식 예측), pre-sigmoid logit이라 margin 기반 지표에 유리.

  검증 결과(yolov8s-worldv2, ultralytics 8.4.121):
    cv4_0 [1,80,56,80] + cv4_1 [1,80,28,40] + cv4_2 [1,80,14,20] -> [5880, 80]
    value range ~ [-53, +2.4] (pre-sigmoid), 전역 최고 앵커 클래스 == 최고신뢰 탐지 클래스.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch


@dataclass
class SimilarityRecord:
    """한 이미지에 대한 캡처 결과."""
    image_id: int
    # region-prompt 유사도(pre-sigmoid logit). shape [num_regions, num_prompts]
    sim: torch.Tensor
    meta: dict = field(default_factory=dict)


class SimilarityHarness:
    def __init__(self, model, device: str = "cuda:0"):
        """
        model : ultralytics YOLOWorld의 .model (nn.Module) — 이미 원하는
                정밀도(FP/양자화)로 준비된 상태로 넘긴다. precision-agnostic.
        """
        self.model = model.to(device).eval()
        self.device = device
        self._head = self._find_head()
        if not hasattr(self._head, "cv4"):
            raise RuntimeError(
                f"head({type(self._head).__name__})에 cv4가 없음. "
                f"reparameterize된 모델일 수 있음 -> 02_dump_similarity.py로 재진단 필요.")
        self._level_buf: dict = {}
        self._handles = self._register_cv4_hooks()

    def _find_head(self):
        core = getattr(self.model, "model", self.model)
        return core[-1] if hasattr(core, "__getitem__") else core.model[-1]

    def _register_cv4_hooks(self):
        handles = []
        for i, sub in enumerate(self._head.cv4):
            def make(idx):
                def hook(_m, _inp, out):
                    self._level_buf[idx] = out  # [B, num_prompts, H, W]
                return hook
            handles.append(sub.register_forward_hook(make(i)))
        return handles

    def _assemble_sim(self) -> torch.Tensor:
        """레벨별 [B, P, H, W] 버퍼를 [anchors, P]로 조립."""
        parts = []
        for i in sorted(self._level_buf):
            t = self._level_buf[i]
            B, P, H, W = t.shape
            parts.append(t.reshape(B, P, H * W))
        cat = torch.cat(parts, dim=2)               # [B, P, total_anchors]
        return cat[0].transpose(0, 1).contiguous()  # [total_anchors, P]

    @torch.no_grad()
    def run_image(self, img_tensor: torch.Tensor, image_id: int) -> SimilarityRecord:
        """
        img_tensor: 전처리된 [B, 3, H, W] 입력 텐서(letterbox 640 등).
                    FP/양자화 모델이 같은 앵커 순서를 공유하도록 전처리를 통일할 것.
        """
        self._level_buf.clear()
        _ = self.model(img_tensor.to(self.device))
        if not self._level_buf:
            raise RuntimeError("cv4 hook이 출력을 잡지 못함. head 구조 확인.")
        sim = self._assemble_sim().detach().cpu()
        return SimilarityRecord(image_id=image_id, sim=sim)

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def paired_run(harness_fp, harness_q, img_tensor, image_id):
    """FP record와 quant record는 같은 이미지·같은 앵커 순서를 공유해야 한다."""
    rec_fp = harness_fp.run_image(img_tensor, image_id)
    rec_q = harness_q.run_image(img_tensor, image_id)
    return rec_fp, rec_q
