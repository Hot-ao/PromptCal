"""
SimilarityHarness — 이 코드베이스의 코어.

역할: 하나의 detector에 대해, region–prompt 유사도 행렬을 캡처하면서 추론을 돌린다.
이 하네스를 FP32 모델과 양자화 모델에 각각 씌워서 나온 두 행렬을 비교하는 것이
Motivation(Sec 3), Main results, Ablation 실험 전부의 공통 연산이다.

설계 원칙:
  - forward hook으로 detection head의 classification logit을 캡처한다.
    YOLO-World에서 이 logit이 곧 region–prompt contrastive 유사도이다.
  - 양자화는 이 하네스 "바깥"에서 모델에 적용된다(quant/에서). 하네스는
    FP든 INT8이든 동일하게 동작해야 한다 → 정밀도에 무지(precision-agnostic)해야 함.

주의(설치 버전 의존): head 출력에서 cls logit을 슬라이스하는 정확한 인덱스는
ultralytics 버전마다 텐서 레이아웃이 다를 수 있다. 아래 `_locate_cls_logits`는
캡처된 텐서의 shape를 보고 검증하도록 되어 있다. 처음 한 번은 02_dump_similarity.py로
shape를 찍어보고 슬라이스가 맞는지 눈으로 확인할 것.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SimilarityRecord:
    """한 이미지에 대한 캡처 결과."""
    image_id: int
    # region–prompt 유사도 (= cls logit). shape [num_regions, num_prompts]
    sim: torch.Tensor
    # 대응하는 박스 (NMS 전/후는 use_postnms 설정에 따름). shape [num_regions, 4]
    boxes: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)


class SimilarityHarness:
    def __init__(self, model, num_prompts: int, device: str = "cuda:0",
                 capture_boxes: bool = True):
        """
        model       : ultralytics YOLOWorld의 .model (nn.Module) — 이미 원하는
                      정밀도(FP/양자화)로 준비된 상태로 넘긴다.
        num_prompts : 현재 vocabulary 크기(예: COCO=80). cls logit 차원 검증에 사용.
        """
        self.model = model.to(device).eval()
        self.num_prompts = num_prompts
        self.device = device
        self.capture_boxes = capture_boxes
        self._buffer = None
        self._handle = self._register_head_hook()

    # ---- hook 등록 -------------------------------------------------------
    def _find_head(self):
        # ultralytics DetectionModel: model.model 은 nn.Sequential, 마지막이 head.
        # YOLO-World면 WorldDetect 계열.
        core = getattr(self.model, "model", self.model)
        head = core[-1] if hasattr(core, "__getitem__") else core.model[-1]
        return head

    def _register_head_hook(self):
        head = self._find_head()

        def hook(_module, _inp, out):
            # out은 head raw 출력. 학습/추론 모드에 따라 형태가 다르다.
            # 목표: [B, num_anchors, 4+num_prompts] 또는 유사 형태에서
            #       cls 파트 [.., num_prompts]를 뽑는다.
            self._buffer = out
        return head.register_forward_hook(hook)

    def _locate_cls_logits(self, raw) -> torch.Tensor:
        """
        캡처된 raw head 출력에서 [num_regions, num_prompts] cls logit을 뽑는다.
        버전별 레이아웃 차이를 흡수하기 위한 유일한 지점. 여기만 맞추면 나머지는 안전.
        """
        t = raw[0] if isinstance(raw, (list, tuple)) else raw
        # 흔한 레이아웃: [B, 4+num_prompts, num_anchors] → transpose 후 cls 슬라이스
        if t.dim() == 3 and t.shape[1] == 4 + self.num_prompts:
            cls = t[:, 4:, :].transpose(1, 2)      # [B, num_anchors, num_prompts]
        elif t.dim() == 3 and t.shape[-1] == 4 + self.num_prompts:
            cls = t[..., 4:]                        # [B, num_anchors, num_prompts]
        else:
            raise RuntimeError(
                f"cls logit 위치를 자동 판별 실패. captured shape={tuple(t.shape)}, "
                f"num_prompts={self.num_prompts}. 02_dump_similarity.py로 shape 확인 후 "
                f"이 함수의 슬라이스를 수정할 것.")
        return cls.squeeze(0)                       # [num_anchors, num_prompts]

    # ---- 추론 + 캡처 ------------------------------------------------------
    @torch.no_grad()
    def run_image(self, img_tensor: torch.Tensor, image_id: int) -> SimilarityRecord:
        self._buffer = None
        _ = self.model(img_tensor.to(self.device))
        if self._buffer is None:
            raise RuntimeError("hook이 head 출력을 잡지 못함. _find_head 확인.")
        sim = self._locate_cls_logits(self._buffer).detach().cpu()
        return SimilarityRecord(image_id=image_id, sim=sim)

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# 두 정밀도의 유사도 행렬을 비교하는 헬퍼(하네스와 metrics를 잇는 얇은 층).
# FP record와 quant record는 "같은 이미지 · 같은 앵커 순서"를 공유해야 한다.
# ---------------------------------------------------------------------------
def paired_run(harness_fp: SimilarityHarness,
               harness_q: SimilarityHarness,
               img_tensor: torch.Tensor, image_id: int):
    rec_fp = harness_fp.run_image(img_tensor, image_id)
    rec_q = harness_q.run_image(img_tensor, image_id)
    return rec_fp, rec_q
