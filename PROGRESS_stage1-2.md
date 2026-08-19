# PromptCal-PTQ 실험 진행 정리 — 1~2단계

작성 시점 기준: FP32 baseline 재현 + SimilarityHarness 검증 완료.
목적은 양자화(3단계)로 넘어가기 전, 토대가 검증되었음을 기록으로 남기는 것.

---

## 요약 (한 줄)

FP32 YOLO-World가 우리 환경에서 공식 수치를 재현했고(1단계), 그 위에서 region-prompt
유사도 행렬을 정확히 캡처하는 측정 도구를 검증(2단계)했다. 논문의 모든 양자화 손상
실험이 딛고 설 두 토대가 놓였다.

---

## 실험 환경

| 항목 | 값 |
|---|---|
| 서버 | Ubuntu, 8-GPU (L40S 46GB × 4, RTX 4000 Ada 20GB × 4) |
| 드라이버 / CUDA | 560.35.05 / 최대 12.6 |
| 환경 | conda `promptcal`, Python 3.11 |
| torch | 2.13.0+cu126 (use-cuda 12.6 ↔ cu126 매칭) |
| ultralytics | 8.4.121 |
| 모델 | yolov8s-worldv2 (fused, 12.7M params, 31.9 GFLOPs) |
| 데이터 | COCO val2017 (5000장), 기존 로컬본을 symlink+라벨변환으로 재사용 |
| 실행 GPU | RTX 4000 Ada (GPU 0) — baseline엔 20GB로 충분 |

데이터는 재다운로드 없이 `/data/taeho/coco_datasets`를 `coco_ultra/`로 브리지
(이미지 symlink, COCO json → YOLO 라벨 변환 cls91to80=True, pycocotools용 annotation symlink).

---

## 1단계 — FP32 Baseline 재현

**목적:** FP32 원본 모델이 우리 환경에서 정상 동작하며 공식 수치를 내는지 확인(기준점).

**결과 (COCO val2017, zero-shot, mAP50-95):**

| 평가 방식 | mAP50-95 | mAP50 | mAP75 |
|---|---|---|---|
| Ultralytics 내부 metric | 37.17 | 51.49 | 40.44 |
| pycocotools (공식) | **37.8** | 52.2 | 41.0 |
| 공식 리더보드 목표 | 37.7 | 52.2 | 41.0 |

**판정: 재현 성공.** 공식 리더보드는 pycocotools 기준이며, 우리 pycocotools 값
37.8이 목표 37.7과 일치(오히려 +0.1). Ultralytics 내부 metric(37.17)과 pycocotools의
0.3~0.6 차이는 매칭/보간 방식 차이로 정상 범위. (스크립트의 ±0.3 tol이 이 차이를 못
감안해 MISMATCH로 표기했을 뿐, 실제 재현은 정확함.)

**부수 관찰 — size별 AP (pycocotools 자동 산출):**

| 영역 | AP |
|---|---|
| small | 0.215 |
| medium | 0.422 |
| large | 0.501 |

작은 객체 AP가 큰 객체의 절반 이하. 향후 양자화 손상을 크기별로 볼 때(ScaleGuard 논지와
연결) 활용 가능한 기준선. 현 단계에서는 "이 축이 살아있음"만 확인.

---

## 2단계 — SimilarityHarness 검증

**목적:** region-prompt 유사도 행렬을 뽑는 측정 도구가 정확한지 검증(온도계 눈금 보정).
논문의 모든 지표(Top-1 flip, GT rank, boundary inversion)가 이 행렬 위에서 계산되므로,
여기가 어긋나면 이후 손상 수치가 전부 무효가 됨.

**핵심 발견 — 캡처 지점:**
WorldDetect head의 `cv4`(레벨별 contrastive head)가 fused 상태에서도 살아있으며,
각 레벨이 `[B, 80, H, W]` pre-sigmoid 유사도 맵을 출력. 이를 flatten+concat하여
`[anchors, 80]` 행렬로 조립.

**검증 결과 (첫 val 이미지 000000000139):**

| 확인 항목 | 결과 |
|---|---|
| head 타입 | WorldDetect (children: cv2, cv3, dfl, cv4) |
| cv4 존재 | O (fused에도 유지) |
| 조립된 행렬 shape | [5880, 80] = [anchors, num_prompts] ✓ |
| 레벨 구성 | 56×80 + 28×40 + 14×20 = 5880 anchors |
| value range | [-53.2, +2.4] → 명백한 pre-sigmoid logit ✓ |
| 전역 최고점 앵커 클래스 | 62 (tv), score 0.919 |
| 모델 정식 예측 최고신뢰 | 62 (tv), conf 0.919 |
| **argmax 일치** | **OK ✅ 완전 일치** |

**판정: 검증 성공.** hook이 가로챈 행렬의 전역 argmax가 모델 정식 예측과 정확히 일치
(클래스·score 모두). 우리가 캡처하는 행렬 = 이 detector의 실제 의사결정임이 증명됨.

**설계 확정:** 초기 harness는 head 출력 튜플을 슬라이스하는 방식이었으나(버전 의존적),
검증 결과 cv4 직접 캡처가 명백히 우월(모호함 없음 + pre-sigmoid logit 보존)하여
`SimilarityHarness`를 cv4 기반으로 확정. `head_out[0]`은 `[1, 84, 5880]`(4 box + 80 cls,
post-sigmoid)로, margin 기반 지표에는 pre-sigmoid인 cv4가 유리하다는 점도 재확인.

---

## 현재 코드베이스 상태

| 파일 | 상태 |
|---|---|
| `scripts/00_setup_coco.py` | 완료 — 로컬 COCO 브리지 |
| `scripts/01_reproduce_baseline.py` | 완료 — FP32 재현 (save_json 켜짐) |
| `scripts/02_dump_similarity.py` | 완료 — 진단/검증 |
| `src/harness.py` | **확정** — cv4 기반 캡처 |
| `src/metrics/semantic_metrics.py` | 골격 — top1_flip / gt_rank_shift / boundary_inversion |
| `src/quant/` | 비어있음 — 3단계에서 채움 |

---

## 다음 (3단계) — Naive W8A8 손상 측정

FP32 모델에 min-max W8A8 양자화를 적용하고, 같은 이미지들에 대해 FP vs 양자화
유사도 행렬을 `SimilarityHarness`로 뽑아 Top-1 flip / GT rank / boundary inversion을
계산. "AP는 소폭 하락하나 region-prompt 의사결정은 크게 손상"을 입증 → 논문 Sec 3.

**미결 설계 결정:**
- 양자화 범위: vision backbone/neck/head만 (text는 offline 임베딩이라 제외)
- 진행 방식: W8만 먼저 → W8A8, 또는 처음부터 W8A8 (선택 필요)
- 구현: PyTorch 네이티브 fake-quant vs 직접 min-max observer (선택 필요)
