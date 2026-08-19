# PromptCal-PTQ 실험 진행 정리 — 3단계 (고정 프롬프트 손상)

작성 시점: naive W8A8 손상 측정 완료(국면 B). 프롬프트 축 실험(국면 C) 진입 직전.
목적: 논문 Sec 3(Motivation)의 핵심 관찰을 데이터로 확정하고 기록으로 남긴다.

---

## 요약 (한 줄)

양자화는 총량 지표(AP)로는 거의 무해해 보이지만, region-prompt 의사결정의
"경계"에 손상이 집중된다. 단, 이는 프롬프트 고정(COCO 80) 조건에서의 기준선이며,
프롬프트 축을 움직이는 실험(국면 C)이 논문의 고유 기여가 된다.

---

## 실험 설정

| 항목 | 값 |
|---|---|
| 모델 | yolov8s-worldv2 (fused) |
| 양자화 | naive W8A8, min-max, fuse 후 vision Conv 68개 래핑, DFL 제외 |
| weight | per-output-channel symmetric int8 |
| activation | per-tensor asymmetric int8, calib 64장 |
| 데이터 | COCO val2017, 프롬프트 = COCO 80 클래스 (고정) |
| calib / eval | 앞 64장 / 다음 200장 (분리) |
| 측정 도구 | SimilarityHarness (cv4 캡처, [5880 x 80]) |

---

## 결과 1 — AP는 거의 안 떨어진다 (03b)

pycocotools 기준:

| | FP32 | W8A8 | 하락 |
|---|---|---|---|
| AP (all) | 37.8 | 37.3 | **-0.5** |
| AP50 | 52.2 | 51.5 | -0.7 |
| AP small | 21.5 | 21.5 | ~0 |
| AP medium | 42.2 | 41.9 | -0.3 |
| AP large | 50.1 | 49.7 | -0.4 |

양자화는 fuse 후 실제 배포 weight에 제대로 적용됨(a_scale 정상). 그럼에도 AP 손실 미미.
→ "reconstruction/AP 관점에선 양자화가 거의 무해해 보인다."

---

## 결과 2 — 그러나 경계 의사결정은 무너진다 (03, 03c)

**단일 지표 (03):**
- Top-1 flip (confident, maxprob>0.25): 0.73%
- Top-1 flip (all anchors): 30.06%
- Boundary inversion (margin=0.5): 46.50%

**confidence 구간별 (03c-A):**
maxprob<0.05 구간이 전체 앵커의 98.7%를 차지하고 flip 30.4%.
→ all-anchor 30%는 거의 전부 "배경/저확신 앵커"의 흔들림(노이즈). 탐지로 이어지지 않음.
→ confident 구간일수록 flip 단조 감소(0.9+ 구간 0.0%). "확실한 판정은 양자화가 안 건드림."

**margin 구간별 (03c-B) — 핵심 곡선:**

| top1-top2 margin (logit) | flip |
|---|---|
| 0.0~0.1 (극경계) | 60.5% |
| 0.1~0.25 | 49.5% |
| 0.25~0.5 | 35.9% |
| 0.5~1.0 | 24.2% |
| 1.0~2.0 | 15.0% |
| 2.0~4.0 | 12.9% |
| 4.0~8.0 | 11.6% |
| 8.0+ (압도적) | 0.0% |

margin이 작을수록(경계일수록) flip 급증, 클수록 0으로 수렴하는 완벽한 단조 감소.
margin<0.25 경계 region이 전체의 23.6%, 그중 절반 이상이 뒤집힘.
→ **"양자화 손상이 semantic decision boundary에 집중된다"가 정량 확정.**

---

## 세 결과의 일관성

- AP -0.5 (총량 멀쩡) + confident flip 0.73% (확실한 판정 유지) + margin 곡선(경계만 붕괴)
- 세 수치가 모순이 아니라 한 방향을 가리킴:
  **양자화는 확실한 의사결정은 보존하고, 애매한 경계 의사결정만 대규모로 흔든다.**
- AP가 안 떨어지는 이유도 설명됨: 손상이 총량에 안 잡히는 경계에 국한되기 때문.

---

## 논문 Sec 3에의 함의

- Sec 3.2 (Precision-Induced Semantic Damage): 양자화가 region-prompt decision을 손상.
- Sec 3.4 (Reconstruction-Utility Misalignment): AP(총량)와 decision 손상의 괴리 =
  "rank/decision preservation은 reconstruction/AP로 보장되지 않는다"의 직접 증거.
- 대표 그림 후보: margin 구간별 flip 곡선(03c-B).

---

## 이 결과의 한계 (→ 국면 C 필요)

**모든 측정이 프롬프트 고정(COCO 80) 조건에서 나왔다.**
이는 "프롬프트가 안 변해도 경계가 이만큼 무너진다"는 기준선일 뿐,
OVOD 고유성(배포 시 프롬프트가 바뀐다)은 아직 건드리지 않았다.

논문의 두 축은 (1) OVOD 특유성(프롬프트 축 변화) x (2) region-prompt 유사도 의사결정이며,
현재 결과는 축(2)의 고정-프롬프트 단면에 해당. 축(1)과 결합해야 논문 고유 기여가 성립.

**다음(국면 C):**
- held-out 프롬프트: calibration에 없던 프롬프트에서의 손상
- semantic hard-negative: 의미적으로 경쟁하는 프롬프트에서의 class substitution
- LVIS 도입: vocabulary shift 환경
- 기대: 위 margin 곡선이 프롬프트 축을 움직이면 더 심하게 악화됨을 보임

---

## 코드베이스 상태

| 파일 | 상태 |
|---|---|
| scripts/00_setup_coco.py | 완료 |
| scripts/01_reproduce_baseline.py | 완료 |
| scripts/02_dump_similarity.py | 완료 |
| scripts/03_motivation_damage.py | 완료 (fuse 후 양자화) |
| scripts/03b_quant_ap.py | 완료 |
| scripts/03c_confidence_curve.py | 완료 |
| src/harness.py | 확정 (cv4 캡처) |
| src/quant/fake_quant.py, quant_model.py | 완료 (naive W8A8) |
| src/metrics/semantic_metrics.py | top1_flip / boundary_inversion 사용 중, gt_rank_shift 미사용 |
