# PromptCal-PTQ 실험 진행 정리 — 국면 C (프롬프트 축 손상) [완료]

작성 시점: C-1(substitution), C-2(held-out), C-3(LVIS) 모두 확정.
목적: 논문의 OVOD 고유 축 — "양자화 손상이 프롬프트 축에서 의미적으로 발현된다" — 을 기록.

---

## 요약 (한 줄)

양자화 손상은 semantic decision boundary(작은 margin)에서, 의미적으로 가까운 프롬프트
방향으로, 프롬프트가 낯설거나(held-out) 촘촘할수록(LVIS) 심하게 발현된다. 세 실험이
서로 다른 조건에서 동일한 margin 메커니즘(국면 B)으로 이 현상을 입증한다.

---

## 공통 설정

- 모델: yolov8s-worldv2 (fused), 양자화: naive W8A8 (fuse 후 vision Conv 래핑)
- calibration: 64장 (vision-only이므로 프롬프트 무관 → 모든 vocabulary에 재사용)
- 측정: SimilarityHarness cv4 캡처, confident region = FP maxprob>0.25
- 전부 GT-free (FP top-1을 pseudo-GT로 사용)

---

## C-1 — Semantic Class Substitution (COCO)

양자화 top-1 flip이 random이 아니라 의미적 이웃으로 향하는가?

| | 200장 | 2000장(확정) |
|---|---|---|
| confident regions | 8,940 | 90,047 |
| top-1 flip | 43 (0.5%) | 483 (0.5%) |
| flip → 이웃(top-5) | 76.7% | **69.2%** |
| 편향 배수(관측/우연 6.3%) | 12.1x | **10.9x** |

표본 10배로도 편향 유지 → 확정. "양자화 flip의 69%가 의미적 이웃으로, 우연의 10.9배."

---

## C-2 — Held-out Vocabulary (COCO, multi-seed)

calibration/vocab에 없던 프롬프트(정답 단어 부재)에서 손상이 악화되는가?
설계: COCO 80을 K/H로 분할, FP top-1이 H인 region에서 H열 마스킹 후 재판정.

| seed | targets | seen flip | held-out flip | mean margin (seen→ho) |
|---|---|---|---|---|
| 0 | 29,044 | 0.67% | 5.96% | 5.25 → 1.81 |
| 1 | 29,290 | 0.92% | 6.88% | 4.88 → 1.70 |
| 2 | 25,479 | 0.76% | 5.93% | 4.86 → 1.72 |
| 3 | 63,682 | 0.38% | 9.19% | 7.10 → 1.12 |
| 4 | 63,907 | 0.41% | 8.01% | 7.02 → 1.34 |
| **mean** | | **0.63%** | **7.19%** | margin 항상 축소 |

**핵심 지표는 held-out flip 절대값 7.2%(seen 0.6% 대비).** 배수(mean 13.6x, std 7.0)는
분모(seen)에 민감해 논문 헤드라인 부적합.

**편차 원인 규명(05b 진단):** seed 3,4는 held-out에 'person'이 포함되어 target의 58%를
person이 차지(person seen-flip 0.01%, 초확신). 분모(seen)가 붕괴해 배수만 폭증.
→ held-out 손상 자체는 견고(held-out flip 6~9%), 배수는 지표 한계.

메커니즘: 정답 단어 제거 → margin 축소(예: 5.25→1.81) → 국면 B 곡선 따라 flip 증폭.

---

## C-3 — LVIS 1203 vs COCO 80 (controlled)

촘촘한 vocabulary에서 손상이 증폭되는가?
공정성 확인(06b): 앵커 수 8400 동일(프롬프트 수 무관), 같은 region·양자화, 프롬프트만 교체.

| | COCO-80 | LVIS-1203 |
|---|---|---|
| flip rate | 0.51% | **4.28%** (8.4x↑) |
| mean margin | 6.41 | **1.29** (5x↓) |
| small-margin(<0.5) 비율 | 1.7% | **28.5%** (17x↑) |
| flip→이웃 (k=5) | 76.8% | 39.7% |
| flip→이웃 (k=20) | 92.0% | **52.0%** |
| 편향 배수 (k=20) | 3.6x | **31.3x** |

- 손상 증폭 확정: flip 8.4배↑, margin 5배↓, small-margin 17배↑.
- substitution 견고: k=5에선 이웃 비율이 낮아 보이나(LVIS top-5가 너무 좁음), k=20에서
  52%로 회복. 편향 배수 31.3x로 오히려 COCO보다 강한 의미적 쏠림.
- 주의: LVIS 이웃 정의(k)에 민감. 배수(우연 대비)가 절대 비율보다 견고한 지표.
- 형식 주의: LVIS 프롬프트는 'car/automobile/auto' 등 다중 동의어가 슬래시로 묶여 있어
  임베딩·이웃 정의에 영향. 결과 해석 시 고려.

---

## 통합 서사 (국면 B + C)

하나의 인과 사슬:
  B: margin이 작을수록 flip 급증 (margin<0.1에서 60%)
  C-1: 의미적 이웃은 임베딩이 가까움 → margin 작음 → flip이 이웃으로 향함 (10.9x)
  C-2: 정답 단어 제거 → margin 축소(5.25→1.81) → flip 악화 (0.6%→7.2%)
  C-3: 촘촘한 vocab → margin 축소(6.41→1.29) → flip 증폭(0.51%→4.28%), 편향 31x

→ "양자화 손상은 semantic decision boundary에서, 의미적 방향으로, 프롬프트가 낯설거나
   촘촘할수록 심하게 발현된다." 세 조건 모두 동일한 margin 메커니즘.

논문 Sec 3 함의: reconstruction/AP(총량)로는 안 잡히는 손상이 프롬프트 축의 의미적
경계에 존재. 이것이 open-vocabulary 양자화를 decision preservation 문제로 봐야 하는 근거.

---

## 코드베이스 상태 (국면 C 추가분)

| 파일 | 역할 |
|---|---|
| 04_semantic_substitution.py | C-1 |
| 05_heldout.py (multi-seed) | C-2 |
| 05b_heldout_diagnose.py | C-2 편차 원인 규명 |
| 06_lvis_sanity.py | C-3 sanity |
| 06b_lvis_diagnose.py | C-3 공정성 확인(앵커·이름) |
| 07_lvis_compare.py | C-3 본 측정 |

---

## 남은 한계 및 다음

- 전 실험 naive W8A8. 강한 baseline(BRECQ/QDrop/Reg-PTQ)에서 이 손상이 유지/증폭되는지
  미확인 → "reconstruction을 잘해도 decision은 못 지킨다" 주장의 결정적 조각.
- flip 절대값은 naive에서 아직 작음(수 %). 강한 압축(W4 등)에서 증폭 여지.
- 다음 갈림길: (A) 강한 baseline 이식 vs (B) 제안 방법 설계 착수.
