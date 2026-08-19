# PromptCal-PTQ 실험 진행 정리 — 국면 D (강한 baseline)

작성 시점: AdaRound·QDrop 확정, PD-Quant 재실행 중(warm-start 수정본), BRECQ 예정.
목적: "reconstruction/prediction 계열의 강한 PTQ로도 held-out decision 손상을 못 지킨다"
      를 표준 baseline으로 입증 → 제안 방법(E)의 필요성 정당화.

---

## 요약 (한 줄)

강한 PTQ(AdaRound=weight, QDrop=activation)는 AP를 개선하고 seen decision을 어느 정도
지키지만, held-out(안 본 프롬프트) decision 손상의 80% 이상을 공통으로 남긴다.
최적화 축(weight/activation)을 바꿔도, reconstruction 목표를 공유하는 한 결론은 동일.

---

## baseline 커버리지 (Reg-PTQ 기준)

Reg-PTQ(WACV급 detection PTQ)의 baseline 등급:
- 필수 3종: BRECQ, PD-Quant, QDrop (거의 모든 비트 설정에서 실측)
- 보조: AdaRound, AdaQuant (일부 설정)
- 최소: SubsetQ

우리 계획:
| 방법 | 축 | 등급 | 상태 |
|---|---|---|---|
| Naive W8A8 | 기초 | 기준선 | ✅ |
| AdaRound | weight rounding | 보조 | ✅ |
| QDrop | activation robust | 필수 | ✅ |
| PD-Quant | prediction difference | 필수 | ⏳ 재실행 |
| BRECQ | block reconstruction | 필수 | ❌ 예정 |
| AdaQuant, SubsetQ | — | 보조/최소 | 논증 커버 |

→ 필수 3종(BRECQ/QDrop/PD-Quant) 실측이 목표. AdaRound는 덤.

---

## 공통 설정

- 모델: yolov8s-worldv2 (fused), W8A8, fuse 후 vision Conv 68개 래핑
- calibration: 32장 (vision-only, 프롬프트 무관)
- 측정: held-out(05 방식) — COCO 80을 K/H 분할, FP top-1=pseudo-GT, H열 마스킹 후
  FP-in-K vs quant-in-K flip. seed 0/1/2, eval 1000장.
- 비교 기준: naive held-out flip 6.14%, seen flip ~0.7%

---

## D-1 — AdaRound (weight rounding) ✅

learnable rounding: round 대신 floor + h(alpha), layer-wise reconstruction으로 alpha 최적화.

| 지표 | naive | AdaRound |
|---|---|---|
| AP (pycocotools) | 37.3 | 37.5 (reconstruction 개선) |
| seen flip | 0.73% | 0.40~0.58% |
| held-out flip | 6.14% | 5.24% (감소 14.7%) |
| flip→이웃 | 69% | 83.5% (남은 손상 더 순수 의미적) |

핵심: AP를 FP(37.8)에 근접시켜도 held-out 손상의 85% 잔존.

## D-3 — QDrop (activation robustness) ✅

AdaRound + 최적화 중 activation을 확률 p로만 양자화(나머지 FP) → activation 노이즈 강건화.

| seed | seen_nv | seen_qd | ho_nv | ho_qd |
|---|---|---|---|---|
| 0 | 0.61% | 0.45% | 5.65% | 4.75% |
| 1 | 0.94% | 0.52% | 6.79% | 5.22% |
| 2 | 0.68% | 0.57% | 5.97% | 4.79% |
| mean | | ~0.51% | 6.14% | 4.92% (감소 19.8%) |

핵심: QDrop이 seen은 세 방법 중 최고로 지키지만(0.51%), held-out은 여전히 80% 잔존.
seen을 잘할수록 held-out과의 격차가 벌어짐.

## D-4 — PD-Quant (prediction difference) ⏳ 재실행 중

end-to-end로 '최종 예측(=cv4 유사도 행렬)의 FP vs 양자화 차이'를 최소화.
우리 방법과 개념적으로 가장 가까운 baseline(예측을 봄) → 가장 위험 = 반드시 실측.

구현: 옵션 A(유사도 행렬 기반 PD loss), DC 생략. 
[중요] 1차 실행 실패: from-scratch end-to-end가 68 layer 결합으로 불안정(pd 진동
0.98→12.5→1.4, reg가 pd 압도). 
수정: AdaRound 초기화 + PD refine(warm-start), 정규화 PD loss, mean reg, grad clipping.
→ 재실행 대기. 성공 기준: seen_pd ≤ AdaRound(0.58%)로 수렴해야 신뢰 가능.
예상: held-out flip ≈ 5% 근처 → "예측을 맞춰도 calib vocab에 갇혀 held-out 실패" 확정.

## D-2 — BRECQ (block reconstruction) ❌ 예정

AdaRound의 block 단위 확장. C2f 등 아키텍처 block 경계로 묶어 block 출력을 재구성.
AdaRound 코드 재활용 가능(재구성 단위만 conv→block). 필수 baseline이라 반드시 추가.

---

## 통합 결론 (현재까지)

reconstruction/prediction 계열의 강한 PTQ가 held-out에서 공통 실패:

| 방법 | 축 | held-out flip | 감소 |
|---|---|---|---|
| naive | 기초 | 6.14% | — |
| AdaRound | weight | 5.24% | 14.7% |
| QDrop | activation | 4.92% | 19.8% |
| PD-Quant | prediction | (재실행) | — |

프레이밍(정직하게): "reconstruction은 held-out에 전혀 무관"(과장) 이 아니라
"정교화해도 held-out 개선은 15~20%에 그치고, 손상의 80%가 구조적으로 잔존".
이 80%가 제안 방법(E)이 공략할 지점 — reconstruction/prediction이 아니라
held-out region-prompt decision을 직접 보존.

---

## 코드베이스 (D 추가분)

| 파일 | 역할 |
|---|---|
| src/quant/adaround.py | AdaRound(learnable rounding) + QDrop(qdrop_prob) + STE + reg_loss |
| src/quant/pdquant.py | PD-Quant end-to-end 정제(warm-start 필수) |
| scripts/08_adaround.py | D-1: AdaRound AP+semantic |
| scripts/09_heldout_adaround.py | D-1: held-out naive vs AdaRound |
| scripts/10_qdrop.py | D-3: held-out naive vs QDrop |
| scripts/11_pdquant.py | D-4: held-out naive vs PD-Quant |
| (예정) 12_brecq.py | D-2: BRECQ |

## 다음

1. PD-Quant 재실행 결과 확인 (seen_pd 수렴 여부 = 신뢰성 판정).
2. BRECQ 추가 → 필수 3종 완성.
3. (선택) AdaQuant/SubsetQ 논증 커버 vs 실측 결정.
4. D 전체 최종 벤치마크표 → E(방법) 착수.
