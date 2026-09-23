# PromptCal-PTQ — 국면 D 결과 (yolov8s-world / v1)

강한 baseline이 held-out region-prompt decision을 지키지 못함을 입증. 제안 방법(E)의 필요성 정당화.

모델: yolov8s-world (v1). W8A8. held-out 측정(05 방식): COCO-80을 K/H 분할, FP top-1=pseudo-GT,
H열 마스킹 후 FP-in-K vs quant-in-K flip. seed 0/1/2, eval 1000장, calib 32장.

---

## D 벤치마크 (held-out, v1)

| 방법 | 최적화 축 | seen flip | held-out flip | held-out 감소 |
|---|---|---|---|---|
| naive W8A8 | — | 1.17% | 9.33% | (기준) |
| AdaRound | weight rounding (layer) | 1.15% | 8.48% | 9.1% |
| BRECQ | weight reconstruction (block) | 1.07% | 8.49% | 8.9% |
| QDrop | activation robustness | 1.21% | 8.53% | 8.5% |
| PD-Quant | prediction difference | 1.15% | 8.48% | 9.1% |

**핵심: 네 방법 모두 held-out flip이 8.48~8.53%로 사실상 동일.** 최적화 축(layer/block/
activation/prediction)과 무관하게 held-out 손상의 ~91%가 잔존. held-out decision 손상은
특정 reconstruction/prediction 방법으로 접근 불가능한 구조적 문제.

Reg-PTQ 기준 필수 3종(BRECQ/QDrop/PD-Quant) 전부 실측 완료 + AdaRound 보조. baseline 커버리지 충족.

---

## 방법별 메모

**AdaRound (weight, layer-wise):** learnable rounding. held-out 8.48%. v1에선 seen조차 거의
개선 못 함(1.17→1.15) — v1 양자화 취약성(AP −3.5) 반영.

**BRECQ (weight, block-wise):** 재구성 단위를 conv→block(C2f 등)으로 확장, block 내 alpha
동시 최적화(layer 간 상관 반영). block단위 17개 + head conv 18개. multi-conv 블록
(block2:4, block4:6, block6:6, block8:4) 정상 joint 최적화(h→0/1 100%). C2fAttn 다중 입력
(vision+text guide) 처리. held-out 8.49% — layer 간 상관을 잡아도 held-out 무의미.

**QDrop (activation robustness):** 최적화 중 activation 확률적 drop. held-out 8.53%. v2에선
seen 최고였으나 v1에선 seen 이점도 사라짐(양자화 취약성).

**PD-Quant (prediction difference):** end-to-end로 cv4 유사도 행렬(예측) 차이 최소화.
warm-start(AdaRound 초기화 + PD refine). held-out 8.48%. PD refine 작동 확인(853초, alpha 71개,
h 99% 유지)했으나 held-out을 AdaRound에서 못 바꿈 → "예측을 맞춰도 held-out 무의미".
주의: v1은 양자화 손상이 커서 pd 값이 진동(0.3~2.1), 완전 수렴 아님. 단 h 99% 유지되어
AdaRound 초기화는 안정. 논문 서술 시 "PD refine이 seen 개선 못 해 held-out에도 무의미"로 정직히.

---

## v1 vs v2 차이 (참고, 논문 프레이밍용)

| | v2 | v1 |
|---|---|---|
| held-out 감소폭 | 15~20% | 8~9% (더 못 지킴) |
| 방법 간 차이 | QDrop이 약간 낮음 | 네 방법 거의 동일(더 극명) |
| seen 개선 | AdaRound/QDrop이 seen 지킴 | 어떤 방법도 seen 거의 못 지킴 |

v1은 양자화 취약성이 커서 baseline들이 seen조차 못 지키고, held-out은 더더욱 무력.
"reconstruction으로 held-out 못 지킴" 논지가 v1에서 더 강하게 성립.

---

## 통합 결론

강한 PTQ(reconstruction: AdaRound/BRECQ, activation: QDrop, prediction: PD-Quant) 모두
held-out region-prompt decision 손상의 ~91%를 동일하게 남긴다. 최적화 축을 바꿔도 결과가
수렴(8.5%)한다는 것은, held-out 손상이 기존 PTQ 패러다임으로 접근 불가능함을 의미.

→ 제안 방법(E)은 reconstruction/prediction이 아니라 held-out region-prompt decision을
   직접 보존해야 하며, 이 8.5%가 공략 지점.

## 코드
| 파일 | 역할 |
|---|---|
| src/quant/adaround.py | AdaRound + QDrop(qdrop_prob) + STE |
| src/quant/pdquant.py | PD-Quant (warm-start 필수) |
| src/quant/brecq.py | BRECQ block-wise (C2fAttn 다중입력 지원) |
| scripts/09~12 | held-out naive vs 각 방법 |

## 다음: 국면 E (제안 방법 설계)
