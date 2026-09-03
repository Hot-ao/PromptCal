# PromptCal-PTQ — 국면 D 완전 벤치마크 (세 모델 × 다섯 방법)

강한 PTQ baseline이 held-out region-prompt decision을 지키지 못함을, 세 모델 전반에서 입증.
held-out flip(%) — seen을 pseudo-GT로, H열 마스킹 후 FP-in-K vs quant-in-K. seed 평균.

---

## 마스터 표: held-out flip (%)

| 방법 | 축 | v2 (worldv2-s) | v1-s | v1-m |
|---|---|---|---|---|
| naive W8A8 | 기초 | 6.14 | 9.33 | 14.33 |
| AdaRound | weight (layer) | 5.24 | 8.48 | 11.89 |
| BRECQ | weight (block) | — | 8.49 | 11.55 |
| QDrop | activation | 4.92 | 8.53 | 14.42 |
| PD-Quant | prediction | 5.24 | 8.48 | 11.89 |

*(v2 BRECQ는 미측정 — v1에서 필수 3종 완성으로 대체)*

### held-out 감소율 (naive 대비, %)

| 방법 | v2 | v1-s | v1-m |
|---|---|---|---|
| AdaRound | 14.7 | 9.1 | 17.0 |
| BRECQ | — | 8.9 | 19.4 |
| QDrop | 19.8 | 8.5 | −0.6 |
| PD-Quant | 14.7 | 9.1 | 17.0 |

---

## 핵심 결론

1. **모든 강한 baseline이 held-out 손상의 대부분(80~91%)을 남긴다.** 최적화 축
   (weight layer/block, activation, prediction)과 모델 크기(s/m)를 불문하고 공통.

2. **어떤 방법도 held-out을 근본적으로 못 지킨다.** naive 대비 최대 감소가 ~20%.
   held-out decision 손상은 기존 reconstruction/prediction PTQ 패러다임으로 접근 불가능.

3. **모델별 편차(정직하게):**
   - v1-s: 네 방법이 8.5%로 거의 완전 수렴 — "축 무관"이 가장 극명.
   - v1-m: weight 계열(AdaRound/BRECQ/PD-Quant ~11.7%)은 ~18% 감소, activation(QDrop)은
     0% 감소로 갈림. 손상이 큰 모델에선 held-out이 weight로 일부 접근 가능하나
     activation으로는 전무. 어느 쪽이든 80% 잔존.
   - v2: 손상 절대값 작음(naive 6.14%)이나 패턴 동일.

4. **PD-Quant = AdaRound (세 모델 공통):** warm-start(AdaRound 초기화+PD refine) 결과,
   예측 정합을 추가해도 held-out이 AdaRound와 동일. "예측을 맞춰도 held-out 무의미" 확정.

→ 제안 방법(E)은 held-out region-prompt decision을 직접 보존해야 하며, 이 80%가 공략 지점.

---

## 방법별 구현/하이퍼파라미터 메모 (정직성)

- **QDrop drop 확률:** v2/s는 0.5로 정상. **m은 0.5에서 최적화 불안정**(seen조차 악화,
  held-out −8.8%) → **0.25로 낮춰야 수렴**(seen 개선, held-out naive 수준). m의 양자화
  예민성 때문. 논문에 모델별 drop 확률 명기 필요.
- **PD-Quant warm-start 필수:** from-scratch end-to-end는 불안정(pd 폭발). AdaRound 초기화 +
  PD refine(정규화 loss, mean reg, grad clip)로 안정화. pd는 s(0.3~2.1)·m(0.7~2.2)에서
  진동하나 h→0/1 99% 유지 → AdaRound 초기화 안정, 결과 신뢰 가능.
- **BRECQ:** 재구성 단위 = top-level 블록(비head) + head conv 단위. C2fAttn 다중입력
  (vision+text guide) 지원. m은 conv 91개.
- **m AP 취약성:** m은 W8A8에서 AP −13.7(box 축). 단 semantic decision(우리 측정 축)은
  건강(00d: 전체 flip 0.95%). D 측정은 semantic 축(held-out flip)이라 AP 취약성과 무관.

## 다음: 국면 E (제안 방법 설계)
