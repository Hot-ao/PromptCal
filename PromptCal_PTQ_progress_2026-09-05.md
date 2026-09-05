# PromptCal-PTQ 연구 진행 정리 — 2026-09-05

## 0. 문서 목적

이 문서는 [PromptCal_PTQ_progress_2026-09-03.md](PromptCal_PTQ_progress_2026-09-03.md) 이후 진행한
진단 실험(29~35번 스크립트) 결과를 정리한다. 09-03 문서 마지막(§23, §30)에서 제안한
"soft-to-discrete mismatch vs H_cal→H_eval generalization failure" 분리 진단을 실제로
수행했고, 그 결과 예상과 다른 두 가지가 추가로 드러났다:

1. 09-03 문서의 모든 PromptCal 결과는 **버그가 있는 상태**(alpha가 soft 상태로 최적화되지 않음)에서 나온 것이었다.
2. 버그를 고친 뒤에도 문제는 재현됐지만, 원인은 "H_cal→H_eval 전이 실패"가 아니라
   **discrete AdaRound alpha rounding이라는 파라미터화 자체의 구조적 한계**였다.
3. 연속 파라미터(activation scale)로 바꾸자 처음으로 긍정적인 신호가 나왔다(아직 완전히 안정적이진 않음).

---

## 1. 버그 발견: soft/ste가 최적화 내내 False였음

`src/quant/promptcal.py`의 `optimize_promptcal()`에서 `ac.soft`/`ac.ste`가 9/3 저녁
"checking problem3" 커밋에서 `True→False`로 바뀐 뒤 되돌아오지 않은 상태였다.

```python
ac.soft = False
ac.ste = False
```

`soft=False`면 margin/decision loss가 alpha로 gradient를 못 보낸다(reg_loss만 alpha에 연결됨).
즉 09-03 문서에 기록된 모든 PromptCal 결과(§16~19, 실험 결과 1~3)는 **목적함수가 사실상
작동 안 한 상태**에서 나온 것이었다. `True`로 되돌려 재검증했다.

---

## 2. 29번: 진단 실험 — Case B 확정

`scripts/29_diag_transfer.py`로 soft/discrete × H_cal/H_eval을 분리 측정했다([결과](results_v1/diag/29_full.txt)).

```text
seed |  Ada_Hcal | PCsoft_Hcal | PCdisc_Hcal | Ada_Hcal_m | PCsoft_Hcal_m | PCdisc_Hcal_m | Ada_Heval | PC_Heval
   0 |     6.69% |       3.92% |       5.23% |     0.1371 |        0.0564 |        0.0790 |     8.64% |     9.41%
   1 |     8.96% |       6.57% |       7.76% |     0.1432 |        0.0775 |        0.1022 |     9.23% |    10.59%
   2 |     8.41% |       6.83% |       6.44% |     0.1415 |        0.0655 |        0.0834 |     9.74% |    10.60%
```

3 seed 모두 discrete H_cal margin/flip은 AdaRound보다 개선됐지만 H_eval flip은 악화됐다.
**Case B(H_cal→H_eval 전이 실패)로 깨끗하게 확정** — soft→discrete mismatch(Case A)도
objective 무력(Case C)도 아니다.

---

## 3. 30번: class-agnostic 재구성 시도 — 실패

Case B에 대한 첫 대응으로 `src/quant/semantic_calib.py`("Semantic Calibration +
Utility-Constrained Refinement")를 구현했다. 고정된 60개 class identity를 암기하는 대신
anchor마다 text-embedding상 가장 가까운 경쟁자로 국소 competitor set을 동적으로 구성하면
class-agnostic한 절차라서 일반화될 것이라는 가설이었다. local reconstruction(cv3 MSE)과
utility constraint(threshold-crossing, box consistency)도 추가했다.

결과([30_full.txt](results_v1/diag/30_full.txt)): 평균 AdaRound 대비 **-16.3%**로 오히려
기존 PromptCal(-3~-10%)보다 악화. 가설 기각.

---

## 4. 31번: S-only control — "H_cal을 쓴 것"이 원인이 아님을 확인

`scripts/31_s_only_control.py`: PromptCal objective의 prompt pool을 `S+H_cal`이 아니라
`S`(이미 본 40개)로만 좁혀서, H_cal을 objective에서 완전히 배제했다.

```text
seed |   Ada_S |    PC_S |  Ada_S_m |   PC_S_m | Ada_Heval | PC_Heval(S-only)
   0 |   9.84% |   9.57% |   0.1264 |   0.0866 |     7.95% |            8.69%
   1 |  10.46% |  10.72% |   0.1303 |   0.1102 |     8.97% |           10.10%
   2 |  11.82% |  12.04% |   0.1349 |   0.0830 |     8.90% |           10.80%
```

margin은 3 seed 모두 개선(objective는 작동함)했는데, **H_cal을 아예 안 건드렸는데도
H_eval 악화 폭이 29번(H_cal 포함)과 비슷한 크기**로 나타났다. 즉 "H_cal(held-out) 클래스를
objective에 넣는 것 자체가 문제"라는 가설은 기각. 진짜 원인은 더 일반적인 것으로 보인다.

---

## 5. 32번: null control — raw init도 일부 원인, 그러나 semantic이 추가 손해

`scripts/32_null_control.py`: PromptCal loss에서 semantic 성분(margin, decision)을 완전히
제거하고 alpha regularizer(reg_loss)만 남긴 채, 완전히 동일한 초기화(raw
`convert_to_adaround`, AdaRound reconstruction warm-start 없음)로 alpha를 최적화했다.
reg_loss는 alpha 자체의 함수라 FP 데이터도 forward pass도 필요 없다.

```text
seed | Ada_Heval | Null_Heval(raw init만) | PC_Heval(S-only, 31번)
   0 |     7.95% |                  8.35% |                  8.69%
   1 |     8.97% |                  9.78% |                 10.10%
   2 |     8.90% |                  9.18% |                 10.80%
```

Null도 AdaRound보다 3 seed 모두 나쁘다(+3~9%) — **raw init(AdaRound reconstruction을
안 거친 것) 자체가 이미 공짜가 아닌 비용**이다. 하지만 PromptCal의 degradation(+9~21%)은
Null보다 뚜렷이 더 크다 — **semantic objective가 raw-init 비용 위에 추가 손해를 얹는다.**
두 요인이 섞여 있었고, 둘 다 실재한다.

---

## 6. 33번: AdaRound warm-start — alpha 포화로 objective가 무력화됨

raw-init 문제를 통제하기 위해 `scripts/33_warmstart_semantic.py`에서 AdaRound의
reconstruction 최적화(1000 iters)로 먼저 warm-start한 뒤 그 위에 semantic objective(S-only)를
얹었다.

결과: **3 seed 모두 AdaRound = warmstart+null = warmstart+semantic, 완전히 동일한 수치**
(flip/margin/H_eval 전부 소수점까지 일치). 원인: warm-start 직후 이미 `h→0/1≈99%`로 alpha가
포화돼 있어서, sigmoid 기반 `h_alpha`의 gradient가 포화 영역에서 극도로 작아지고(정확히
0은 아님 — "alpha 총 변화량"이 null=243827.72, semantic=236289.63로 미세하게 다름), 1500
iteration 동안 값이 계속 움직여도 rounding 임계값(0.5)을 하나도 못 넘는다. 최종 discrete
weight가 AdaRound와 bit-for-bit 동일해져서 이후 모든 측정이 같은 모델을 본 것이었다.

**결론**: discrete alpha rounding은 "포화 전(raw init, negative transfer)" 아니면
"포화 후(완전 무력화)" 둘 중 하나뿐이고, 유효하게 semantic objective가 작동하는 중간
지점이 사실상 없다 — 구조적 한계.

---

## 7. 34번: 방향 C(연속 s_mult) — 첫 긍정 신호

discrete alpha 대신 코드베이스에 이미 있던 `optimize_promptcal_scale`("방향 C": weight
rounding은 건드리지 않고 연속값인 activation quantization scale multiplier `s_mult`를
margin objective로 학습)을 검증했다(`scripts/34_promptcal_scale.py`, pidx=S).

```text
seed |   nv_Heval | Ada_Heval | PCscale_Heval |  Ada_S_m | PCscale_S_m
   0 |      8.14% |     7.95% |         7.67% |   0.1264 |      0.1107
   1 |      9.56% |     8.97% |         9.12% |   0.1303 |      0.1137
   2 |      9.18% |     8.90% |         9.07% |   0.1349 |      0.1166
```

margin은 3 seed 모두 AdaRound보다도 개선. 공정한 비교축인 naive 대비로는 **3 seed 모두
H_eval 개선**(seed 0은 AdaRound보다도 좋음) — 진단 시리즈 전체에서 첫 긍정 신호.
연속 파라미터는 포화-절벽 문제가 없다는 것이 실제로 확인됨.

---

## 8. 35번: AdaRound weight + 방향 C scale 결합 — 2승 1패

AdaRound로 weight rounding을 먼저 최적화(reconstruction 품질 확보)하고, alpha는 그대로
고정한 채 그 위에 방향 C의 연속 scale만 추가로 얹었다(`scripts/35_adaround_plus_scale.py`).
weight rounding이 완전히 동일하므로 지금까지 중 가장 공정한 비교.

```text
seed |  Ada_S_m | Comb_S_m | Ada_Heval | Comb_Heval
   0 |   0.1264 |   0.1082 |     7.95% |      7.72%   (개선)
   1 |   0.1303 |   0.1068 |     8.97% |      8.21%   (개선, -8.5%)
   2 |   0.1349 |   0.1415 |     8.90% |      9.58%   (악화, margin도 악화)
```

seed 0/1은 margin·H_eval 모두 AdaRound보다 좋아졌다. seed 2는 **S margin 자체가
악화**(34번에서는 3 seed 모두 margin이 개선됐던 것과 다른 패턴) — 최적화 자체가 불안정하게
실패한 것으로 보인다(margin loss가 수렴 없이 진동). "일관되게 이긴다"고 하기엔 이르다.

---

## 9. 종합 결론

```
29: Case B 확정 (H_cal 개선 → H_eval 악화, 전이 실패)
31: S-only도 동일하게 악화 → "H_cal을 쓴 것"이 원인 아님
32: raw-init도 일부 원인이지만 semantic이 추가 손해를 얹음
33: AdaRound 완전 warm-start → alpha 포화 → semantic objective 무력화
    → discrete alpha rounding의 구조적 한계 확인
34: discrete alpha 대신 연속 s_mult(방향 C) → naive 대비 H_eval 개선(첫 긍정 신호)
35: AdaRound weight + 방향 C scale 결합 → 3 seed 중 2개는 AdaRound보다 우수,
    1개는 최적화 불안정으로 악화
```

지금까지의 핵심 재발견: **문제는 "semantic margin objective 자체가 나쁘다"가 아니라
"discrete AdaRound alpha rounding이라는 파라미터화가 이 semantic objective와 근본적으로
안 맞는다"였다.** 연속 파라미터(activation scale)로 바꾸자 같은 objective가 처음으로
순수하게 도움이 되는 방향으로 작동했다. 다만 35번의 seed 2 불안정성 때문에 아직
"방법이 확정됐다"고 말하기는 이르다.

---

## 10. 다음 실험 후보 (미착수)

1. **seed 확대 검증**: 35번 조합을 seed 3~6까지 더 돌려서 2승 1패가 우연인지, 평균적으로
   AdaRound를 이기는지 확인.
2. **seed 2 불안정성 원인 진단**: lr을 낮추거나 gradient clipping을 강화해서 margin loss
   발산/진동을 먼저 잡을 수 있는지 확인.
3. (미착수) H_cal을 포함한 pidx로 35번 조합을 재현해서 지금까지의 "H_cal 포함 여부는
   무관하다"는 31번 결론이 이 새 파라미터화에서도 유지되는지 확인.

---

## 11. 코드 구조 (09-03 문서 이후 추가분)

```text
scripts/29_diag_transfer.py       -- soft/discrete x H_cal/H_eval 분리 진단
scripts/30_semantic_calib.py      -- class-agnostic 재구성 시도(실패)
scripts/31_s_only_control.py      -- H_cal 배제 control
scripts/32_null_control.py        -- semantic=0 null control
scripts/33_warmstart_semantic.py  -- AdaRound warm-start 후 semantic (alpha 포화 확인)
scripts/34_promptcal_scale.py     -- 방향 C(연속 s_mult) 단독 검증
scripts/35_adaround_plus_scale.py -- AdaRound weight + 방향 C scale 결합

src/quant/semantic_calib.py       -- 30번에서 사용한 class-agnostic objective 구현
src/quant/promptcal.py            -- optimize_promptcal_scale("방향 C") 기존 구현 활용,
                                      soft/ste 버그 수정(True로 복원)

results_v1/diag/29_full.txt ~ 35_full.txt  -- 각 스크립트 실행 로그/결과
```
