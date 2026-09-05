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

## 10. (경과) 최초에 세운 다음 실험 후보

1. seed 확대 검증: 35번 조합을 seed 3~6까지 더 돌려서 2승 1패가 우연인지 확인.
2. seed 2 불안정성 원인 진단: lr을 낮추거나 gradient clipping을 강화.
3. H_cal 포함 pidx로 35번 조합 재현.

실제로는 1, 2번을 진행하다가 훨씬 근본적인 문제(§11~13)를 발견해서 3번은 보류됐다.

---

## 11. 36번: seed 2 lr 스윕 — lr 문제가 아니었다

`scripts/36_seed2_stability.py`로 seed 2 고정, lr ∈ {0.01, 0.005, 0.003, 0.001}을
스윕하며 150 iter마다 실제 group_margin(S)/group_flip(H_eval) 궤적을 측정했다
([36_full.txt](results_v1/diag/36_full.txt)).

```text
      lr |  final_S_m | final_Heval |  best_it | best_Heval
    0.01 |     0.1186 |       8.64% |      900 |      8.36%   (35번과 동일 설정인데 개선됨!)
   0.005 |     0.1581 |       8.67% |      750 |      8.27%
   0.003 |     0.1339 |       9.24% |     1200 |      8.64%
   0.001 |     0.1405 |       9.64% |     1350 |      9.41%
```

lr을 낮추는 건 도움이 안 됐다(오히려 원래 lr=0.01이 최선). 결정적으로: **lr=0.01·seed=2로
35번과 완전히 동일한 설정인데, 35번은 악화(S_margin=0.1415, H_eval=9.58%)였고 36번은
개선(S_margin=0.1186, H_eval=8.64%)이었다.** 코드 차이는 없다(eval_hook은
`@torch.no_grad()`라 학습에 영향 불가). → **run-to-run non-determinism**을 의심.

---

## 12. 37번: 반복 실행으로 분산 직접 측정 — "2승 1패"는 노이즈였다

AdaRound 체크포인트를 1회만 만들어 고정하고, 그 지점에서 scale-tuning만 6회 독립
반복해서 분산을 측정했다([37_full.txt](results_v1/diag/37_full.txt),
[37_seed0_full.txt](results_v1/diag/37_seed0_full.txt),
[37_seed1_full.txt](results_v1/diag/37_seed1_full.txt)).

```text
seed | Ada_Heval | Comb mean H_eval | std  | 개선 trial | 평균 격차
   0 |     7.59% |            7.53% | 0.55 |       4/6 | -0.06pp (노이즈 범위)
   1 |     8.92% |            8.81% | 0.36 |       4/6 | -0.11pp (노이즈 범위)
   2 |     8.90% |            9.20% | 0.21 |       1/6 | +0.30pp (악화, 노이즈보다 큼)
```

seed 0/1의 "평균적 개선"은 trial-to-trial 표준편차보다 훨씬 작아 통계적으로 0과
구분되지 않는다. seed 2만 격차가 std보다 커서 어느 정도 실제 신호에 가깝고, 방향은
"악화"다. **35번(및 34번)에서 "방향 C가 AdaRound를 이긴다"고 봤던 결론은 기각된다.**
추가로 AdaRound 기준선 자체도 재구축할 때마다 값이 조금씩 다르다는 것도 확인됨
(reconstruction도 완전히 결정적이지 않음).

---

## 13. 결정성 도입 (torch.manual_seed + cudnn.deterministic)

원인은 코드베이스 어디에도 `torch.manual_seed`/`cudnn.deterministic` 설정이 없어서
cuDNN이 실행마다 다른 conv 알고리즘을 골라 미세한 수치 차이가 1500 iteration에 걸쳐
증폭된 것으로 보인다. `scripts/37_variance_check.py`, `scripts/35_adaround_plus_scale.py`에
다음을 추가했다.

```python
torch.manual_seed(args.torch_seed)   # 기본값 0
torch.cuda.manual_seed_all(args.torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

검증([38_determinism_check.txt](results_v1/diag/38_determinism_check.txt)): 동일 설정
2회 반복이 **완전히 동일한 값**(S_margin=0.1439, H_eval=9.24%, std=0.0000)을 냈다 —
결정성 도입 성공.

이 설정으로 35번(AdaRound vs Combined)을 GPU 7 고정해서 재실행했다
([35_deterministic_full.txt](results_v1/diag/35_deterministic_full.txt)):

```text
seed | Ada_Heval | Comb_Heval | 격차
   0 |     7.59% |      7.36% | -0.23pp
   1 |     8.92% |      9.00% | +0.08pp
   2 |     9.64% |      9.10% | -0.54pp
```

세 값 모두 37번이 보여준 노이즈 구름(분포) 범위 안에 있는 값이다. **중요한 구분**:
결정성은 "같은 설정을 다시 돌리면 같은 숫자가 나온다"는 재현성만 보장하며, 그 숫자
자체가 방법의 진짜 평균 성능을 대표한다는 뜻은 아니다 — scale-tuning 최적화 자체가
여전히 초기 부동소수점 조건에 민감한(margin_loss가 매 checkpoint마다 크게 진동하는)
카오스적 landscape를 갖고 있어서, torch_seed를 바꾸면 여전히 결과가 흔들린다. 즉
37번의 반복측정 결론(사실상 무승부/근소 악화)이 여전히 유효한 결론이다.

또한 흥미롭게도 **AdaRound reconstruction 자체는 GPU/실행이 달라도 거의 재현**됐다
(seed별 baseline이 서로 다른 GPU/실행에서 여러 번 정확히 일치). 즉 non-determinism의
주 원인은 AdaRound reconstruction이 아니라 **scale-tuning(margin loss 기반 Adam) 단계
자체**로 좁혀진다.

---

## 14. 09-05 최종 결론

```
29: Case B 확정 (H_cal 개선 → H_eval 악화, 전이 실패)
31: S-only도 동일하게 악화 → "H_cal을 쓴 것"이 원인 아님
32: raw-init도 일부 원인이지만 semantic이 추가 손해를 얹음
33: AdaRound 완전 warm-start → alpha 포화 → semantic objective 무력화
    → discrete alpha rounding의 구조적 한계
34/35: 연속 s_mult(방향 C)로 첫 긍정 신호처럼 보였음
36/37: 그 신호는 대부분 run-to-run non-determinism에 의한 노이즈였음이 확인됨
    → 반복 측정 결과 seed 0/1은 통계적으로 무승부, seed 2는 오히려 악화
38: torch.manual_seed+cudnn.deterministic 도입, 재현성 확보(작동 검증 완료)
    → 그러나 결정적 재실행도 37번의 노이즈 구름 범위 안의 한 값일 뿐,
      방향 C의 우위를 되살리지 못함
```

**현재까지의 진짜 결론**: discrete AdaRound alpha rounding 기반 semantic objective도,
연속 activation scale(방향 C) 기반 semantic objective도, held-out(H_eval) 일반화를
안정적으로 개선한다는 근거를 찾지 못했다. 유일한 절차적 성과는 실험 인프라
자체(torch.manual_seed+cudnn.deterministic)가 이제 재현 가능해졌다는 것 —
앞으로의 실험은 이 설정을 기본으로 써야 한다.

다음 단계로는 (a) 이 family(AdaRound alpha/scale 기반 semantic objective) 자체를
접고 완전히 다른 접근을 찾거나, (b) 여러 torch_seed로 반복 측정하는 프로토콜을
표준화해서 다른 하이퍼파라미터/objective를 계속 탐색하는 두 갈래가 있다. 09-05
시점에는 아직 결정하지 않음.

---

## 15. 코드 구조 (09-03 문서 이후 추가분)

```text
scripts/29_diag_transfer.py       -- soft/discrete x H_cal/H_eval 분리 진단
scripts/30_semantic_calib.py      -- class-agnostic 재구성 시도(실패)
scripts/31_s_only_control.py      -- H_cal 배제 control
scripts/32_null_control.py        -- semantic=0 null control
scripts/33_warmstart_semantic.py  -- AdaRound warm-start 후 semantic (alpha 포화 확인)
scripts/34_promptcal_scale.py     -- 방향 C(연속 s_mult) 단독 검증
scripts/35_adaround_plus_scale.py -- AdaRound weight + 방향 C scale 결합
                                      (--torch-seed로 결정성 지원)
scripts/36_seed2_stability.py     -- seed 2 lr 스윕 + 학습 궤적(eval_hook) 측정
scripts/37_variance_check.py      -- 동일 설정 N회 반복으로 run-to-run 분산 측정
                                      (--torch-seed로 결정성 지원)

src/quant/semantic_calib.py       -- 30번에서 사용한 class-agnostic objective 구현
src/quant/promptcal.py            -- optimize_promptcal_scale("방향 C") 기존 구현 활용,
                                      soft/ste 버그 수정(True로 복원),
                                      eval_hook 파라미터 추가(학습 중 궤적 측정용)

results_v1/diag/29_full.txt ~ 35_full.txt        -- 각 스크립트 실행 로그/결과
results_v1/diag/36_full.txt                      -- seed 2 lr 스윕 결과
results_v1/diag/37_full.txt                      -- seed 2 6회 반복 (non-determinism)
results_v1/diag/37_seed0_full.txt, 37_seed1_full.txt -- seed 0/1 6회 반복
results_v1/diag/38_determinism_check.txt         -- 결정성 도입 검증(2회 반복 동일값 확인)
results_v1/diag/35_deterministic_full.txt        -- 결정성 적용 후 35번 재실행
```
