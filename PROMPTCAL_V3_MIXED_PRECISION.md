# PromptCal-PTQ — head 양자화 정도 비교 → cv3 @ W8A11 결론 정리

> 국면 E의 최종 method 근거. "held-out semantic 손상을 어디를, 어느 정밀도로 지키면
> 복구되는가"를 head 양자화 정도를 계단식으로 바꿔가며 특정한 기록.
> 결론: **전체 W8A8, 단 head의 cv3 branch(9개 conv)의 activation만 A11.**

---

## 0. 한 줄 결론

**모델 전체는 W8A8로 양자화하되, WorldDetect head의 `cv3`(region embedding branch,
9개 conv)의 activation만 A11로 둔다. weight는 전부 8bit 유지.**
→ held-out semantic flip을 학습 없이 ~34~40% 낮춘다(3 seed 일관). 비용: 9개 conv에
activation +3bit. anti-transfer 없음.

---

## 1. 왜 "학습"이 아니라 "정밀도 배분"인가 (배경)

측정 지표는 **held-out flip**: calibration에서 안 본 프롬프트가 정답인 region에서,
그 정답을 뺐을 때 남은 프롬프트 중 top-1이 FP↔quant에서 뒤집히는 비율.

학습으로 이 손상을 고치려는 시도는 **세 번 모두 실패(anti-transfer)** 했다:

| 시도 | 방식 | held-out 결과 |
|---|---|---|
| v1 | margin loss로 파라미터 학습 | 악화 (독립 최적화 -60% 등) |
| v2 | decision CE로 activation scale 학습 | 악화 (fit할수록 held-out↑) |
| 결합 | cv3@A11 위에 decision CE 학습 | **naive보다 악화 (+69.6%, seed0)** |

공통 원인: **calibration에 보이는 vocabulary(S/H_cal)에 학습으로 맞추면 안 보이는
vocabulary(H_eval)로 일반화가 안 된다(오히려 역효과).** loss 종류(margin/decision)는
무관 — "보이는 것에 맞추는 학습" 자체가 문제.

→ 그래서 **프롬프트에 맞추지 않는(prompt-agnostic) 방법**으로 전환: 손상이 물리적으로
발생하는 지점을 찾아 거기에 정밀도를 배분한다. 이게 학습을 안 하므로 anti-transfer가 없다.

---

## 2. head 양자화 정도 비교 (손상 위치 특정)

WorldDetect head 구조(실측):

| branch | 역할 | 유사도(held-out) 경로? | quant conv 수 |
|---|---|---|---|
| cv2 | box 좌표 회귀 | 아니오 | (head 부분) |
| **cv3** | **region embedding 생성** (cv4에 입력) | **예 — 핵심** | **9** |
| dfl | box 분포 디코딩 | 아니오 | 0 |
| cv4 | ContrastiveHead: embedding↔text 유사도 | 예(계산), **conv 없음** | 0 |

유사도 logit = `cv4(cv3(x), text)`. 즉 결정은 **cv3의 embedding**에 달려 있다.
text embedding은 offline FP(양자화 안 됨). 따라서 유사도를 흔드는 양자화 소스는 cv3(과 상류).

### 2-1. 무엇을 FP로 빼면 held-out이 회복되나 (held-out flip, eval=300, 3 seed)

| config | held-out | Δ vs naive | 주석 |
|---|---|---|---|
| P0 naive (전부 W8A8) | 8.83% | — | 기준 |
| P1 cv4-FP | 8.83% | 0% | **cv4는 conv 없음 → 변화 없음** |
| P2 head-FP (cv2+cv3+cv4 skip) | 3.98% | −55% | ⚠ **confound**(아래 2-3) |
| P3 head + neck1 | 4.01% | −55% | neck 추가효과 없음 |
| P4 head + neck2 | 4.01% | −55% | 〃 |

관찰: neck을 더 빼도 개선이 없다 → 손상은 상류 neck이 아니라 **head에 국소화**. cv4는
conv가 없어 무관 → 후보는 **cv3**(또는 cv2).

### 2-2. cv3 격리 + activation bit sweep (held-out flip, eval=300, 3 seed)

| config | held-out | Δ vs naive | 해석 |
|---|---|---|---|
| naive (cv3 A8) | 8.83% | — | 기준 |
| cv3 @ W8A9 | 6.62% | −25% | 무너지기 시작 |
| cv3 @ W8A10 | 5.64% | −36% | 거의 회복 |
| **cv3 @ W8A11** | **5.23%** | **−41%** | **knee (포화 시작, 최적)** |
| cv3 @ W8A12 | 5.38% | −39% | 포화 |
| cv3 @ W8A13 | 5.36% | −39% | 포화 |
| cv3-FP (상한) | 5.45% | −38% | A11이 이미 FP급 |
| cv3 @ W16A16 | 5.45% | −38% | = cv3-FP |

두 가지가 확정된다:
- **weight bit는 무관.** W16A16 = W8A16 = cv3-FP. → 문제는 cv3 **weight**가 아니라
  cv3 **activation** 정밀도.
- **A11이 knee.** A9에서 급락, A10~A11에서 포화. A11이면 cv3-FP(상한)를 이미 달성.

→ **method의 cv3 config = W8A11.**

### 2-3. ⚠ 주의: P2/P3/P4의 "head-FP −55%"는 confound

`wrap_convs(skip_names={'cv2'})`는 head의 box branch뿐 아니라 **backbone/neck의 모든 C2f
블록 내부 서브모듈 `cv2`까지 전부** FP로 만든다(이름 충돌). 실제로 skip 시 71→43(28개
제외 — box branch만이면 이렇게 많지 않음). 기전 확인(25번): naive vs cv2-FP에서 **cv3의
입력(=neck 출력)이 달라짐**(Δ1.26) → cv2(box)는 cv3의 상류가 아니므로 이는 neck이 바뀐
것이며, 곧 C2f.cv2들이 대량 FP가 된 결과다.

정정:
- 표준 구조대로 **box branch(cv2)는 유사도에 영향 없음**(이론 확인).
- 따라서 "head-FP −55%"는 [cv3(−38%)] + [C2f.cv2 대량 FP에 의한 추가분]의 혼합.
  **정직한 회복치는 cv3 단독 −38~41%.**
- **cv3는 깨끗하다**: `skip={'cv3'}`가 정확히 9개(=head cv3)만 제외했고, `cv3`라는 이름은
  head에만 존재. 또한 최종 method는 skip이 아니라 **head-scoped bit override**(cv3 conv의
  `a_obs.bits=11`)로 적용하므로 충돌 경로를 애초에 타지 않는다.
- 부수 관찰: C2f.cv2 대량 FP가 held-out을 약간(−12%) 낮춘 것 = backbone 잔여 손상은
  실재하나 **분산적**이라 단일 lever가 아님 → core method 밖(future work).

---

## 3. 최종 method

**"전체 W8A8, head cv3 branch(9 convs)의 activation만 A11 (weight 8bit 유지)."**

구현(가산, 학습 없음):
```python
wrap_convs(model.model, 8, 8)                 # 전체 W8A8
for c in cv3_convs(model.model):              # head.cv3 아래 QuantConv2d 9개
    c.a_obs.bits = 11                          # activation만 A11 (weight 8bit)
calibrate(model.model, calib)                 # min/max로 scale freeze
```

특성:
- **학습 없음** → anti-transfer 없음 (v1/v2/결합이 실패한 지점을 원천 회피).
- **값싸다** — 9개 conv activation에 +3bit. head 전체 FP(관행)의 비용을 피함
  (Reg-PTQ의 "head-FP 비용" 우려 해소).
- **baseline·v1·v2가 못한 held-out 감소를 달성** — 아래 4의 대비.

---

## 4. 대비 (메인표에 배치할 위치)

held-out flip 기준(정성):

| 방법 | held-out flip | 비고 |
|---|---|---|
| FP32 | 0 | 상한 |
| Naive W8A8 | 높음 (~8.8~9.4%) | 기준 손상 |
| AdaRound / BRECQ / QDrop / PD-Quant | ~naive (≈8.5%) | reconstruction/prediction 계열, 실패 |
| PromptCal v1 (margin) / v2 (decision) | ≥ naive | 학습 = anti-transfer, 실패 |
| **Ours (cv3 @ W8A11)** | **−34~41%** | **유일하게 감소, 학습 없음, AP 유지 예상** |

(참고: eval 크기에 따라 naive 절대값이 달라진다 — 8.83%@eval300, 9.41%@eval1000.
헤드라인은 **상대 감소율**로 읽을 것. clean 경로 재현: 26번 eval=1000, 3 seed에서 −33.9%.)

---

## 5. 남은 검증 (method 완성까지)

1. **AP 확인** — Ours(cv3@A11)의 mAP50-95가 naive와 동일한지(양자화 둔감 전제 + 최소
   개입이라 거의 안 흔들려야). 26번을 `--no-ap` 없이 실행.
2. **baseline held-out 동일 harness 재측** — AdaRound 등을 같은 probe/seed로 재서 표의
   apples-to-apples 확보(기존 수치 ~8.5%와 일치 확인).
3. **(권장) 진단의 절차화** — "cv3를 사람이 골랐다"가 아니라 held-out 결정 민감도로
   레이어를 자동 랭킹해 정밀도를 배분하는 절차로 승격 → "decision-sensitivity-guided
   mixed-precision PTQ for OVD"로 원리화(다른 OVD 모델 일반화 + novelty 강화).

---

## 부록. 근거 스크립트

- `22_precision_probe.py` — head/neck skip 사다리 (2-1)
- `23_mixed_precision.py` — cv3 격리 + bit sweep (2-2)
- `24_method_confirm.py` — cv3 bit knee 세밀 스윕(A9~A13) (2-2)
- `25_cv2_mechanism.py` — cv2 confound 규명 (2-3)
- `26_main_eval.py` — FP/naive/Ours held-out(+AP) 메인표 (4)
- `27_combine_test.py` — cv3@A11 + decision loss = anti-transfer 재유입 확인 (1)
