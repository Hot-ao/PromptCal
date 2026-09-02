# PromptCal-PTQ 방법 설계 정리 (내부용)

> 현재 상황: 국면 A~D(관찰·baseline)는 완료. 국면 E(제안 방법)를 설계 중이며,
> 세 번의 시도가 모두 baseline을 못 넘었고, 지금 네 번째 방향(activation scale 최적화)을
> 시험하는 단계. 이 문서는 "지금 우리가 뭘 하고 있고, 뭘 고려해야 하며, 어디서 막혔는지"를
> 정리한다.

---

# 파트 1. 우리가 지금 목표하는 것 (기여점·중점 사항)

## 요약

우리 논문의 핵심 주장은 **"open-vocabulary detector의 양자화 손상은 tensor를 얼마나 잘
복원하느냐(reconstruction)의 문제가 아니라, region-prompt 의미적 의사결정(semantic decision)을
보존하느냐의 문제"** 라는 것이다. 국면 A~D에서 이 주장을 데이터로 입증했고(기존 baseline 전부
held-out 손상의 80%+를 남김), 국면 E에서 **이 손상을 실제로 줄이는 방법**을 만들어 baseline을
넘는 것이 목표다. 즉 우리는 "문제를 보여주는 것"(measurement)을 넘어 "문제를 푸는
것"(positive method)을 지향한다.

## 상세

### 세 가지 기여점 (우선순위 순)

1. **문제 정의와 측정 (이미 확보됨).** 양자화가 손상시키는 것이 AP(탐지 총량)가 아니라
   held-out prompt에 대한 semantic decision임을 세 모델(v2, v1-s, v1-m)에서 일관되게 보였다.
   특히 held-out flip이라는 진단 축을 세웠고, 이것이 모델이 커질수록 심해짐을 확인했다.

2. **기존 baseline의 구조적 한계 (이미 확보됨).** AdaRound, BRECQ, QDrop, PD-Quant 등 최적화
   축이 다른 네 방법이 held-out을 8.5%(v1-s)에서 거의 동일하게 남긴다는 것을 보였다. 즉
   reconstruction/prediction 계열 전체가 이 문제에 무력하다.

3. **제안 방법 PromptCal-PTQ (설계 중, 미완).** held-out semantic decision을 직접 보존하는
   PTQ를 만들어 baseline을 넘는 것. **이것이 지금 막혀 있는 부분이다.**

### 지금 설계에서 반드시 지켜야 할 중점 사항

- **strong naive를 넘어야 한다.** 단순 양자화(naive)보다 나빠지면 실패다. (과거 실패의 핵심)
- **held-out에서 개선되어야 한다.** calibration(보이는 프롬프트)에서 좋아지는 건 의미 없다.
  안 본 프롬프트에서 좋아져야 한다.
- **multi-seed로 일관되어야 한다.** 한 번 잘 나온 건 우연일 수 있다.
- **정직한 측정.** 학습에 쓴 프롬프트로 평가하면 안 된다. (아래 파트 3의 prompt 3분할)

> **용어 설명**
> - **PTQ (Post-Training Quantization)**: 재학습 없이, 이미 학습된 모델의 weight/activation을
>   낮은 bit(예: 8bit)로 바꾸는 것.
> - **open-vocabulary detector**: 미리 정한 클래스만 아니라, 사용자가 텍스트 프롬프트로 준
>   임의의 클래스를 탐지하는 검출기. YOLO-World가 대표.
> - **region-prompt semantic decision**: 이미지의 한 영역(region)이 어떤 텍스트 프롬프트로
>   판정되는가. 여러 프롬프트 유사도의 상대 순서로 결정됨.
> - **held-out**: calibration(보정)에 쓰지 않은, 모델이 "안 본" 프롬프트.
> - **flip**: FP32(원본) 모델의 판정이 양자화 후 다른 프롬프트로 바뀌는 것. 손상의 척도.
> - **AP (Average Precision)**: 탐지 성능 총량 지표. 우리 주장은 "AP로는 이 손상이 안 보인다".
> - **calibration**: 양자화 scale을 정하려고 소수 이미지를 모델에 통과시키는 절차.
> - **naive quantization**: 아무 보정 없이 min/max로 scale 잡는 가장 단순한 양자화.

---

# 파트 2. 참고 자료에서 지금 설계에 반영할 점

## 요약

세 자료가 각각 다른 걸 준다. **09_phase1(우리 이전 실패 기록)** 은 "하지 말아야 할 것"의
목록 = 설계 제약을 준다. **Reg-PTQ** 는 FGIC(필터링된 전역 손실)와 LLAQ(비균등 분포 양자화)라는
구체적 기법을 준다. **YOLO-World 구조** 는 우리가 보존해야 할 대상(region-text 정렬)이 어디
있는지를 알려준다. 이 중 가장 무겁게 반영해야 할 것은 09_phase1의 실패 교훈이다.

## 상세

### 2-1. 09_phase1 (이전 RankSafe-v2 시도의 실패 기록) — 설계 제약

이 문서는 우리 팀이 이전에 비슷한 시도(RankSafe-v2)를 했다가 **"실패"로 정직하게 결론낸**
기록이다. 핵심 교훈 네 가지가 그대로 우리 설계 제약이 된다:

| 그때의 실패 | 우리가 지켜야 할 제약 |
|---|---|
| calibration objective를 줄여도 held-out은 악화됨 | 목적함수가 held-out을 **직접** 겨냥해야 함 |
| strong naive N=1을 못 넘음 | 처음부터 강한 naive를 기준으로 설계 |
| seed 하나 실패가 우연이 아니었음 | multi-seed 일관성 필수 |
| raw NDCG/cosine 좋아져도 GT semantic error 악화 | proxy가 아니라 실제 결정으로 검증 |

그리고 그 손실함수는 이런 형태였다(팀 전언):
`L = λr·rank + λt·topk + λd·dist + λb·box + λc·hardware` — **다섯 항 짬뽕.**
→ 이게 실패 요인 중 하나. 항이 너무 많아 무엇이 효과를 내는지 식별 불가.
→ **우리 교훈: 항을 최소화하고, held-out을 직접 겨냥한다.**

단, 이 손실함수의 한 통찰은 옳다: **"top-k 경계(순위 k와 k+1 사이)의 margin 보존이 중요하다.
1위-500위 사이보다 이 경계 역전이 실제 탐지에 훨씬 큰 영향."** → 우리 margin_loss가 이걸 계승.

### 2-2. Reg-PTQ의 FGIC — 우리 방법의 뼈대 후보

**FGIC (Filtered Global-loss Integration Calibration)**: 기존 PTQ는 local loss(레이어 출력 MSE)만
보는데, 그게 최적 scale을 못 준다. FGIC는 여기에 **global loss(최종 예측 정합)** 를 결합하되,
저품질 예측(낮은 confidence, 낮은 IoU box)을 **두 단계로 필터링**해서 노이즈를 뺀다.

→ **우리 번역**: FGIC의 "global loss = box 예측 정합"을 → "global loss = region-prompt 유사도
결정"으로 바꾸고, "confidence+IoU 필터링"을 → "신뢰 region 필터링"으로 바꾼다. 즉 신뢰할 만한
region-prompt 쌍만 골라 semantic decision을 보존. **이게 우리 방법의 ① Semantic Calibration Set.**

### 2-3. Reg-PTQ의 LLAQ — 보조 도구 (선택적)

**LLAQ (Learnable Logarithmic-Affine Quantizer)**: regression head의 weight는 분포가 중앙에
몰린 비균등(non-uniform) 형태라 균등 양자화에 안 맞는다. LLAQ는 이를 로그 공간으로 변환해
균등하게 만든 뒤 양자화한다.

→ **우리 활용**: v1-m에서 본 양자화 취약성(특정 레이어 분포가 특이)에 적용하면 완화 가능.
단 핵심은 아니고, ablation으로 효과 확인 후 넣을지 결정.

### 2-4. YOLO-World의 region-text matching 구조 — 보존 대상의 위치

YOLO-World의 핵심은 vision region feature와 text(prompt) embedding을 대조(contrastive)해
유사도를 내는 것(RepVL-PAN + region-text matching). 우리가 harness로 캡처하는 cv4가 바로 이
정렬의 출력이다.

→ **우리 활용**: 양자화가 손상시키는 건 이 region-text 정렬이다. 따라서 우리 방법은 "정렬(유사도
순서)이 FP와 유지되도록, 특히 held-out text에 대해서도" 최적화해야 한다. 그리고 이 정렬을 만드는
weight(matching) 자체는 건드리지 않고, 양자화로 틀어진 부분만 바로잡는 게 구조를 존중하는 길.

> **용어 설명**
> - **FGIC**: 위 2-2 참조. 필터링된 전역 손실 보정.
> - **LLAQ**: 위 2-3 참조. 학습 가능한 로그-affine 양자화기.
> - **local loss / global loss**: local = 레이어별 출력 오차. global = 모델 최종 출력(예측) 오차.
> - **IoU (Intersection over Union)**: 두 박스가 겹치는 정도. box 품질 척도.
> - **non-uniform distribution**: 값이 특정 구간(주로 중앙)에 몰린 분포. 균등 양자화에 불리.
> - **contrastive matching**: 두 종류(여기선 vision과 text)를 유사도로 정렬하는 학습 방식.
> - **RepVL-PAN**: YOLO-World가 vision과 language를 융합하는 신경망 구조 이름.
> - **cv4**: YOLO-World head 안에서 region-text 유사도를 내는 부분(ContrastiveHead). 우리 측정 지점.

---

# 파트 3. PromptCal 방법의 주요 파이프라인 (설계 중점)

## 요약

방법은 세 요소로 구성된다. **① Semantic Calibration Set**(held-out 프롬프트를 흉내내어
calibration에 포함), **② Decision-Preserving Objective**(레이어 복원 대신 유사도 순서/경계 margin
보존), **③ GT Semantic Utility Gate**(안 본 프롬프트에서 multi-seed로 검증). 실험 정직성의 핵심은
프롬프트를 세 그룹(S / H_cal / H_eval)으로 나눠 "학습에 쓴 걸로 평가하지 않는" 것이다. ③(gate)은
잘 작동해서 실패를 정직하게 잡아냈고, ①·②는 아직 형태를 바꿔가며 시험 중이다.

## 상세

### 3-1. Prompt 3분할 (정직성의 핵심)

COCO-80 프롬프트를 seed마다 무작위로 나눈다:

- **S (seen, 40개)**: calibration에 쓰는 보이는 프롬프트.
- **H_cal (calib held-out, 20개)**: calibration에 **포함**하되 "안 본 프롬프트 흉내"용. 방법이
  이걸로 일반화를 학습한다.
- **H_eval (eval held-out, 20개)**: 학습에 **절대 안 씀.** 최종 flip은 여기서만 측정.

→ 최적화는 S+H_cal(60개)의 margin을 보존, 측정은 H_eval에서만. 이래야 "측정할 프롬프트를
학습에 넣는 반칙"을 피한다.

### 3-2. 세 구성요소

**① Semantic Calibration Set** (Reg-PTQ FGIC 차용)
- 신뢰 region 필터링: FP가 confident한 region-prompt 쌍만 사용.
- held-out 포함: H_cal을 calibration에 넣어 일반화 유도 (D에서 본 "calib vocab 과적합" 차단).
- hard-negative 주입 (미구현): 의미적 이웃 프롬프트를 경쟁자로 넣어 경계 강건화.

**② Decision-Preserving Objective** (핵심 목적함수)
- 레이어 출력 MSE(reconstruction) 대신 **region-prompt 유사도의 순서/margin 보존**.
- 현재 구현: `margin_loss` — 각 anchor의 top-(k+1) 유사도에서 인접 pairwise margin을 뽑아
  FP와 MSE로 맞춤. top-k 경계(마지막 margin)에 3배 가중.
- weight는 고정, 양자화 파라미터만 최적화.

**③ GT Semantic Utility Gate** (09_phase1 실패 방지) — **잘 작동 중**
- strong naive 대비 held-out 개선을 multi-seed로 검증.
- calibration objective 개선이 아니라 GT held-out 결정 개선으로만 성공 판정.

### 3-3. 최적화 절차 (현재 구현)

```
1. wrap_convs(W8A8) → calibrate (naive 양자화 초기화)
2. cv4에 hook 걸어 FP·quant의 유사도 행렬 [anchors, prompts] 캡처
3. iters회 반복:
   - quant 모델 forward → 유사도 행렬 (STE로 gradient 흐름)
   - confident anchor + S/H_cal 프롬프트에서 margin_loss 계산
   - backward → 양자화 파라미터 업데이트
4. 최종 확정(hard) 후 H_eval에서 flip 측정 → baseline과 비교
```

> **용어 설명**
> - **anchor**: 검출기가 이미지에서 보는 후보 위치들(YOLO-World는 8400개). 각 anchor가 프롬프트와
>   유사도를 가짐.
> - **margin**: 1위 유사도와 2위(또는 k위와 k+1위) 유사도의 차이. 작을수록 경쟁이 팽팽 = 뒤집히기 쉬움.
> - **top-k**: 유사도 상위 k개. top-k 경계 = k위와 k+1위 사이.
> - **pairwise margin**: 인접한 두 순위 사이의 차이.
> - **hard-negative**: 정답과 의미적으로 가까워 헷갈리기 쉬운 오답 프롬프트.
> - **STE (Straight-Through Estimator)**: 반올림처럼 미분 불가능한 연산을, 역전파 때는 그냥
>   통과시켜 gradient를 흐르게 하는 기법.
> - **hook**: 신경망 중간 출력을 가로채 꺼내오는 장치. 우리는 cv4 출력을 이걸로 캡처.
> - **MSE (Mean Squared Error)**: 평균 제곱 오차. 두 값의 차이를 제곱해 평균.

---

# 파트 4. 실험 과정에서 부딪힌 문제와 수정 과정

## 요약

네 번의 시도가 있었다. **(0) 측정 버그** — held-out flip을 잘못 재서 방법 효과가 안 보였던 것(수정
완료). **(1) AdaRound 갇힘** — 방법을 AdaRound 위에 얹으니 AdaRound와 똑같아짐. **(2) 독립
최적화** — AdaRound를 떼니 held-out이 오히려 악화(-60%). **(3) warmup 추가** — 여전히 악화,
원인이 "rounding이 안 굳음"으로 확정. **(4) 방향 C(scale 최적화)** — rounding(이산) 대신 activation
scale(연속)을 최적화. s_mult가 매끄럽게 움직이는 것까지 확인, held-out 측정은 아직 안 함. 핵심
교훈: **margin_loss는 매번 줄어드는데 held-out은 안 좋아진다 = 목적함수(margin 값)와 측정
지표(argmax 결정)가 다른 것을 보고 있을 가능성(의심 C).**

## 상세

### (0) 측정 버그 — held-out flip 계산이 틀림
- 증상: PromptCal이 naive/AdaRound와 0.2%로 똑같이 바닥. (05에선 held-out이 10.8%였는데)
- 원인: `heldout_flip`에서 H_eval을 "남기는" 방향으로 짜서, 실제 held-out 경쟁을 안 봄.
- 수정: 05와 동일 로직(H_eval 마스킹 후 K-only argmax 비교)으로 교체. 이후 naive 8.6%,
  AdaRound 8.5%로 05와 일치 → 측정 건강해짐.
- **교훈: 새 측정 코드는 검증된 기존 코드(05)와 수치가 맞는지부터 확인.**

### (1) AdaRound 갇힘 — 방법이 baseline과 동일해짐
- 구조: calibrate → AdaRound(완전 수렴) → promptcal refine.
- 증상: PromptCal = AdaRound 소수점까지 동일 (8.55 = 8.55).
- 원인: AdaRound가 rounding을 100% 굳힌 뒤라 refine이 그 상태를 못 벗어남. (PD-Quant가
  AdaRound와 같아진 것과 동일한 warm-start 갇힘)
- 수정: AdaRound 초기화 제거, calibration 직후 바로 margin 최적화.

### (2) 독립 최적화 — held-out 악화
- 설정: AdaRound 제거, lr=1e-2, reg=0.1.
- 결과: held-out 9.24% → 14.80% (**-60%**, 3 seed 일관). 즉 baseline보다 나빠짐.
- 관찰: rounding 수렴률(h→0/1)이 26%에서 멈춤. margin_loss는 감소하는데 held-out은 악화.

### (3) warmup 추가 — 원인 확정
- 설정: 전반 40%는 reg 강하게(h 굳히기 시도), lr=3e-3.
- 결과: 여전히 악화 (7.99% → 9.07%). h가 32%에서 정체. alpha 변화량 폭주(446만).
- **빠른 진단(dbg 스크립트, 21초)로 원인 확정:**
  - h→0/1이 11%→14%로 거의 안 오름. alpha의 80~85%가 0/1로 안 가고 **중간에 껴 있음.**
  - 이 어중간한 rounding이 soft→hard 전환 때 대량 오류 → held-out 악화.
  - reg를 강하게 줘도 안 굳는 이유: **margin_loss와 reg가 서로 반대로 당김.** reg는
    "h를 0/1로", margin_loss는 "특정 중간값으로" → 싸우다 중간에 갇힘.
  - **즉 margin_loss라는 목적함수가 이산적 rounding(alpha)과 근본적으로 미스매치.**

### (4) 방향 C — activation scale 최적화 (현재 위치)
- 착상: alpha(이산, 0/1로 굳어야 함)가 연속 목적함수(margin)와 안 맞는 게 문제라면,
  **연속값인 activation scale을 최적화 대상으로** 바꾸자. scale은 굳을 필요가 없다.
- 구현: AdaRoundQuantConv2d에 learnable scale multiplier `s_mult`(초기 1.0) 추가.
  rounding은 round-to-nearest로 고정, s_mult만 margin_loss로 최적화. (STE 버그 —
  scale이 detach돼 grad 끊기던 것 — 수정함)
- 결과(dbg 25초): s_mult가 1.0 → 0.971로 **매끄럽게 이동.** alpha처럼 "안 굳어서 정체"하는
  문제 없음. 단 margin_loss는 여전히 진동(calib 6장이라 이미지별 편차).
- **아직 안 한 것: held-out flip 측정.** s_mult가 움직이는 건 확인했지만, 그게 held-out을
  개선하는지는 미확인.

### 네 번의 시도를 관통하는 핵심 관찰

**margin_loss는 모든 시도에서 줄어들었다(0.03 수준까지). 그런데 held-out flip은 개선되지
않거나 악화됐다.** 이는 09_phase1이 경고한 바로 그 함정("calibration objective 개선 ≠ held-out
개선")의 재현이며, 근본 원인으로 **의심 C(margin 값 보존 ≠ argmax 결정 보존)** 가 가장 유력하다.
margin은 연속값이고 MSE로 맞추는데, held-out flip은 argmax(이산 결정) 기반이라, margin 값을
맞춰도 결정은 다르게 뒤집힐 수 있다.

> **용어 설명**
> - **alpha**: AdaRound에서 각 weight를 올림/내림 할지 정하는 학습 변수. 0 또는 1로 굳어야 함.
> - **rounding**: 반올림. weight를 양자화 격자에 맞출 때 올릴지 내릴지.
> - **h→0/1 수렴률**: alpha가 0 또는 1(확정)로 굳은 비율. 높을수록 rounding이 안정.
> - **soft→hard 전환**: 최적화 중엔 부드러운(soft) 반올림, 확정 시 딱딱한(hard) 반올림으로 바꿈.
> - **reg (regularization)**: alpha를 0/1로 밀어 굳히는 정규화 항.
> - **warm-start**: 다른 방법(AdaRound)의 결과에서 출발해 이어서 최적화하는 것.
> - **s_mult (scale multiplier)**: activation 양자화 scale에 곱하는 학습 가능한 계수. 방향 C의 최적화 대상.
> - **round-to-nearest**: 가장 가까운 격자로 반올림(학습 없는 기본 반올림).
> - **activation scale**: activation을 정수로 바꿀 때 나누는 값. 이게 양자화 정밀도를 좌우.

---

# 파트 5. 내 제안 — 다음 설계 방향

## 요약

지금 상황을 냉정히 보면, 네 번의 시도가 공통으로 "margin_loss는 주는데 held-out은 안 된다"에
부딪혔다. 이건 **목적함수 문제(의심 C)** 일 가능성이 가장 크다. 그래서 방향 C(scale 최적화)로
"안 굳는 문제"는 풀었지만, **목적함수 자체를 margin(값)에서 decision(순위/결정)으로 바꾸지 않으면
같은 벽에 부딪힐 것**으로 본다. 제안: (1) 먼저 방향 C의 held-out을 측정해 "대상 변경만으로 되는지"
확인하고, (2) 안 되면 목적함수를 decision 기반으로 교체하는 것을 다음 핵심 실험으로 삼는다.

## 상세

### 지금 당장 (1스텝): 방향 C의 held-out을 측정

방향 C에서 s_mult가 움직이는 건 봤지만 held-out 개선은 미확인이다. 이걸 먼저 재야 판단이 선다.
- 결과가 **held-out < baseline** 이면: 대상 변경(alpha→scale)만으로 충분했던 것. 방법이 선다.
- 결과가 **held-out ≥ baseline** 이면: 대상이 아니라 목적함수가 범인 = 의심 C 확정.
- 어느 쪽이든 명확한 정보. (가벼운 설정 eval 300, seed 1개로 20초~1분)

### 만약 방향 C도 안 되면 (2스텝): 목적함수를 decision 기반으로

margin(연속값 MSE)이 아니라 **결정(argmax)을 직접 보존**하는 목적함수로 바꾼다. 후보:

- **Ranking loss (순위 보존)**: FP의 유사도 순위를 quant가 유지하도록. pairwise
  "FP에서 A>B면 quant에서도 A>B" 형태. margin 값이 아니라 순서를 지킴.
- **Held-out top-1 분류 손실**: H_cal 프롬프트에 대해, FP의 top-1을 정답처럼 두고
  quant가 그 top-1을 맞추도록 하는 cross-entropy. argmax 결정을 직접 겨냥.
- **Hard-negative contrastive**: 각 region에서 FP top-1 프롬프트 vs 그 의미적 이웃(hard-neg)의
  순위를 유지. C-1에서 본 "손상이 이웃으로 향함"을 정면 대응.

내 직관으로는 **held-out top-1 분류 손실**이 가장 유망하다. 이유: (a) argmax 결정을 직접
보존하므로 의심 C를 정면 해결, (b) 분류 손실은 rounding을 굳히는 데도 우호적(레이어 MSE처럼
"맞다/틀리다"가 명확), (c) 09_phase1의 "GT semantic utility"와도 결이 맞는다.

### 전략적 판단 (병행 고려)

E는 어려운 문제다(09_phase1도 실패). 두 가지를 병행하는 게 안전하다:
- **E를 계속 시도**하되, 목적함수를 decision 기반으로 바꾸는 것에 집중.
- **동시에 A~D의 measurement 기여를 논문의 확실한 바닥으로** 정리해 둔다. E가 끝내 안 되면
  "measurement + 부분적 방법 개선"으로, 되면 "완전한 방법 논문"으로 갈 수 있게.

### 다음 한 걸음 (구체적 실행 제안)

1. 방향 C를 20번 실험에 연결해 held-out 측정 (가벼운 설정). — **즉시**
2. 결과 보고: 개선되면 방향 C 발전(calib 늘리고 multi-seed), 안 되면 3번으로.
3. 목적함수를 held-out top-1 분류 손실로 교체한 버전을 dbg로 시험. — **핵심 다음 실험**

> **용어 설명**
> - **decision 기반 손실**: 유사도 "값"이 아니라 "어느 프롬프트가 1위인가(argmax)"를 직접 맞추는 손실.
> - **ranking loss**: 순위(대소 관계)를 보존하는 손실. 값이 아니라 순서를 지킴.
> - **cross-entropy**: 분류에서 예측 분포와 정답 분포의 차이를 재는 손실.
> - **contrastive**: 정답은 가깝게, 오답(negative)은 멀게 만드는 학습 방식.
> - **measurement contribution**: "문제를 측정·정의"하는 학술 기여. (방법 제안과 대비)
