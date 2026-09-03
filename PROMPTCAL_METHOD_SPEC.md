# PromptCal-PTQ 방법 명세 (현재 구현 기준)

세 번의 시도 후 현 시점의 설계·구현·실패를 정리. 수정/재설계를 위한 기준 문서.

---

## 1. 큰 그림: 무엇을 하려는 방법인가

**목표.** open-vocabulary detector(YOLO-World)를 W8A8 양자화할 때, held-out prompt
(calibration에서 안 본 프롬프트)에 대한 region-prompt 의사결정을 보존한다. D에서 모든
baseline(AdaRound/BRECQ/QDrop/PD-Quant)이 held-out 손상의 80%+를 남긴 것을 넘어서는 게 목표.

**핵심 가설.** "held-out prompt를 calibration에 흉내내어 포함하고, 그 prompt들의 top-k 경계
margin을 FP와 맞추면, 완전히 안 본 prompt에서도 결정이 보존된다(전이된다)."

**무엇을 최적화하나.** weight는 고정. AdaRound식 learnable rounding 변수(alpha)만 최적화.
즉 "어떤 weight를 반올림 up/down 할지"만 학습해서 semantic decision을 지키려 함.

---

## 2. Prompt 3분할 (실험 정직성의 핵심)

COCO-80 프롬프트를 seed마다 무작위로 3분할:

| 그룹 | 개수 | 역할 |
|---|---|---|
| S (seen) | 40 | calibration에 사용, 보이는 프롬프트 |
| H_cal (calib held-out) | 20 | calibration에 **포함**하되 "안 본 프롬프트 흉내"용. 방법이 이걸로 일반화 학습 |
| H_eval (eval held-out) | 20 | 학습에 **절대 안 씀**. 최종 flip 측정 전용 |

- 최적화는 **S + H_cal (60개)** 의 margin을 보존.
- 측정은 **H_eval (20개)** 에서만. → "측정할 걸 학습에 넣는 반칙" 방지.
- H_eval 측정은 05와 동일: H_eval 열 마스킹 후, 원래 H_eval로 판정된 confident anchor에서
  K-only(H_eval 제거) argmax의 FP vs quant 불일치율.

---

## 3. 목적함수 (실제 구현)

### 3.1 margin_loss — 유일한 핵심 손실

```
margin_loss(sim_q, sim_fp, k=5, boundary_w=3.0):
    각 anchor에서 top-(k+1) 유사도 값을 뽑음  → [A, k+1]
    인접 pairwise margin = top[i] - top[i+1]   → [A, k]  (FP와 quant 각각)
    loss = weighted_MSE(quant_margin - fp_margin)
    가중치 w = [1,1,1,1,3]  (마지막 = top-k 경계 margin을 3배 강조)
```

**의도.** 09_phase1 이미지의 통찰("순위 k와 k+1 사이 역전이 실제 detection에 큰 영향")을 반영.
1위-500위가 아니라 top-k 경계의 pairwise margin을 지키려 함.

**대상.** confident anchor(FP maxprob>0.25)에서만, S+H_cal 프롬프트 열에서만 계산.

### 3.2 reg — rounding 정규화 (AdaRound에서 차용)

```
reg = mean over layers of (1 - |2·h(alpha) - 1|^beta)
```
h(alpha)를 0 또는 1로 밀어 hard rounding에 수렴시키는 항. beta annealing으로 후반에 강해짐.

### 3.3 전체 손실 (현재: warmup 방식)

```
전반 40% (warmup): loss = margin_loss + 10.0 · reg   (reg 강하게, h를 0/1로 수렴 의도)
후반 60% (margin) : loss = margin_loss + 0.1 · reg    (margin 주도)
```
+ gradient clipping (max_norm=1.0)

---

## 4. 최적화 절차 (실제 구현)

```
1. wrap_convs(W8A8) → calibrate (naive 양자화 초기화)
2. convert_to_adaround (learnable rounding alpha 생성, soft=True, STE=True)
3. FP 모델로 calib 이미지들의 cv4 유사도 행렬 [A,P] 캐시 (타깃)
4. iters회 반복:
   a. quant 모델 forward → cv4 유사도 행렬 [A,P] (grad 유지, STE로 흐름)
   b. confident anchor + S/H_cal 프롬프트 부분집합 추출
   c. margin_loss + reg 계산 → backward → alpha 업데이트
5. soft=False (hard rounding으로 확정)
```

**핵심 메커니즘.** cv4에 hook(pdquant의 _CV4Capture 재사용), activation STE로 cv4→초기 layer까지
gradient 흐름. alpha만 optimizer에 등록(weight 고정).

---

## 5. 세 번의 시도와 실패 (중요 — 여기서 배워야 함)

### 시도 1: AdaRound refine 방식
- 구조: calibrate → **AdaRound(완전 수렴)** → promptcal refine
- 결과: PromptCal = AdaRound와 **소수점까지 동일** (seed 0: 8.55=8.55)
- 원인: AdaRound가 h를 100% 수렴시킨 뒤라, margin refine이 그 상태를 못 벗어남.
  = PD-Quant가 AdaRound와 같아진 것과 동일한 "warm-start 갇힘".

### 시도 2: 독립 최적화 (AdaRound 제거, lr=1e-2, reg=0.1)
- 결과: held-out **악화** (9.24% → 14.80%, −60%). 3 seed 일관.
- 관찰: h→0/1이 26%에서 멈춤. margin_loss는 감소(0.22→0.03)하나 held-out은 나빠짐.
- 원인 후보: (1) 어중간한 rounding(h 26%)이 soft→hard 전환 시 오류, (2) lr 1e-2 진동,
  (3) confident anchor margin ≠ held-out 경계.

### 시도 3: warmup 추가 (전반 reg 강하게, lr=3e-3)
- 결과: 여전히 **악화** (AdaRound 7.99% → PromptCal 9.07%, −13.6%).
- 관찰: warmup으로도 h가 **32%에서 멈춤**(의도한 90%+ 수렴 실패).
  alpha 총 변화량 446만(폭주 신호). margin_loss는 감소(0.09→0.037).
- 원인: warmup에서 margin_loss와 reg가 동시 작용 → reg가 h를 못 굳힘. 진단 불완전.

### 세 시도의 공통 패턴
- **margin_loss는 항상 줄어드는데(0.03 수준까지) held-out flip은 안 좋아지거나 나빠짐.**
- = 09_phase1의 함정 재현: "calibration objective 개선이 held-out으로 전이 안 됨."
- PromptCal이 rounding을 AdaRound만큼 못 굳힘(h가 26~32%에서 정체).

---

## 6. 근본적으로 의심되는 지점 (재설계 시 고려)

### 의심 A: alpha(rounding)만으로 held-out 경계를 형성하기엔 자유도가 부족/부적합
weight 반올림 방향(up/down)만으로 "안 본 프롬프트의 결정 경계"를 만드는 게 결이 안 맞을 수
있음. AdaRound reg 없이는 h가 안 굳고, reg를 넣으면 AdaRound로 수렴. 딜레마.

### 의심 B: confident anchor의 margin ≠ held-out flip이 일어나는 영역
- 최적화: confident anchor(maxprob>0.25) & S/H_cal 프롬프트의 margin.
- 측정: H_eval로 판정된 anchor에서, H_eval 제거 후 K-only argmax flip.
- 이 두 영역이 다르면, margin을 아무리 맞춰도 측정 지표는 안 움직임.

### 의심 C: margin(값) 보존 ≠ decision(순위) 보존
margin은 연속값. 이걸 MSE로 맞춰도 argmax(이산 결정)는 다르게 뒤집힐 수 있음.
held-out flip은 argmax 기반인데 우리는 margin 값을 맞추고 있음. 목적-측정 불일치.

### 의심 D: soft→hard 전환의 불연속
h가 0/1로 안 굳은 상태(26~32%)에서 hard로 확정하면, soft로 최적화한 것과 다른 모델이 됨.
이 전환 오류가 held-out을 악화시킬 수 있음. (AdaRound는 h 100%라 이 문제 없음)

---

## 7. 아직 구현 안 된 원래 설계 요소 (참고)

원래 3요소 설계 중 현재 구현된 건 ①의 일부 + ②의 일부뿐:
- ① Semantic Calibration Set: **부분 구현** (H_cal 포함 O, hard-negative 주입 X)
- ② Decision-Preserving Objective: **margin 버전만 구현** (ranking/identity 버전 X)
- ③ GT Semantic Utility Gate: **구현 O** (H_eval 측정, multi-seed) — 이건 잘 작동, 실패를 정직히 잡아냄
- 보조 LLAQ: 미구현

---

## 8. 재설계 시 선택지 (방향 옵션)

**옵션 1: rounding을 확실히 굳히고(순수 reg로 h 100%) 그 위에 margin.**
= 시도 3의 warmup을 제대로 작동시킴. 단 이러면 AdaRound에 가까워질 위험(시도 1).

**옵션 2: 목적함수를 margin(값)에서 decision(순위/identity)으로.**
held-out top-1 identity 보존, 또는 hard-negative 대비 순위 손실. 의심 C 대응.

**옵션 3: 최적화 대상을 alpha에서 activation scale/clip으로.**
rounding(alpha) 대신 activation 양자화 파라미터를 최적화. 의심 A 대응.

**옵션 4: 최적화 영역과 측정 영역을 일치.**
confident anchor가 아니라 "held-out 경계 근처 anchor"를 타깃. 의심 B 대응.

**옵션 5: 전략 재검토.**
E�� 어려우면(09_phase1도 실패) A~D의 measurement contribution을 주력으로.
