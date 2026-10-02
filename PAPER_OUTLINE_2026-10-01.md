# 논문 개요 (초안, 2026-10-01 작성, 10-02 갱신: 프로토콜 확정)

**대상 독자:** 공저자. 논문의 이야기 흐름, 각 절에 들어갈 내용, 그리고 지금 근거가 있는 것과 없는 것을 한 번에 본다.

**관련 문서**
- 방법의 구조와 작동 원리: [`PROMPTCAL_METHOD_LOWBIT_2026-09-29.md`](PROMPTCAL_METHOD_LOWBIT_2026-09-29.md)
- 6 seed 수치: [`docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md`](docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md)
- 경위: [`PROMPTCAL_HISTORY_2026-09-30.md`](PROMPTCAL_HISTORY_2026-09-30.md)
- 코드와 실행: [`pipeline/README.md`](pipeline/README.md)

**10-02 확정 사항**
- **프로토콜:** head 포함, 첫·마지막 8bit, attention·contrastive matmul 8bit, head 마지막 conv 입력 A16. 모든 모델에 같게 적용한다.
- **방법:** G + P + M + BRECQ. M은 모델별 누수 없는 사전 검사로 채택한다(m에서는 꺼짐).
- **일반화 (2 seed):** s, v2, m 모두 BRECQ보다 좋아졌다. 최종 표는 확정 프로토콜로 다시 만든다.

**상태 표시:** ✅ 근거 있음(6 seed) · 🟡 일부 있음(2 seed 또는 일부 설정) · ⏳ 진행 중 · ❌ 아직 없음

---

## 0. 한 문장 요약

open-vocabulary 검출기를 저비트로 PTQ하면 **calibration 때 보지 않은 어휘의 판정이 AP보다 먼저 무너진다.** 이 손상은 **텍스트 융합 conv와 초반 backbone 몇 곳에 몰려 있다.** 우리는 그 지점을 **구조 재배치(게이트 교환), 누수 없는 진단 기반 보호, 배포 제약을 지키는 scale 이전**으로 막는다. 그 결과 학습 없는 PTQ로 W4A5까지 held-out 어휘 판정을 지킨다.

---

## 1. 제목 후보

1. *Where Open-Vocabulary Detectors Break Under Low-Bit Quantization — and How to Fix It Without Training*
2. *Gate Commutation: Structure-Aware Post-Training Quantization for Open-Vocabulary Detectors*
3. *Preserving Held-Out Vocabulary in Low-Bit Open-Vocabulary Detection*

1번은 분석과 방법을 함께, 2번은 방법을, 3번은 문제를 앞에 세운다. novelty가 게이트 교환과 분석에 있으므로 1번이나 2번을 권한다.

---

## 2. 초록 (초안)

> Open-vocabulary detectors such as YOLO-World are attractive for edge deployment, where vocabularies change after deployment. We show that standard post-training quantization (PTQ) damages them in a way that AP on the calibration vocabulary hides: at W4A8, BRECQ flips the top-1 class of 18% of confident LVIS predictions while COCO AP drops by only 3.6 points. Using leak-free, layer-wise diagnosis, we find that the damage concentrates in a handful of layers. Most of it comes from the 1×1 convolution that fuses the text-gated attention branch, where a few outlier channels of the gated branch corrupt the shared quantization scale. The rest comes from early backbone split/concat boundaries. We propose three training-free, deployment-compatible fixes: (i) **gate commutation**, an FP-equivalent restructuring that moves the text gate behind its own 1×1 convolution, at zero bit cost; (ii) **leak-free protection** of the few remaining sensitive convolutions at 8 bit (+0.3% model size); and (iii) **tied scale migration** that respects producer–consumer constraints so it can be folded at deployment. Migration is enabled per model only when a leak-free check on held-out calibration-domain images does not increase decision flips. With all convolutions, attention and region–text matmuls quantized, on top of BRECQ, the method improves LVIS AP of YOLO-World-S by +1.2 to +2.5 points from W4A8 to W4A5 and reduces LVIS decision flips by about 30% relative, while QDrop, AdaRound and calibration-vocabulary fine-tuning fall below BRECQ. It transfers to YOLO-World-S v2 (up to +7.9 points) and YOLO-World-M (up to +3.2 points).

s의 숫자는 확정 프로토콜 6 seed(표 1, runs/158)다. W4A5는 seed 0(기준선 붕괴)을 뺀 값이다. 일반화 숫자는 확정 프로토콜 2 seed다.

---

## 3. 기여 (Introduction 마지막 문단)

| # | 기여 | 상태 |
|---|---|---|
| C1 | **문제 정의:** 저비트 PTQ는 held-out 어휘 판정을 AP보다 먼저, 더 크게 무너뜨린다. LVIS가 COCO보다 2~3배 민감하다. 판정 충실도 지표(LVIS_flip, lost)를 함께 보고해야 한다 | ✅ |
| C2 | **원인 분석:** 손상 위치(텍스트 융합 conv, 초반 backbone)와 메커니즘(게이트된 attention 분기의 outlier가 공유 scale을 오염). 누수 없는 진단 절차. 보호 목록이 프로토콜 변경에도 그대로다 | ✅ s-v1 · 🟡 v2·m (보호 목록은 확정) |
| C3 | **게이트 교환(G):** FP 등가, 비트 비용 0. open-vocab 검출기의 텍스트 게이팅 구조를 이용한 양자화 전용 재배치 | ✅ s-v1 (A6·A5) · 🟡 v2, **m (W4A5 +2.3~3.2, 2 seed)** |
| C4 | **완전한 학습 없는 파이프라인 GPM:** 보호와 공유 제약 이전을 더해 BRECQ·QDrop 위에 plug-in으로 동작하고, 배포형 등가성을 검증. **이전은 누수 없는 사전 검사로 모델별 채택** | ✅ s-v1 · 🟡 일반화 (s·v2·m 2 seed 모두 개선) |
| C5 | **엄격한 프로토콜:** head 포함, attention·contrastive matmul까지 8bit(LVIS 기준 행렬 연산 ~100% 정수) | ✅ W4A8·W4A5 6 seed |

---

## 4. 본문 구성

### 1. Introduction (1~1.25쪽)
- 동기: 엣지에서 어휘가 바뀌는 검출(로봇, 모바일). 배포 뒤에 어휘가 바뀌므로 calibration 어휘 밖에서의 충실도가 중요하다.
- 관찰: AP만 보면 W4A8 BRECQ가 쓸 만해 보이지만, LVIS 판정의 18%가 FP와 달라진다(**그림 1**).
- 원인 한 줄과 해법 한 줄, 그리고 기여 C1~C5.

### 2. Related Work (0.75쪽)
- **PTQ:** AdaRound, BRECQ, QDrop, PD-Quant. 검출용으로는 Reg-PTQ.
- **outlier 처리와 scale 이전:** SmoothQuant, AWQ, OmniQuant 계열. 우리 이전(M)과의 차이는 **conv 그래프에서 배포 제약(생산자 공유, residual add)을 지킨다**는 점이다.
- **혼합 정밀도:** HAWQ 계열. 차이는 **누수 없는 판정 기반 진단**이고, 같은 예산의 HAWQ식 보호와 비교한다.
- **OVOD:** YOLO-World, YOLOE, GLIP, Grounding DINO, OWL-ViT.
- **OVOD 양자화:** QATMA(QAT, YOLO-World M/L/X)가 가장 가깝다. QATMA는 PTQ가 W4A4에서 완전히 실패한다고 보고하고, 층별 원인 분석은 하지 않는다.

### 3. Analysis: Where Quantization Breaks OVOD (1.5쪽)
- **3.1 설정과 판정 충실도 지표**
  - LVIS_flip: FP가 확신하는 anchor에서 top-1 클래스가 바뀐 비율. lost: 맞던 것이 틀린 수.
  - COCO = calibration 어휘, LVIS = held-out 어휘.
  - 지표 선택의 근거: AP보다 먼저 움직이고 seed 분산이 작다.
- **3.2 손상은 몇 곳에 몰려 있다**
  - conv 하나만 W4로 내리는 진단. **그림 2: 층별 민감도 지도.**
  - 12.cv2 하나가 LVIS flip을 6.5%에서 79%로 만든다.
- **3.3 메커니즘**
  - C2fAttn concat 입력 중 attention 분기의 activation과 weight가 동시에 크다(중앙값의 87배).
  - 이 분기가 출력 채널별 weight 범위를 지배해서 다른 세 분기를 0 근처로 뭉갠다. **그림 3.**
  - 그 결과 확신하는 판정도 뒤집힌다(margin > 2인 판정의 70% 이상).
  - LVIS는 비슷한 클래스가 많아(점수 차가 작은 anchor가 COCO의 18배) 같은 손상에 더 크게 흔들린다.
- **3.4 이전(migration)은 취약성을 옮긴다**
  - 12.cv2의 민감도는 줄지만 1.conv의 민감도는 3.5~4배 는다. 그래서 보호가 함께 필요하다.

### 4. Method (1.5쪽)
- **4.1 게이트 교환 (G).** 식 `cv2(cat(y0,y1,y2,aw⊙p)) = W_main·cat(y0,y1,y2) + b + Σ_h aw_h⊙(W_h·p_h)`, **그림 4: 구조도.**
  - 효과 세 가지: scale 분리, 게이트 전 값을 양자화, 게이트가 작은 위치에서 오차 축소.
  - naive 12번 블록 실험에서 LVIS flip 68.8%가 9.7%로 줄었다. scale만 분리하면 18.5%다.
- **4.2 누수 없는 보호 (P).** train2017 200장, COCO 어휘만 써서 conv별 flip 증가를 재고, 예산 1.5% 안에서 탐욕 선택한다.
  - G를 쓰면 C2fAttn cv2는 목록에서 빠지고 초반 backbone 3개만 남는다(+0.3%).
- **4.3 공유 제약 scale 이전 (M).** 채널 계보 추적(측정 기반), union-find로 공유 그룹을 만들고, α를 단위별로 고른다(s=1 포함).
  - **채택 검사:** calibration 밖 train2017 200장과 COCO 어휘에서 M 켬/끔의 flip을 비교한다. 1.5배를 넘으면 끈다. 모델당 한 번만 한다.
  - 평균 임베딩 오차로는 해를 잡지 못했다는 관찰을 함께 싣는다(판정 기준이 필요한 이유).
  - 배포형 등가성: 상대 오차 5.5e-6.
- **4.4 BRECQ/QDrop과의 결합.** 빌드 순서, 그리고 FP 기준 모델의 구조 정합.

### 5. Experiments (2.5쪽)
- **5.1 설정:** 모델, 데이터(train2017 256장, COCO val, LVIS minival), 프로토콜(head 포함, 첫·마지막 레이어 8bit, attention·contrastive 8bit, head 마지막 conv 입력 A16), seed 6개, 짝 Δ ± 95% CI와 Holm 보정. 붕괴한 기준선 seed는 밝혀 적는다.
- **5.2 주 결과 (표 1):** W8A8 / W4A8 / W4A6 / W4A5. 기준선은 naive, AdaRound, BRECQ, QDrop, PromptCal(Combined)이고 우리는 PM, GPM이다. 크기를 함께 적는다.
- **5.3 ablation (표 2):** G / P / M / GM / PM / GPM. 같은 예산의 무작위 보호(R), HAWQ식 보호(H), 채널 단위 W8, 독립 s 이전.
- **5.4 plug-in (표 3):** QDrop + PM.
- **5.5 일반화 (표 4):** v2(PM/GPM), m(GP, M은 검사에서 꺼짐), (l), YOLOE. 모델별 M 검사 결과를 함께 싣는다.
- **5.6 프로토콜 엄격도 (표 5):** attention·contrastive FP와 8bit 비교(6 seed, 결론 유지). 마지막 conv 입력 A8 vs A16(m의 seed 붕괴). QATMA와 같은 조건 비교.
- **5.7 비용:** 모델 크기, PTQ 제작 시간, (latency).

### 6. Discussion & Limitations (0.5쪽)
- **W4A4는 열지 못했다.** conv의 76%를 A8로 되돌려도 붕괴한다. per-tensor A4가 절벽이다.
- **W4A8에서는 G의 추가 이득이 없다.** PM과 GPM이 동급이다.
- **calibration 어휘 기반 학습(PromptCal)은 held-out을 해친다.** 음성 결과로 싣는다.
- **실제 기기 가속은 미검증이다.** W4 커널 지원에 의존한다.
- **이전(M)은 모든 모델에 이롭지 않다.** m에서는 해롭고, 사전 검사로 끈다. 검사는 COCO 어휘로만 보므로 v2 W8A8처럼 LVIS 판정에서만 나빠지는 경우는 잡지 못한다.
- **기준선(BRECQ)이 seed에 따라 붕괴한다.** s W4A5 seed 0에서 프로토콜과 무관하게 재현된다. 우리 방법은 같은 seed에서 정상이다.

---

## 5. 그림과 표 계획

| 번호 | 내용 | 상태 |
|---|---|---|
| 그림 1 | 동기: 비트별 COCO AP와 LVIS_flip (AP는 버티는데 flip은 무너짐) | 데이터 ✅, 그림 ❌ |
| 그림 2 | 층별 민감도 지도 (conv 하나만 W4일 때 flip 증가) | 데이터 ✅ (runs/128, 134), 그림 ❌ |
| 그림 3 | C2fAttn 분기별 activation과 weight 통계, 공유 scale 오염 | 데이터 ✅ (runs/139), 그림 ❌ |
| 그림 4 | 게이트 교환 구조도 | ❌ |
| 그림 5 | 이전이 취약성을 옮기는 모습 (이전 전후 민감도) | 데이터 ✅ (runs/148), 그림 ❌ |
| 표 1 | 주 결과 4개 설정 × 기준선 | ✅ 확정 프로토콜 6 seed (`docs/PROMPTCAL_RESULTS_2026-10-03.md` §1) |
| 표 2 | ablation | ✅ |
| 표 3 | QDrop plug-in | ✅ |
| 표 4 | 일반화 | 🟡 |
| 표 5 | attention·contrastive 8bit, QATMA 조건 | 🟡 / ⏳ |

---

## 6. 주장과 근거 대조

| 주장 | 근거 | 상태 | 막히면 |
|---|---|---|---|
| held-out 판정이 AP보다 먼저 무너진다 | 모든 설정, 6 seed | ✅ | – |
| 손상이 몇 conv에 몰려 있다 | 층별 진단, 무작위 보호 대조 | ✅ s · 🟡 v2·m | – |
| G는 비트 비용 0으로 저비트를 개선한다 | W4A6·W4A5 6/6 | ✅ | – |
| GPM이 모든 비트에서 최선이다 | W4A8에서는 PM이 LVIS AP로 약간 앞섬. v2는 AP는 PM, flip은 GPM | ❌ | "G는 저비트에서 판정 보존에 기여, 최종 구성은 설정·모델별"로 서술 |
| 기존 PTQ 위에 plug-in으로 동작한다 | QDrop + PM 6 seed | ✅ | – |
| YOLO-World 계열에서 일반화된다 | 확정 프로토콜 2 seed: s, v2(+6~8), m(G +0.8~3.2) 모두 개선 | 🟡 (3 seed 확장 예정) | – |
| 다른 OVOD 구조로 일반화된다 | YOLOE 미실행 | ❌ | 주장 범위를 YOLO-World 계열로 축소 |
| attention까지 정수화해도 유지된다 | W4A8·W4A5 6 seed에서 유지, 짝 차이 변화 0.3점 이내 | ✅ | – |
| M 채택 검사가 해로운 경우를 걸러낸다 | m 3경우 끔, s 4경우 켬, 실제 평가와 일치. v2 W8A8은 놓침 | 🟡 | 한계로 명시 |
| 엣지에서 추가 비용이 없다 | 크기·연산량 분석만 있음 | 🟡 | latency 측정 없이 "연산 형태 동일"까지만 서술 |

---

## 7. 남은 일 (논문 기준 우선순위)

1. ~~프로토콜 확정~~ (10-02): A16 + `attn_cls`, M은 모델별 사전 검사.
2. ~~일반화 1단계~~ (10-02): 확정 프로토콜로 s / v2 / m 2 seed. 세 모델 모두 개선.
3. **최종 6 seed 일괄 (s):** ~~표 1~~ 완료(10-03). 표 2(W4A5 ablation)·표 3(선택 기준 대조, QDrop plug-in) 진행 중.
4. **일반화 3 seed:** v2(PM/GPM), m(GP).
5. **YOLOE-v8s:** 구조가 다른 OVOD. 코드 지원부터.
6. **보강:** SmoothQuant·AWQ 기준선, Reg-PTQ, QATMA 조건(첫·마지막 FP, YOLO-World-L).
7. **그림 1~5 작성.**
8. **(가능하면) 엣지 latency 측정.**

---

## 8. 리뷰어 예상 질문

| 질문 | 준비된 답 |
|---|---|
| "혼합 정밀도는 흔한 기법 아닌가?" | 보호 자체는 기여의 중심이 아니다. 같은 예산의 무작위·HAWQ식 보호와 비교해 위치 선택이 결정적임을 보이고, 새로움은 G(구조)와 분석에 둔다. 보호 비용도 +0.3%다 |
| "attention과 DFL을 FP로 둔 것 아닌가?" | attention·contrastive 8bit에서도 결론이 유지된다(표 5). DFL은 고정 디코딩 연산이다 |
| "QAT(QATMA)와 비교하면?" | 학습 없이 W4A5에서 동작한다. QATMA와 같은 조건 비교는 표 5에 둔다 |
| "LVIS_flip은 왜 믿을 만한가?" | AP와 함께 보고하고, Holm 보정한 주 지표로 쓴다. seed 분산이 작고 AP보다 먼저 움직인다 |
| "다른 OVOD에서도 되나?" | YOLO-World 세 모델(s, v2, m)에서 재현된다. YOLOE는 진행 예정이고, G를 적용할 수 없으므로 P(+M 검사)만 확인한다 |
| "이전(M)을 모델마다 고르는 건 튜닝 아닌가?" | 평가를 보지 않는 검사(calibration 밖 이미지, COCO 어휘)로 모델당 한 번 정한다. 문턱을 1.1~2배 어디에 둬도 결정이 같다 |
| "W4A4는?" | 열지 못했다. per-tensor A4가 이 모델의 절벽이고, 한계로 명시한다 |
