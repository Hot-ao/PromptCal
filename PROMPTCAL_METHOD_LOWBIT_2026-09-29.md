# 저비트 PTQ 방법 — 구조와 작동 원리 (2026-09-29, 09-30·10-01·10-02·10-03 갱신)

**대상 독자:** 이 방법을 이어서 구현·실험하거나 논문을 쓰는 공저자.

**목적:** "무엇을 하는 방법인가, 모델의 어디를 어떻게 바꾸는가, 왜 그게 통하는가"를 한 문서에서 설명한다.

**관련 문서**
- **논문에 쓰는 수치는 확정 프로토콜 결과 [`docs/PROMPTCAL_RESULTS_2026-10-03.md`](docs/PROMPTCAL_RESULTS_2026-10-03.md)가 기준이다.**
- 이전 프로토콜의 수치와 원인 분석 경위는 [`docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md`](docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md)(6-seed, 게이트 교환)와 [`docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-29.md`](docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-29.md)(원인 분석)에 있다(run 번호로 추적).
- 코드 구성과 실행은 [`pipeline/README.md`](pipeline/README.md)에 있다.
- **09-30 변경:** 텍스트 게이트 교환(⓪, §4.4)을 방법에 추가했다. 저비트(A6·A5)에서는 GPM이 최종 구성이다.
- **10-01 변경:**
  - attention(Linear/matmul)과 head contrastive matmul도 8bit로 양자화하는 옵션을 추가했다(`--attn-quant`, §3.3). 2 seed에서 결론(GPM − BRECQ)이 그대로 유지된다.
  - 다른 모델로 일반화한 결과와 m 모델의 불안정 원인을 §9에 기록했다.
- **10-02 변경 (프로토콜 확정):**
  - **확정 프로토콜 = head 포함 + 첫·마지막 8bit + attention·contrastive 8bit(`attn_cls`) + head 마지막 conv 입력 A16.** 모든 모델에 같게 적용한다(§3.4). 공통 인자는 `pipeline/scripts/protocol.sh`에 있다.
  - A16 도입으로 생긴 BRECQ 버그(16bit activation에 LSQ를 걸면 step이 발산)를 고쳤다(§3.4).
  - **이전(M)은 모델별 누수 없는 사전 검사로 채택 여부를 정한다**(`+A`, §4.5). m에서는 꺼지고(최종 GP), s·v2에서는 켜진다.
  - 새 프로토콜에서 s, v2, m 세 모델 모두 BRECQ보다 좋아졌다(2 seed, §9).
- **10-03 변경 (표 1 확정):**
  - 확정 프로토콜로 s의 주 결과를 6 seed로 다시 냈다(§1, runs/158).
  - W4에서는 결론이 그대로다. W4A8은 PM, W4A6·W4A5는 GPM이 최선이다.
  - W8A8의 M 효과는 이전 프로토콜보다 작아졌다(LVIS AP +0.32 → +0.20, flip −1.28 → −0.37%p).
  - 표 2(W4A5 ablation), 표 3(선택 기준 대조, QDrop plug-in)도 6 seed로 확정했다. 세 요소가 모두 기여하고, M은 G나 P 위에서만 이득이다. 무작위 보호는 효과가 거의 없다.
  - **방법을 GPM 하나로 고정했다.** 비트마다 평가 결과를 보고 구성을 고르지 않는다. PM은 변형(ablation)으로 함께 싣는다.
  - W8A8 GPM과 QDrop + GPM을 6 seed로 추가했다(runs/159). W8A8 GPM은 LVIS_flip −24%로 M 단독보다 확실히 낫다. QDrop + GPM은 BRECQ + GPM과 같은 수준이다.
- 이전 설계(Combined = BRECQ + s_mult margin 학습)는 [`PROMPTCAL_PAPER_DESIGN_2026-09-28.md`](PROMPTCAL_PAPER_DESIGN_2026-09-28.md)에 있다. 이 문서의 방법은 그 설계를 대체한다. 이유는 §7에 있다.

---

## 1. 한눈에 보기

**문제.** open-vocabulary 검출기(YOLO-World)를 저비트로 양자화하면 AP보다 먼저 **held-out 어휘의 판정 순위**가 무너진다. held-out 어휘는 calibration 때 보지 않은 LVIS 1203클래스 같은 것이다.
- W8A8 BRECQ에서도 LVIS 판정의 약 4%가 FP와 다른 클래스로 바뀐다.
- W4A8에서는 18%가 바뀐다.

**관찰.** 이 손상은 모델 전체에 고르게 퍼져 있지 않다. 두 곳에 몰려 있다.
1. **텍스트 융합 지점:** neck의 `C2fAttn` 블록에서 텍스트 가이드 attention 출력이 concat으로 합쳐진 직후의 1×1 conv. 특히 `12.cv2`다.
2. **초반 backbone:** stem 직후 conv(`1`)와 첫 C2f(`2`, `4`)의 concat 경계.

**방법.** 같은 절차를 모든 비트 설정과 모델에 적용한다. 이전(②)만 모델별 사전 검사(§4.5)로 켤지 정한다.

```
 ⓪ 게이트 교환(Gate commute) : C2fAttn의 텍스트 게이트를 1×1 conv 뒤로 옮기는 FP 등가 재배치 (비트 비용 0)   [09-30 추가]
 ① 보호(Protect)   : 누수 없는 진단으로 고른 소수 conv의 weight를 8bit로 유지 (예산 = weight 파라미터의 1.5%)
                     ⓪을 쓰면 C2fAttn cv2(12.cv2 등)는 목록에서 빠진다 -> 실제 보호는 1, 2.cv2, 4.cv1 (+0.3%)
 ② 이전(Migrate)   : 입력 채널별 scale s를 생산 conv → 소비 conv weight로 옮김 (배포 제약을 지키는 공유 s)
                     모델별 사전 검사(calibration 밖 이미지 + COCO 어휘의 flip)에서 해로우면 끈다   [10-02 추가]
 ③ 재구성(BRECQ)   : 표준 블록 재구성 PTQ (rounding + activation step 학습). QDrop으로 바꿔도 된다
```

- 이름: ①+② = **PM**, ⓪+①+② = **GPM**, ⓪+① = **GP** (조건 접미사, §5). `+A`는 ②를 사전 검사로 켤지 정한다.
- **우리 방법 = GPM (+ ②의 사전 검사).** 모든 비트 설정과 모델에 같은 구성을 쓴다. 비트마다 평가 결과를 보고 구성을 바꾸지 않는다.
  - W8A8에서는 ①이 규칙상 아무것도 하지 않으므로 실제로는 ⓪+②다.
  - m은 사전 검사에서 ②가 꺼져 ⓪+①이 된다. m의 보호 대상은 C2fAttn cv2뿐이라 ⓪이 대신하므로, 실제로는 ⓪만 남는다.
  - **PM은 변형으로 함께 싣는다.** W4A8에서는 PM이 크기를 1% 더 쓰고 LVIS AP를 0.34점 더 얻는다(표 1: GPM +0.81, PM +1.15, flip은 같음). W4A6·W4A5에서는 GPM이 PM보다 작고 좋다.
- **W4A8에서 PM이 조금 나은 이유:** W4A8의 주 병목은 weight다. PM은 12.cv2 전체를 W8로 올리지만, GPM은 ⓪이 12.cv2를 대신 다루므로 그 weight가 4bit로 남는다. A6·A5에서는 activation이 병목이 되어, 텍스트 분기의 activation scale을 분리하는 ⓪이 더 중요해진다(표 2: G 단독 +1.37 > P 단독 +1.04).
- W8A8에서는 모든 weight가 이미 8bit라 ①이 규칙상 아무것도 하지 않는다(비용 0). ⓪+②로 LVIS AP +0.31, flip 3.83 → 2.91%(−24%)다. M 단독(+0.20, −0.37%p)보다 크다.
- **W4A6·W4A5에서는 GPM이 최선이다:** PM보다 작은 모델로 LVIS AP와 LVIS_flip이 모두 더 좋다(확정 프로토콜 표 1: W4A6 +1.66 vs +1.22, W4A5 +2.53 vs +2.15).

**결과 요약 (확정 프로토콜, s-World, 6 seed, BRECQ → GPM) — 표 1, runs/158**

| 설정 | 크기 | COCO AP | LVIS AP | LVIS_flip | Δ LVIS AP (95% CI) | 변형 PM의 Δ LVIS AP (크기) |
|---|---|---|---|---|---|---|
| W8A8 | +0% | 36.75 → 36.78 | 25.46 → 25.77 | 3.83 → 2.91% | +0.31 ± 0.12 | – (W8에서 P는 효과 없음. M 단독 +0.20) |
| W4A8 | +0.3% | 33.77 → 34.81 | 22.74 → 23.55 | 17.58 → 12.32% | +0.81 ± 0.15 | +1.15 ± 0.16 (+1.3%) |
| W4A6 | +0.3% | 32.42 → 34.11 | 21.52 → 23.18 | 20.61 → 14.77% | +1.66 ± 0.26 | +1.22 ± 0.18 (+1.3%) |
| W4A5 | +0.3% | 29.90 → 32.81 † | 19.76 → 22.29 † | 28.04 → 19.60% † | +2.53 ± 0.23 † | +2.15 ± 0.31 † (+1.3%) |

- FP32는 COCO AP 36.80, LVIS AP 25.89다.
- † W4A5는 seed 0에서 BRECQ·QDrop이 붕괴(COCO 19.3 / 9.8)해서 seed 0을 뺀 5 seed 값을 적었다. 6 seed 전체는 GPM +3.96 ± 3.67(Holm 후 비유의)이다.
- 전체 표(기준선 포함, p_Holm)는 [`docs/PROMPTCAL_RESULTS_2026-10-03.md`](docs/PROMPTCAL_RESULTS_2026-10-03.md) §1에 있다.
- 기준선: QDrop은 저비트일수록 BRECQ보다 나쁘고(W4A5 −3.9), AdaRound는 W4A8에서 붕괴하며, PromptCal(Combined)은 BRECQ보다 나쁘다(W4A8 −1.06).

**ablation과 대조 (확정 프로토콜, 6 seed) — 표 2·3**

| 항목 | 결과 (LVIS AP 짝 Δ, flip) |
|---|---|
| W4A5 단일 요소 (vs BRECQ, seed 0 제외) | G +1.37 / −5.4%p, P +1.04 / −3.7%p, M −0.75 (비유의) |
| W4A5 요소 추가 | G를 PM 위에 +0.39 / −1.33%p, M을 G 위에 +0.59, M을 GP 위에 +0.60 |
| W4A8 같은 예산 보호 (vs BRECQ) | 무작위 R +0.21 / −0.5%p, HAWQ식 H +0.90 / −5.5%p, P +1.05 / −5.7%p |
| QDrop 위 plug-in (vs QDrop) | GPM: W8A8 +0.45, W4A8 +1.12, W4A6 +3.04, W4A5 +6.42. QDrop + GPM ≈ BRECQ + GPM(±0.2 이내) |

- W4A5 seed 0에서는 BRECQ와 P 단독이 무너지고, G나 M이 들어간 조합은 모두 정상이다.
- 이전 프로토콜의 6 seed 결과(W8A8 +0.32, W4A8 +1.36, W4A6 +1.59, W4A5 +2.70)는 `docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md`에 남아 있다. W8A8을 빼면 확정 프로토콜과 같은 크기다.

**일반화 (확정 프로토콜, 2 seed, runs/154): LVIS AP, BRECQ 대비 차이(점)**

| 모델 | W4A8 | W4A5 | 최종 구성 |
|---|---|---|---|
| YOLOv8s-World | 6 seed 표 1 참고: PM +1.15 | 6 seed 표 1 참고: GPM +2.53 (seed 0 제외) | PM / GPM |
| YOLOv8s-WorldV2 | PM +6.54 / +6.39 | PM +7.32 / +7.86, GPM +5.66 / +6.13 | PM (flip은 GPM이 더 낮음) |
| YOLOv8m-World | G +0.77 / +1.50 | **G +3.17 / +2.26** | GP (= G. M은 검사에서 꺼짐) |

---

## 2. 대상 모델의 구조 (YOLOv8s-World)

### 2.1 전체 흐름

```
 이미지 640×640
   │
   ▼  backbone  (model.0 ~ model.9)
   │   0 Conv(stem) → 1 Conv → 2 C2f → 3 Conv → 4 C2f → 5 Conv → 6 C2f → 7 Conv → 8 C2f → 9 SPPF
   │                                         (P3)              (P4)              (P5)
   ▼  neck  (model.10 ~ model.22) — 텍스트가 여기서 이미지 특징에 섞인다
   │   10 Upsample → 11 Concat(P5↑, P4) → 12 C2fAttn ─┐
   │   13 Upsample → 14 Concat(↑, P3)  → 15 C2fAttn ──┼─→ P3 출력
   │   16 ImagePoolingAttn (이미지 → 텍스트 임베딩 보정)
   │   17 Conv(↓) → 18 Concat → 19 C2fAttn ──────────┼─→ P4 출력
   │   20 Conv(↓) → 21 Concat → 22 C2fAttn ──────────┴─→ P5 출력
   ▼  head  (model.23, WorldDetect) — 레벨(P3/P4/P5)마다
       cv2[l] : Conv → Conv → 1×1        → 박스 분포 (DFL로 디코딩)
       cv3[l] : Conv → Conv → 1×1        → region 임베딩 x (512차원)
       cv4[l] : ContrastiveHead           → 클래스 점수 = τ · cos(x, t_j) + b   (conv 없음, 영역×텍스트 matmul)

 텍스트: 클래스 이름 → CLIP 텍스트 인코더(FP, 오프라인) → t_j
         t_j는 두 곳에 쓰인다: ① neck C2fAttn의 guide, ② head cv4의 유사도
```

- 클래스는 텍스트 프롬프트로 정한다. 어휘를 바꾸면 t_j만 바뀐다.
- 평가는 COCO-80(calibration 어휘)과 LVIS-1203(held-out 어휘)에서 한다.

### 2.2 C2fAttn: 이 방법의 핵심 지점
`ultralytics/nn/modules/block.py`의 `C2fAttn.forward`는 다음과 같다(블록 12 기준, 채널 수 c = 128).

```
 x ─ cv1 (1×1) ─ chunk ─┬─ y0 (128ch) ───────────────────────────────────────┐
                        └─ y1 (128ch) ─┬──────────────────────────────────────┤
                                       ▼                                      │
                                   m.0 Bottleneck (3×3, 3×3) ─ y2 (128ch) ─┬───┤
                                                                         ▼   │
                                   attn = MaxSigmoidAttnBlock(y2, 텍스트) ─ y3 (128ch)
                                                                             │
                     cv2 (1×1)  ◀── concat(y0, y1, y2, y3)  (512ch) ◀────────┘
```

`MaxSigmoidAttnBlock`은 이렇게 동작한다.

```
 aw  = sigmoid( max_j ⟨embed(y2), guide_j⟩ / √d + bias ) × scale   # 위치별 "어떤 프롬프트와든 잘 맞는가"
 y3  = proj_conv(y2) × aw                                            # 채널 전체에 공간 게이트를 곱함
```

- 여기서 embed(y2) = y2이고, guide = Linear(텍스트 임베딩)이다.
- **y3(텍스트 attention 출력)만 다른 세 분기와 통계가 다르다.**

| 분기 (12.cv2 입력) | activation 평균 \|x\| | activation 최대 | 받는 weight 열의 최대 (중앙 / 최대) |
|---|---|---|---|
| y0, y1 (cv1 조각) | 0.44 / 0.34 | 7.7 / 7.0 | 0.27 / 0.81, 0.25 / 0.64 |
| y2 (bottleneck) | 0.45 | 9.3 | 0.22 / 0.89 |
| **y3 (텍스트 attention)** | **1.62** | **20.4** | 0.25 / **3.32** |

- y3의 일부 채널은 activation과 그 채널을 받는 weight가 **동시에** 크다.
- 이 채널들이 모델 전체의 outlier 1·2위다(해당 conv 안 중앙값의 87배, 79배).

---

## 3. 양자화가 어떻게 시뮬레이션되는가

### 3.1 fake-quant 모듈
`pipeline/quant/quant_model.py::wrap_convs`가 모든 `nn.Conv2d`를 `QuantConv2d`(`pipeline/quant/fake_quant.py`)로 감싼다. 70개이고, CLIP과 DFL은 제외한다. 양자화는 conv 입력과 weight에서 일어난다.

```
 QuantConv2d.forward(x):
     x̃ = Q_act(x)        per-tensor 비대칭:  x̃ = (clamp(round(x/Δ)+z, 0, 2^b−1) − z)·Δ
     W̃ = Q_w(W)           출력 채널 o마다 (Δ_o, z_o), 비대칭
     return conv(x̃, W̃)
```

- **activation 범위(Δ, z):** calibration 이미지 256장을 흘려 모은 표본에서 L2.4 오차가 최소인 범위를 고른다(MSE observer, 한 번만 탐색).
- **weight 범위:** 출력 채널마다 [min, max]를 1%씩 줄여 가며 L2.4 오차가 최소인 값을 고른다(80개 후보).
- **첫/마지막 레이어:** stem 첫 conv와 head cv2/cv3 마지막 1×1은 W8A8로 둔다(`set_first_last_bits`, 저비트 표준 관례). 확정 프로토콜에서는 head 마지막 1×1의 입력 activation을 A16으로 올린다(`set_last_abits`, §3.4).
- **범위 밖:** 확정 프로토콜(§3.4)에서 FP로 남는 것은 LayerNorm, softmax, sigmoid, max, residual add, DFL, CLIP(오프라인 텍스트 임베딩), NMS뿐이다. 10-01까지의 결과는 attention Linear/matmul과 head contrastive matmul도 FP였다(`--attn-quant none`).
- 실제 정수 커널이 아니라 "양자화 → 곧바로 역양자화"로 정밀도 손실만 재현한다.

### 3.2 BRECQ (③)
`pipeline/quant/brecq.py::optimize_brecq`, conv 모듈은 `AdaRoundQuantConv2d`(`pipeline/quant/adaround.py`)다.

- **weight rounding 학습:** 원소마다 `W_int = floor(W/Δ) + h(α)`로 둔다. `h(α) = clamp(sigmoid(α)·1.2 − 0.1, 0, 1)`이고, 올림/내림을 연속 변수로 학습한다.
- **재구성 단위:** 블록 출력 오차 `‖f_block(q) − f_block(fp)‖²`를 줄이도록 α와 activation step(LSQ)을 함께 2000 iter 학습한다.
  - 입력 q는 앞 블록들이 이미 양자화된 상태의 실제 입력이다. 그래서 누적 오차가 보상된다.
- **이 저장소의 설정:** batch 1, α와 LSQ 공동 최적화, neck도 블록 단위. 공식 설정과 다른 점은 BASELINE_STATUS에 기록돼 있다.
- **16bit activation은 LSQ에서 뺀다**(`LSQ_MAX_BITS = 16`). observer step을 그대로 쓴다. 이유는 §3.4에 있다.
- **QDrop**은 같은 함수에 activation drop 확률 0.5를 준 것이다. LSQ 규칙도 같다.
- **다른 단계와의 관계:** BRECQ는 ①과 ②가 바꾼 모델을 그대로 받는다.
  - AdaRoundQuantConv2d는 QuantConv2d의 `w_bits`, 이미 계산된 weight scale, `mig` 버퍼를 물려받는다.
  - ①·②와 BRECQ는 코드상 서로 간섭하지 않는다.

### 3.3 attention·contrastive 8bit 양자화 (10-01, `--attn-quant`, 확정 프로토콜은 `attn_cls`)
10-01까지의 프로토콜은 conv만 양자화했다. 이 옵션은 남은 행렬 연산까지 8bit로 내린다. "FP로 둔 것이 결론에 영향을 주는가"를 확인하고, QATMA와 조건을 맞추는 데 쓴다. **10-02부터는 `attn_cls`가 확정 프로토콜이다.** 구현은 `pipeline/quant/attn_quant.py`에 있다.

| 수준 | 양자화하는 연산 |
|---|---|
| `none` (CLI 기본값, 10-01까지의 프로토콜) | 없음 (conv만) |
| `attn` | C2fAttn 게이트 경로(guide projection, 이미지 임베딩, max-sigmoid 게이트 aw), ImagePoolingAttn(key·value·proj Linear, query, softmax 출력) |
| `attn_cls` (**확정 프로토콜**) | `attn` + head contrastive matmul(정규화된 영역 임베딩 × 텍스트) |

**양자화 규칙 (배포 관점)**
- **이미지에서 오는 실행 중 값**(matmul 피연산자, Linear 입력, softmax/sigmoid 출력): per-tensor 비대칭 8bit `ActObserver`를 쓴다. conv와 같은 calibrate()에서 관측하고 freeze한다(`QPoint`).
- **텍스트에서만 오는 상수**(어휘가 고정되면 미리 계산해 두는 값): weight처럼 행(클래스)별 비대칭 MSE 8bit로 양자화한다(`ConstQ`). 데이터가 필요 없고, 입력이 바뀔 때만 다시 계산한다.
- **실행 중 입력을 받는 Linear:** weight는 출력 채널별 MSE 8bit, 입력은 A8로 양자화한다(`QuantLinear`).
- **FP로 남기는 것:** LayerNorm, softmax, sigmoid, max, residual add, logit scale/bias. 모두 원소별 연산이고 통상 LUT나 FP로 처리한다.
- **v1 주의:** ImagePoolingAttn(16)이 이미지로 텍스트를 갱신한다. 그래서 그 뒤의 C2fAttn(19, 22)이 받는 guide는 상수가 아니고 실행 중 값으로 다룬다. v2는 ImagePoolingAttn이 없어 네 곳 모두 상수다. head는 항상 원래 텍스트를 쓴다.
- **게이트 교환(⓪)과의 연동:** 교환된 forward와 원래 forward가 같은 게이트 함수(`attn_gate`)를 쓰므로 양자화 지점도 같다.
- **BRECQ와의 연동:** 실행 중 값의 양자화는 STE로 gradient를 통과시킨다. 그래야 앞 블록 conv의 재구성이 막히지 않는다. 이 지점들의 scale은 학습하지 않는다.

**연산량 (s-World, 640×640)**
- conv 16.29 GMAC, attention 0.12 GMAC이다.
- contrastive는 COCO-80에서 0.34 GMAC, LVIS-1203에서 **5.17 GMAC**이다.
- 따라서 LVIS 어휘에서 기본 프로토콜은 행렬 연산의 약 24%를 FP로 둔다. `attn_cls`를 쓰면 이미지 경로의 행렬 연산이 사실상 전부 정수가 된다.

**결과 (s-World, 6 seed, runs/153; attention FP는 같은 seed의 runs/147)**

짝 Δ는 평균 ± 95% CI(t 분포)이고 단위는 LVIS AP점, LVIS_flip %p다. W4A5 `attn_cls`는 seed 0에서 BRECQ가 붕괴해서(아래), BRECQ가 들어간 비교는 seed 0을 뺀 5 seed 값을 같이 적는다.

| 설정 | 비교 | attention FP | `attn_cls` |
|---|---|---|---|
| W4A8 | GPM − BRECQ (LVIS AP / flip) | +1.09 ± 0.21 / −6.13 ± 0.17 | +1.21 ± 0.11 / −5.67 ± 0.29 |
| W4A8 | PM − BRECQ | +1.36 ± 0.16 / −6.31 ± 0.27 | +1.49 ± 0.12 / −5.78 ± 0.24 |
| W4A8 | GPM − PM | −0.27 ± 0.23 / +0.18 ± 0.13 | −0.28 ± 0.14 / +0.12 ± 0.18 |
| W4A5 | GPM − BRECQ (5 seed) | +2.62 ± 0.42 / −8.63 ± 0.80 | +2.62 ± 0.44 / −9.07 ± 0.49 |
| W4A5 | PM − BRECQ (5 seed) | +2.44 ± 0.22 / −7.33 ± 0.54 | +2.27 ± 0.41 / −7.71 ± 0.54 |
| W4A5 | GPM − PM (6 seed) | +0.25 ± 0.40 / −1.28 ± 0.38 | **+0.39 ± 0.25 (p 0.01)** / −1.36 ± 0.33 |

같은 조건에서 `attn_cls − none` 차이(LVIS AP)는 다음과 같다. 모두 CI가 0을 포함하거나 0.3점 이내다.
- W4A8: BRECQ −0.03, PM +0.10, GPM +0.09
- W4A5 (seed 0 제외): BRECQ +0.02, PM −0.15, GPM +0.02

**정리**
- attention과 contrastive matmul을 8bit로 내려도 절대 성능과 짝 차이가 그대로다. 그러니 FP로 둔 연산이 결론을 만든 것이 아니다. 이 프로토콜(`attn_cls`)을 기본으로 삼을 수 있다.
- **G의 패턴도 같다.** W4A8에서는 PM이 GPM보다 LVIS AP 0.28점 앞선다. W4A5에서는 GPM이 flip을 1.4%p 낮춘다. `attn_cls`에서는 W4A5 LVIS AP 차이(+0.39)도 유의해졌다.
- **BRECQ 붕괴 빈도:** W4A5에서 6 seed 중 1번(seed 0), W4A8에서 0번이다. PM·GPM은 두 설정 모두 0/6이다.

- **W4A5 `attn_cls` seed 0의 BRECQ 붕괴:** COCO AP가 18.74로, 같은 seed의 `attn` BRECQ(29.87)보다 크게 낮다. seed 1은 정상(30.00)이고, 같은 seed의 PM(32.40)과 GPM(32.84)도 정상이다. 진단은 runs/153의 `diag*_w4a5_*_s0.log`에 있다. 빌드 직후 훅(`PTQ_POST_BUILD`)으로 같은 모델을 다시 만들어 쟀고, 붕괴는 18.74로 똑같이 재현된다.

  | 진단 | 방법 | 결과 | 해석 |
  |---|---|---|---|
  | diag1 | 평가 때 양자화 지점을 묶음별로 끔 (cv4 영역 쪽 / cv4 텍스트 쪽 / C2fAttn 게이트 / ImagePoolingAttn) | 18.74 / 18.74 / 18.77 / 18.69 | attention·contrastive 양자화가 평가 시점에 직접 만든 손상이 아니다 |
  | diag2 | conv 70개의 activation 양자화를 전부 끔 | 19.16 | activation 범위 문제(m 모델 사례)가 아니다 |
  | diag3 | conv weight를 FP로 되돌림 (앞 35개 / 뒤 35개) | 0.01 / 4.09 | 더 나빠진다. BRECQ는 블록끼리 서로 보상하도록 학습하므로, 일부만 되돌리는 진단은 쓸 수 없다 |
  | diag4 | 블록별 출력 상대오차(FP 대비, calibration 32장), 정상 모델(`attn`, 같은 seed)과 비교 | 아래 표 | 특정 conv 하나가 폭발하는 형태가 아니다 |

  | 구간 | 붕괴 (`attn_cls`) | 정상 (`attn`) |
  |---|---|---|
  | 블록 0~14 | 거의 같음 (블록 12: 0.83 vs 0.81) | |
  | 블록 15~22 | 0.20 ~ 0.68 | 0.18 ~ 0.64 (블록마다 15~30% 작음) |
  | head cv3 (P3/P4/P5) | 0.34 / 0.67 / 0.59 | 0.28 / 0.58 / 0.48 |
  | head cv4 (P3/P4/P5) | 0.11 / 0.14 / 0.16 | 0.08 / 0.09 / 0.13 |

  - 블록 12의 큰 오차(0.8)는 정상 모델에도 있다. 그러니 12.cv2가 원인이 아니다. 처음에는 12.cv2 재구성 실패로 추정했지만 diag4에서 기각됐다.
  - 차이는 블록 15부터 생겨 뒤로 갈수록 커진다. 블록 15 이후에는 ImagePoolingAttn(16)과, 실행 중 텍스트를 받는 C2fAttn(19·22)이 있다.
  - calibration 이미지에서는 오차가 15~30% 늘어날 뿐인데 평가 AP는 크게 무너진다. calibration 데이터의 재구성 오차로는 드러나지 않는 실패다. m 모델 때 flip·box 오차 지표로 붕괴가 보이지 않았던 것과 같은 양상이다.
  - **추정:** `attn_cls`의 양자화 지점이 calibrate 중 난수(MSE observer 부분추출)를 소비한다. 그래서 이후 activation 범위와 BRECQ의 무작위성이 `attn`과 달라지고, 이 seed에서 블록 15 이후 재구성이 나쁜 해로 갔다. 원인을 conv 하나로 좁히지는 못했다.
  - **보고 원칙:**
    - 이 seed의 GPM − BRECQ(+11점)는 방법의 이득으로 쓰지 않는다. 표에는 6 seed 값과 seed 0을 뺀 값을 같이 싣고, 붕괴 seed를 밝힌다.
    - **6 seed 결과:** BRECQ는 6 seed 중 1번 붕괴했고(W4A5), PM과 GPM은 붕괴하지 않았다. 빈도가 낮아서 "우리 방법이 이 불안정을 막는다"고 주장하기에는 근거가 약하다. "기준선이 seed에 따라 불안정할 수 있다"는 관찰로만 쓴다.

### 3.4 확정 프로토콜과 head 마지막 conv 입력 A16 (10-02)

| 항목 | 설정 | 인자 |
|---|---|---|
| 양자화 범위 | 모든 conv(head 포함) + attention·contrastive matmul 8bit | `--no-skip-head --attn-quant attn_cls` |
| 첫·마지막 레이어 | stem 첫 conv, head cv2/cv3 마지막 1×1은 W8A8 | `--first-last-bits 8` |
| head 마지막 conv 입력 | A16 (weight는 8bit 그대로) | `--last-abits 16` |
| 재현성 | 결정적 실행, 조건별 RNG 복원 | `--deterministic` |

**A16을 넣은 이유 (runs/152)**
- YOLOv8m-World에서 naive W8A8 COCO AP가 seed에 따라 40.38 / 24.30으로 갈렸다.
- 원인은 head `cv3.0` 마지막 1×1의 입력 범위다. per-tensor MSE observer의 무작위 부분추출에 따라 0~115와 0~144로 달라진다.
- 이 conv들의 입력만 A16으로 두면 두 seed 모두 40.7로 고쳐진다. s는 +0.30, v2는 +0.04점으로 손해가 없다.
- **m에만 쓰지 않고 모든 모델에 같은 프로토콜로 적용한다.** 평가 결과를 보고 모델별로 프로토콜을 바꾸면 테스트 세트로 튜닝하는 것이 되기 때문이다.
- 선행 연구의 관례(Reg-PTQ는 마지막 예측층 FP, QATMA는 첫·마지막 레이어 FP)보다 덜 관대한 조건이다. weight는 8bit 그대로이고 입력만 16bit다.
- 모델 크기는 변하지 않는다. 해당 conv는 연산량의 2~4%다.

**A16으로 생긴 버그와 수정 (runs/155)**
- **증상:** 새 프로토콜에서 BRECQ+M 모델의 head cv3 마지막 conv 한 레벨이 FP와 반대 방향의 출력을 냈다(cos −0.15, 상대오차 1.1). 그런데 weight만 양자화하면 오차는 0.5%였다.
- **원인:** BRECQ가 16bit activation의 step까지 LSQ(Adam)로 학습했다. 16bit step(~1e-4)은 Adam의 스텝 크기(≈ 학습률)보다 작거나 비슷하다. 그래서 step이 자기 크기만큼씩 흔들리다 수십 배로 커졌다. 이전(M)은 입력 범위를 줄여 step을 더 작게 만들어서 증상이 컸다.
- **수정:** `brecq.py`의 `LSQ_MAX_BITS = 16`. activation이 16bit 이상인 conv는 LSQ에서 빼고 observer step을 고정한다. 16bit 미만 conv의 동작은 그대로라 10-01까지의 결과에는 영향이 없다.
- 수정 전에 돌린 새 프로토콜 결과는 `runs/154_protocol/old_lsqbug/`에 따로 보관하고, 전부 다시 돌렸다.

---

## 4. 방법의 작동 원리

### 4.0 빌드 순서 (`pipeline/run_comparison.py::build`)

```
 ⓪ gate_commute_all(m.model)                  (+G) C2fAttn cv2를 cv2_main + head별 cv2_side로 분리, 게이트를 뒤로
 wrap_convs(W,A 비트)                         모든 conv → QuantConv2d
 set_first_last_bits(8)                       stem·head 마지막 = 8bit
 set_last_abits(16)                           (--last-abits) head cv2/cv3 마지막 1×1 입력만 A16 (§3.4)
 ① set_conv_wbits(보호 목록, 8)               보호 conv의 weight 비트만 8로   ← calibrate 전이어야 scale이 8bit 기준
 quantize_attention(scope)                    (--attn-quant) attention/contrastive 양자화 지점 삽입 (§3.3)
 model.to(device)
 ② search_and_apply_tied(...)                 채널 계보 추적 → 공유 s 탐색 → W←W·diag(s), x←x/s
 calibrate(...)                               activation 범위·weight scale 확정 (이미 이전된 값 기준)
 ③ convert_to_adaround + optimize_brecq       rounding·LSQ 재구성 (+G면 FP 기준 모델도 같은 구조로 변환해 넘김)
                                              16bit 이상 activation은 LSQ 제외(LSQ_MAX_BITS)
 (+A) 위 과정을 M 끔/켬으로 두 번 → mcheck_flip → 하나를 고름 (§4.5)
```

- 보호 목록은 `main()`에서 한 번 정해 모든 조건에 똑같이 넘긴다(`--protect-budget`).
- 짝비교에서 이 옵션들은 run 안의 모든 조건(brecq, combined 등)에 동일하게 적용된다.

### 4.1 ① 보호: 어느 conv를 8bit로 둘 것인가

**진단** (`pipeline/diag_w4_sensitivity.py::rank_convs_leakfree`)
1. calibration 이미지(train2017 앞 200장)에서 FP 모델의 region 임베딩과 COCO 점수를 저장한다.
2. naive W8A8 모델에서 **conv 하나만** weight를 4bit로 내리고, 나머지는 8bit로 둔 채 다시 흘린다. 기준 모델은 실험과 같은 프로토콜을 쓴다(확정 프로토콜이면 `attn_cls` + A16).
3. **기준값 = COCO flip 증가량:** FP가 확신(sigmoid > 0.25)하는 anchor에서 COCO top-1 클래스가 바뀐 비율이 얼마나 늘었는지.
4. conv 63개 전부에 대해 반복한다. 첫/마지막 8bit conv 7개는 제외한다.

**선택** (`select_protected`)
- 증가량이 큰 conv부터 담는다. 담은 파라미터 합이 예산 B의 110%를 넘지 않게 하고, 90% 이상 차면 멈춘다.
- **B = 양자화 weight 파라미터의 1.5%.** YOLOv8s-World에서는 0.187M 중 0.172M을 쓰고, 4개가 뽑힌다.

```
 12.cv2  (C2fAttn concat → 1×1, 0.131M)   ← 텍스트 융합 지점
 1       (stem 직후 3×3, 0.018M)
 2.cv2   (첫 C2f concat → 1×1, 0.006M)
 4.cv1   (C2f 입력 1×1, 0.016M)
```

**누수 방지**
- 평가 이미지(val2017)를 쓰지 않는다. **LVIS 어휘는 로드조차 하지 않는다.**
- 진단은 전역 RNG를 저장·복원한다. 진단 유무가 뒤따르는 BRECQ의 무작위성을 바꾸지 않는다.
- 결과는 `configs/protect_cache/`에 모델·프로토콜별로 캐시한다. 확정 프로토콜 캐시는 `<모델>_n200_c256_img640_la16_aqattn_cls.json`이다.

**모델별 보호 목록 (확정 프로토콜과 이전 프로토콜에서 같다)**

| 모델 | 보호 목록 (예산 1.5%) | `+G`일 때 남는 것 |
|---|---|---|
| YOLOv8s-World | 12.cv2, 1, 2.cv2, 4.cv1 | 1, 2.cv2, 4.cv1 (+0.3%) |
| YOLOv8s-WorldV2 | 12.cv2, 15.cv2, 1 | 1 |
| YOLOv8m-World | 12.cv2, 15.cv2 | 없음 (GP = G) |

- 세 모델 모두 텍스트 융합 지점(C2fAttn cv2)이 가장 먼저 뽑힌다.
- 프로토콜이 바뀌어도 목록이 같다는 것은 진단이 안정적이라는 근거다.

**왜 이 conv들인가 (작동 원리)**
- `12.cv2` 입력에서 y3의 outlier 채널이 출력 채널별 weight 범위 [min, max]를 지배한다. 그래서 4bit(16단계) 격자가 너무 성겨진다.
- **두 가지 손상이 겹친다.**
  - **(a)** attention 분기 자체의 반올림 오차가 큰 activation(최대 20)과 곱해져 증폭된다.
  - **(b)** 같은 scale을 공유하는 다른 세 분기의 작은 weight들이 대부분 0 근처로 뭉개진다.
- 측정값 (FP 입력, 출력 상대 오차):

  | 설정 | 출력 상대 오차 |
  |---|---|
  | 12.cv2 W4 | 0.83 (출력이 사실상 파괴) |
  | attention 분기 열만 W8 | 거의 회복 |
  | 모델 전체, 12.cv2 하나 W4 | LVIS flip 6.5% → 79% |
  | 모델 전체, 12.cv2 W4 + attention 분기만 W8 | 6.9% |

- 이 손상은 임베딩을 통째로 뒤튼다. 그래서 점수 차가 큰 확실한 판정(margin > 2)도 70% 넘게 뒤집힌다.
- LVIS는 비슷한 클래스가 많아(점수 차 < 0.5인 anchor가 COCO의 약 18배) 같은 손상에도 더 크게 흔들린다.
- `1`, `2.cv2`, `4.cv1`은 텍스트와 무관한 초반 backbone의 concat/split 경계다. BRECQ 이후에도 남는 손상의 상당 부분이 여기서 나온다.
  - fusion 지점 네 곳만 W8로 두면 상한이 LVIS AP +0.65점이다.
  - 초반 conv까지 넣으면 +1.25점이다.

**8bit 보호의 구현과 배포**
- 해당 `QuantConv2d.w_bits = 8`로 두면, weight scale 탐색과 AdaRound 격자가 8bit(256단계)로 잡힌다.
- 배포에서는 **레이어별 정밀도**(이 conv만 INT8 weight)다. 대부분의 엣지 툴체인이 지원한다.

### 4.2 ② 공유 제약 scale 이전: activation의 부담을 weight로 옮기기

**기본 아이디어** (SmoothQuant/AWQ 계열)
conv의 입력 채널 c마다 양수 s_c를 두면 FP 함수는 변하지 않는다.

```
 y = conv(x, W) = conv( x / s , W · diag(s) )
 양자화:  ŷ = conv( Q_act(x/s) , Q_w(W · diag(s)) )
```

- x의 채널별 크기 편차를 s로 나눠 평평하게 만들면 per-tensor activation 격자가 덜 낭비된다.
- 그 대가로 weight 열이 s배 커진다.
- s_c = max|x_c|^α / max|W_{:,c}|^(1−α)이고, α가 둘 사이의 배분을 정한다.

**배포 제약: 왜 "공유"가 필요한가**
- 배포에서 x/s는 소비 conv가 아니라 **생산 conv가 만든다.** 생산 conv의 `conv → SiLU → requant` 단계에서, 정수로 반올림하기 직전에 채널별 1/s를 곱한다.
- 그러므로 두 제약이 생긴다.
  - (1) **같은 생산 채널을 받는 모든 소비 conv는 같은 s를 써야 한다.** 예: C2f의 y1은 m.0.cv1과 cv2가 함께 받는다.
  - (2) **residual add(a + b)로 합쳐지는 생산 채널들은 같은 s여야 한다.** 그래야 덧셈 뒤에도 하나의 s가 유지된다.
- concat, split, upsample, maxpool은 양수 채널 배율과 교환 가능해서 s가 그대로 통과한다.
- attention처럼 x를 conv 없이 FP로 쓰는 곳은 그 앞에서 x·s로 역양자화한다.

**채널 계보 추적** (`pipeline/quant/channel_graph.py::trace_channel_producers`)
어느 소비 입력 채널이 어느 생산 채널에서 왔는지를 모델 구조를 하드코딩하지 않고 **측정으로** 찾는다.

```
 for 생산 conv P (ultralytics Conv 모듈 61개):
     run1: P 출력을 채널별 r1_j = 1.5 + j/C 로 나눔
     run2: P 출력을 2 로 나눔
     (두 실행 모두 모든 QuantConv2d 출력을 기준 실행 값으로 고정 → 교란이 conv를 넘어 퍼지지 않음)
     각 소비 conv 입력 채널 c에서  d_k = x0 − x_k  (k=1,2)
       d1/d2 = 2(1 − 1/r1_j)  → j를 역산   (add로 a+b가 섞여도 a와 무관하게 성립)
       원소별 d1/d2가 일정하지 않으면(비선형 경로로 새는 의존) 계보에서 제외
```

- 결과: 소비 입력 채널 15,648개는 생산 채널 1개에, 864개는 residual add로 2~3개에 의존한다. stem 입력 3개는 계보가 없다.
- union-find로 묶으면 생산 채널 10,592개가 9,920개 그룹이 된다. 가장 큰 그룹은 3채널이다.

**s 탐색** (`search_and_apply_tied`)
- **단위:** residual add로 묶인 생산자들의 연결요소.
- **절차:**
  1. 단위마다 α ∈ {0, 0.25, 0.5, 0.75, 1} 각각으로 그룹별 s를 만든다(그룹 안 모든 소비 채널의 max|x|, max|W| 사용).
  2. 영향받는 모든 소비 conv의 상대 출력 오차 합을 계산한다. 오차는 `‖conv(Q_act(x/s), Q_w(W·s)) − conv(x, W)‖² / ‖conv(x, W)‖²`이고, calibration 이미지 8장으로 잰다.
  3. 가장 작은 α를 채택한다. **"이전 없음(s=1)"도 후보**라서 어떤 단위도 나빠지지 않는다.
- **적용:** 소비 conv마다 `QuantConv2d.apply_migration(s)`를 호출한다. `W ← W·diag(s)`를 실제로 곱해 두고, forward에서 `x ← x/s`로 나눈다.
  - 이후 calibration과 BRECQ는 이전된 값 위에서 그대로 동작한다.
- 보호(①)가 먼저 적용되어 있으므로, 보호된 conv의 오차는 8bit 기준으로 평가된다.

**검증**
- FP 등가성: TF32를 끄면 이전 전후의 상대 오차가 약 3e-6이다.
- 배포형(생산자가 x/s를 내보내고 소비자는 W·s만 사용) 등가성: 상대 오차 5.5e-6.

**왜 W8A8에서 특히 잘 듣는가**
- W8A8의 남은 손상은 weight가 아니라 activation 쪽이다. 예전 진단에서도 cv3 activation만 A11로 올리면 held-out flip이 −41%였다(`docs/PROMPTCAL_V3_MIXED_PRECISION.md`).
- ②는 activation의 채널 편차를 여유가 있는 8bit weight로 옮긴다. 그래서 비트를 늘리지 않고 같은 효과의 상당 부분을 얻는다(이전 프로토콜에서 LVIS_flip −30~33%).
- **확정 프로토콜에서는 ② 단독의 효과가 작아졌다.** LVIS AP +0.20, flip 3.83 → 3.47%(−9%)다. attention·contrastive 8bit와 A16이 W8A8에 남아 있던 activation 손상의 분포를 바꾼 것으로 보인다(원인은 미확인).
- **⓪을 더하면 W8A8에서도 크게 개선된다.** GPM(실제로 ⓪+②)은 flip 3.83 → 2.91%(−24%), LVIS_lost 820 → 581이다.
- W4A8에서는 weight에도 여유가 없어 이득이 작다. ①이 큰 손상을 막은 뒤에 소폭 더한다(LVIS AP +0.23~0.25점).

### 4.3 두 단계의 역할 분담 (비트별)

| | 주 병목 | 주로 기여하는 단계 | 근거 (run) |
|---|---|---|---|
| W8A8 | activation (채널 편차, cv3 경로, 텍스트 융합 지점) | ② 이전 + ⓪ 게이트 교환 (확정 프로토콜에서는 ⓪이 더 큼) | 135, 147, 158, 159 |
| W4A8 | weight (텍스트 융합 conv, 초반 backbone) | ① 보호 | 132 vs 134, 141, 147 |
| W4A6·W4A5 | weight + activation (텍스트 융합 conv가 이전의 부담까지 받음) | ⓪ 게이트 교환 + ① + ② | 147 (G-*) |

**이전의 부작용과 그 해소:** A6·A5에서 이전(②) 단독은 BRECQ보다 나쁘다(W4A6 0/6).
- 이전은 취약성을 없애지 않고 옮긴다. `runs/148`에서 12.cv2의 민감도는 줄고, 1.conv의 민감도는 3.5~4배 는다.
- 보호(①)나 게이트 교환(⓪)이 취약한 conv를 받쳐 줄 때만 이전이 이득이 된다. W4A6 G → GM은 LVIS AP +0.31점, W4A5는 +0.65점이다(이전 프로토콜).
- **확정 프로토콜 표 2(W4A5, 6 seed):** M 단독은 −0.75점(비유의)이고, G 위에서는 +0.59, GP 위에서는 +0.60점이다. 같은 메커니즘이 확정 프로토콜에서도 확인된다.
- **모델에 따라 이전 자체가 해로울 수 있다.** YOLOv8m-World에서는 ①·⓪이 있어도 ②를 더하면 BRECQ보다 나빠진다(PM −1.5 ~ −3.5점). 그래서 ②는 사전 검사(§4.5)로 켤지 정한다.

### 4.4 ⓪ 텍스트 게이트 교환 (09-30)

**무엇을 바꾸나.** C2fAttn(§2.2)의 마지막 1×1 conv `cv2`는 네 분기 concat을 받는다. 그중 텍스트 attention 분기는 `a_h = p_h ⊙ aw_h`다.
- `p = attn.proj_conv(y2)`
- `aw_h`: head h의 위치별 텍스트 게이트, `sigmoid(max_j⟨embed, guide_j⟩/√d + b)·scale`
- 1×1 conv는 선형이고 게이트는 위치별 스칼라라서, 다음이 FP로 정확히 같다.

```
 cv2(cat(y0, y1, y2, a)) = W_main · cat(y0, y1, y2) + b + Σ_h aw_h ⊙ (W_h · p_h)
```

- **구현 (`pipeline/quant/fusion_quant.py::gate_commute`):**
  - `cv2.conv`의 weight 열을 앞 세 분기(`cv2_main`, bias 포함)와 head별 텍스트 열(`cv2_side[h]`, bias 없음)로 쪼갠다.
  - forward를 교체한다: 게이트 aw는 원래 식 그대로 계산하고, p는 게이트 없이 `cv2_side`에 넣은 뒤 그 출력에 aw를 곱해 더한다. 마지막에 SiLU를 적용한다.
  - YOLOv8s-World는 C2fAttn 4곳(12, 15, 19, 22), head 4개다.
- **양자화에서 달라지는 점:**
  1. 텍스트 분기가 자기 전용 conv를 가진다. 다른 세 분기와 **weight scale(출력 채널별)과 activation scale을 공유하지 않는다.** 09-29 분석의 원인 "공유 scale 오염"을 구조적으로 없앤다.
  2. 양자화되는 입력이 게이트가 곱해지기 **전**의 p다. 게이트가 만드는 위치별 크기 편차(0~scale)가 양자화 범위에서 빠진다.
  3. 게이트가 작은 위치(텍스트와 무관한 영역)에서는 텍스트 분기의 양자화 오차도 게이트 배율만큼 줄어든다.
- **효과 (naive, 12번 블록만 W4, 이전 프로토콜):** LVIS flip이 68.8%에서 **9.7%**로 줄었다. 분기별 scale 분리만 하면 18.5%다. 게이트를 뒤로 옮긴 것 자체가 핵심이다.
- **확정 프로토콜에서의 게이트:** 게이트 aw도 8bit로 양자화된다(`attn_gate`의 `QPoint`). 그래도 G의 패턴은 그대로다(W4A5 GPM − PM LVIS_flip −1.4%p, 6 seed, §3.3).
- **ablation (확정 프로토콜 표 2, W4A5, 6 seed):** G 단독이 +1.37점, flip −5.4%p로 단일 요소 중 가장 크다. PM 위에 더해도 +0.39점, flip −1.33%p다.
- **다른 모델:** m에서는 G 하나로 W4A5 LVIS AP +2.3~3.2점이다(2 seed). P(+1.5~2.4)보다 낫다.
- **BRECQ와의 연동:**
  - neck/head는 conv 단위로 FP conv 출력을 목표로 재구성하므로, FP 기준 모델도 같은 구조로 변환해 넘긴다(FP 등가라 목표는 같다).
  - 새 conv(`cv2_main`, `cv2_side`)는 일반 conv처럼 AdaRound/LSQ를 받는다.
- **이전(②)과의 연동:**
  - 채널 계보 추적은 새 conv를 그대로 소비자로 인식한다.
  - C2fAttn 출력은 사용자 정의 forward에서 나와 계보가 끊긴다. 그래서 그 출력을 받는 채널에는 이전을 적용하지 않는다(s=1, 안전).
- **보호(①)와의 연동:** 게이트 교환이 C2fAttn cv2를 대신 다루므로, 보호 목록에서 해당 conv를 뺀다(`+G` 규칙). 남는 보호 대상은 초반 backbone 3개이고, 크기는 +0.3%다.
- **배포:** 1×1 conv 두 개(main, side), 위치별 곱, 덧셈, SiLU. 모두 표준 연산이고 FP 연산량은 같다. 추가 비트는 없다. 실제 엔진에서 fusion되는지는 미확인이다.

### 4.5 이전(M)의 채택 사전 검사 (10-02, `+A`)

**왜 필요한가.** 이전(M)은 s와 v2에서는 저비트 성능을 올리지만, **YOLOv8m-World에서는 크게 해친다.** 확정 프로토콜, 2 seed 기준 LVIS AP는 PM −1.53 ~ −3.53, GPM −0.53 ~ −3.75점이다. 같은 seed의 P, G는 개선된다. 평가 결과를 보고 모델별로 M을 빼면 테스트 세트 튜닝이 되므로, 누수 없는 규칙으로 정한다.

**검사** (`diag_w4_sensitivity.py::mcheck_flip`)
- **이미지:** train2017 정렬 순서에서 calibration 256장 **다음** 200장. calibration에 쓴 이미지는 BRECQ가 이미 맞춘 이미지라 M의 해가 드러나지 않을 수 있어서 쓰지 않는다.
- **어휘:** COCO만. val2017과 LVIS는 보지 않는다.
- **지표:** FP가 확신하는(sigmoid > 0.25) anchor에서 top-1 클래스가 바뀐 비율(COCO flip).
- **규칙:** M을 켠 모델의 flip이 끈 모델의 **1.5배**를 넘으면 M을 끈다(`MCHECK_RATIO`).

**검증 (BRECQ 2000 iter 후 검사. `+A` 행의 s만 seed 1, 나머지는 seed 0)**

| 경우 | M 끔 | M 켬 | 결정 | 실제 평가 |
|---|---|---|---|---|
| m W4A8 (P vs PM) | 3.19% | 6.74% | 끔 | P가 낫다 ✅ |
| m W4A5 (P vs PM) | 9.54% | 44.46% | 끔 | P가 낫다 ✅ |
| m W4A5 (GP vs GPM, `+A`) | 6.24% | 43.46% | 끔 | GP가 낫다 ✅ |
| s W4A5 (P vs PM) | 7.30% | 5.55% | 켬 | PM이 낫다 ✅ |
| s W4A5 (GP vs GPM, `+A`) | 6.19% | 6.04% | 켬 | GPM 채택, 단독 실행과 bit-identical |
| s W4A8 (P vs PM) | 2.17% | 2.29% | 켬 | PM이 약간 낫다 ✅ |
| s W8A8 (BRECQ vs M) | 0.20% | 0.21% | 켬 | M이 낫다 ✅ |
| v2 W8A8 (BRECQ vs M) | 0.37% | 0.37% | 켬 | AP +0.5~0.7, **LVIS_flip +3~4%p** ✗ (한계) |

- 문턱을 1.1~2배 사이 어디에 두어도 결정이 같다. m은 2.1~7배이고 나머지는 1.06배 이하라 간격이 크다.
- **임베딩·logit 오차는 기준이 될 수 없다.** M은 m에서도 이 오차들을 줄인다(0.34 → 0.24). 평균 오차는 줄이면서 판정을 뒤집는 손상이라, 판정을 직접 보는 기준이 필요하다.
- **한계:** v2 W8A8처럼 LVIS(held-out) 판정에서만 나빠지는 경우는 COCO 어휘 검사로 잡히지 않는다.

**비용과 운영 (runs/156, 157)**
- 검사 자체는 22~24초다.
- `+A`는 BRECQ를 두 번 돌려서 빌드 시간이 약 2.2~2.4배가 된다(m W4A5 3232초, s W4A5 2684초).
- 짧은 BRECQ(200 iter)로 미리 검사하는 방법도 시험했다. 후보당 6~10분이 들어 절약이 크지 않았고, m W4A5에서 결정이 틀렸다(두 후보 모두 flip 약 50%).
- **운영 방식: 모델당 한 번만 검사**하고(W4A8 또는 W4A5), 그 결정을 모델의 모든 설정에 쓴다. M이 해로운지는 비트 설정이 아니라 모델에 따라 갈렸다(m은 두 설정 모두 끔, s·v2는 모두 켬).
- 결정: s·v2는 M 켬, m은 M 끔.

---

## 5. 실행 방법

실행 절차, 파일 구성, 대기열 작업자는 [`pipeline/README.md`](pipeline/README.md)에 정리했다. 확정 프로토콜의 공통 인자는 `pipeline/scripts/protocol.sh`다.

```bash
cd /home/taeho/promptcal-ptq
source pipeline/scripts/protocol.sh      # PROTOCOL 변수 + CUDA_DEVICE_ORDER=PCI_BUS_ID
.venv/bin/python pipeline/run_comparison.py $PROTOCOL --model yolov8s-world.pt \
    --w-bits 4 --a-bits 5 --conditions naive,brecq,brecq+PM,brecq+GPM --seed 0 --device 4
# m: 이전은 검사에서 꺼지므로 최종 구성은 brecq+GP
.venv/bin/python pipeline/run_comparison.py $PROTOCOL --model yolov8m-world.pt \
    --w-bits 4 --a-bits 5 --conditions naive,brecq --seed 0 --device 5
```

| 조건 접미사 | 의미 |
|---|---|
| `+P` | 보호 (누수 없는 진단, `--protect-criterion`, 예산 `--protect-budget` 또는 기본 1.5%) |
| `+M` | 공유 제약 scale 이전 (`--mig-alphas`가 없으면 0,0.25,0.5,0.75,1) |
| `+G` | 텍스트 게이트 교환. P/R/H와 함께 쓰면 C2fAttn cv2는 보호 목록에서 빠짐 |
| `+R` / `+H` | 같은 예산의 무작위 보호 / HAWQ식(출력 MSE) 보호 (대조군) |
| `+A` | 이전(M)을 사전 검사로 켤지 정함(§4.5). M 끔/켬 두 번 빌드 |

| 옵션 | 의미 | 기본 |
|---|---|---|
| `--protect-budget` | 보호 예산(양자화 weight 파라미터 비율). 0이면 끔 | 0 |
| `--protect-criterion` | 선택 기준(`coco` 확정; `emb`, `cls`, `box`는 ablation) | coco |
| `--protect-images` / `--protect-cache` | 진단 이미지 수 / 캐시 위치 | 200 / `configs/protect_cache` |
| `--mig-alphas` / `--mig-tied` / `--mig-images` | 이전 α 후보 / 공유 제약 버전 / 탐색 이미지 수 | 끔 / 끔 / 8 |
| `--hi-wbit-convs`, `--hi-wbit-blocks` | 보호 목록 수동 지정(진단 재현용) | 끔 |
| `--attn-quant` | `none` / `attn` / `attn_cls` (§3.3). 확정 프로토콜은 `attn_cls` | none |
| `--last-abits` | head cv2/cv3 마지막 1×1 입력 비트(§3.4). 확정 프로토콜은 16 | 0 (끔) |
| `--post-build` | 빌드 직후 진단 스크립트를 실행하고 평가 없이 종료 | 끔 |

모든 옵션은 기본값이 꺼짐이고, 꺼져 있으면 기존 코드와 bit-identical이다(검증함).

**코드 위치**

| 파일 | 역할 |
|---|---|
| `pipeline/run_comparison.py` | 빌드 순서(§4.0), CLI |
| `pipeline/diag_w4_sensitivity.py` | ① 진단(`rank_convs_leakfree`), 선택(`select_protected`), M 사전 검사(`mcheck_flip`) |
| `pipeline/quant/quant_model.py` | `set_conv_wbits`, `set_block_wbits`, `set_first_last_bits`, `set_last_abits` |
| `pipeline/quant/channel_graph.py` | ② 채널 계보 추적, 공유 그룹, s 탐색, 배포형 등가성 검사 |
| `pipeline/quant/migrate.py` | ② 독립 s 버전(상한, ablation), 채널 단위 W8(ablation) |
| `pipeline/quant/fake_quant.py` | `QuantConv2d` (`mig`, `hi_cols`, 비트별 scale) |
| `pipeline/quant/fusion_quant.py` | ⓪ 게이트 교환(`gate_commute`), 분기 인식 concat 양자화 프로토타입(기각) |
| `pipeline/quant/attn_quant.py` | attention·contrastive 8bit 양자화(`quantize_attention`, `attn_gate`) |
| `pipeline/quant/adaround.py`, `brecq.py` | ③ BRECQ (`LSQ_MAX_BITS`) |
| `pipeline/scripts/protocol.sh`, `worker.sh` | 확정 프로토콜 인자, GPU별 대기열 작업자 |

`src/quant/`는 `pipeline/quant/`의 사본으로 동기화해 둔다.

**진단 훅:** `--post-build <스크립트>`(또는 환경변수 `PTQ_POST_BUILD`)를 주면 빌드 직후, 평가 전에 그 스크립트를 실행하고 끝난다. `models`, `fp`, `args`, `device`를 쓸 수 있다. 주 실험 경로에는 영향이 없다.

---

## 6. 배포 형태와 비용 (엣지 관점)

| 구성요소 | 배포 형태 | 추가 비용 | 확인 상태 |
|---|---|---|---|
| ① 보호 | 해당 conv만 INT8 weight (레이어별 정밀도) | 크기 +1.3% (W4A8), 연산 형태 동일 | 시뮬레이션만 |
| ② 이전 | 생산 conv의 SiLU 뒤 채널별 곱셈(1/s), 소비 conv는 W·s를 표준 커널로 | 채널별 곱셈 1회. conv에 fusion되면 거의 0 | 배포형 FP 등가성은 확인. 실제 엔진 fusion은 미확인 |
| ⓪ 게이트 교환 | C2fAttn cv2를 1×1 conv 두 개 + 위치별 곱 + 덧셈으로 | 없음 (FP 연산량 동일) | FP 등가성 확인. 엔진 fusion 미확인 |
| ③ BRECQ | 학습된 rounding이 반영된 정적 weight | 없음 | – |
| attention·contrastive 8bit (`attn_cls`) | INT8 GEMM. 텍스트 상수는 클래스별 INT8로 미리 저장 | 텍스트 상수 저장 약 1 MB (LVIS) | 시뮬레이션만 |
| head 마지막 conv 입력 A16 | 해당 6개 1×1 conv만 INT16 activation × INT8 weight | 연산량의 2~4%가 16bit. 모델 크기 변화 없음 | 시뮬레이션만 |
| M 사전 검사 (`+A`) | PTQ 시점에만 쓰이고 배포 모델에는 없다 | 모델당 1회. 검사 22~24초 + 빌드 1회 추가 | 검증 완료 |

- **PTQ 제작 비용:** 서버에서 1회 수행한다. GPU 메모리 약 15GB(s), 조건 하나에 빌드 17~22분이다(RTX 4000 Ada). 평가까지 넣으면 약 30분이다. 기기는 완성된 모델만 받는다.
- **추론 모델 크기 (s, 확정 프로토콜, 10-03 수정):** W4 BRECQ 6.58 MiB, GPM 6.60 MiB(+0.3%), PM 6.66 MiB(+1.2%), W8A8 12.52 MiB. FP32는 약 49 MiB다.
  - 8bit attention Linear weight(0.46M, conv의 3.6%, +0.44 MiB)를 포함한 값이다. 기준선과 우리 방법에 똑같이 더해진다. 그 이전 로그의 크기는 conv weight만 센 값이다.
  - m은 BRECQ와 GPM 모두 14.10 MiB, v2는 실행 중 Linear가 없어 conv만의 값과 같다.
- **FP로 남는 연산:** 확정 프로토콜(`attn_cls`)에서는 원소별 연산(LayerNorm, softmax, sigmoid, add)과 DFL뿐이고, head 마지막 conv 입력은 16bit다. 10-01까지의 결과는 attention의 Linear/matmul과 contrastive matmul도 FP였다. 어휘가 고정되면 텍스트 임베딩은 미리 계산해 두므로 기기에서 CLIP을 돌릴 필요는 없다.
- **W4 conv의 실제 가속:** 하드웨어 지원이 제한적이다. 모든 W4 방법에 공통인 조건이다. 목표 기기 검증이 필요하다(결과 문서 §12 P7).

---

## 7. 설계 근거: 기각된 대안들

| 대안 | 무엇을 했나 | 결과 | 교훈 |
|---|---|---|---|
| **PromptCal / Combined** (이전 방법) | BRECQ 뒤 conv별 activation scale(s_mult)을 COCO top-k margin 손실로 학습 | W8A8 6-seed: LVIS AP +0.41점이지만 Heval_flip·lost 0/6 악화. W4A8: BRECQ보다 악화(0/3). **새 양자화기 위에 얹어도** 이득 없음, held-out 지표 2/2 악화(runs/140) | calibration 어휘에 맞춘 학습은 held-out 순위를 해친다 |
| **vocabulary-metric 재구성** (brecq_vm) | BRECQ 재구성 손실을 "임의 어휘에서의 유사도 오차" 2차형식으로 교체 | W4A8: 모든 변형(lam 0/1, identity, mix, uniform)에서 LVIS AP −0.1~−0.65점 | 손실 설계로는 저비트 손상의 위치를 못 바꾼다 |
| **rounding 공동 학습** (claim21·22) | s_mult와 α(또는 α bias)를 margin 손실로 함께 학습 | W4A8 붕괴 | sparse한 순위 손실이 weight를 움직이면 안 된다 |
| **채널 단위 W8** (outlier 열만 8bit) | conv마다 \|W\|·\|x\| 상위 k% 입력 채널만 W8 | LVIS AP 약 0.234에서 포화. 같은 크기의 conv 단위 보호보다 약간 못함 | 초반 backbone의 퍼진 손상을 못 잡는다 |
| **fusion 지점만 보호** | C2fAttn cv2 네 곳 W8 | 상한 +0.65점 (크기 +6.4%) | 텍스트 융합만으로는 부족하고 초반 backbone도 필요하다 |
| **독립 s 이전** | conv마다 s를 따로 | COCO +0.8~1.0, 하지만 배포 불가 | 공유 제약 버전(②)으로 대체 |
| **새로운 선택 기준** | 임베딩 방향 오차 등 "순위 전용" 기준 | 출력 MSE와 순위 상관 0.99. 데이터 기반 기준은 모두 같은 conv를 고른다. 확정 프로토콜 표 3a에서도 P − H +0.15점(비유의) | 기준은 기여가 아니다. 가장 단순한 COCO flip으로 고정 |
| **W4A4** | 소수 conv만 A8 | conv의 76%를 A8로 되돌려도 붕괴. A5부터 풀림 | per-tensor A4는 이 모델의 절벽. 한계로 명시 |
| **분기 인식 concat 양자화** (09-30) | concat 입력을 생산자별 구간으로 나눠 구간마다 activation·weight scale | A4 붕괴 해결 실패(LVIS flip 95% 이상). 효과는 weight 쪽(그룹 양자화)뿐 | A4 손상은 concat을 넘어 전체에 퍼져 있다. 기존 기법과 같아 기각 |
| **이전 고려 보호 배분** (A안, 09-30) | 이전 적용 뒤의 모델로 보호 대상 진단 | 보호 목록의 마지막 한 자리만 바뀜 | 기대 이득이 작아 보류. 대신 "이전은 취약성을 옮긴다"는 메커니즘을 얻음 |
| **모델별로 프로토콜·구성을 수동으로 바꾸기** (10-01~02) | m에만 A16을 쓰거나, m에서만 M을 빼기 | 평가 결과(val, LVIS)를 보고 정하는 셈이라 기각 | A16은 모든 모델에 적용하고, M은 누수 없는 검사로 정한다 |
| **마지막 conv에 이전(M)으로 A8 유지** (10-01) | A16 대신 head 마지막 conv에 이전을 적용 | 국소 기준은 이전을 고르지 않고, 방향 오차 기준은 AP를 약 14로 무너뜨림 | 범위 불안정은 비트로만 풀린다 |
| **임베딩·logit 오차 기준의 M 검사** (10-02) | M 켬/끔을 평균 임베딩 오차로 비교 | m에서도 M이 오차를 줄여(0.34 → 0.24) 해를 못 잡음 | 판정(flip) 기준이어야 한다 |
| **짧은 BRECQ(200 iter) M 검사** (10-02) | 짧게 재구성한 뒤 검사하고 본 BRECQ는 한 번만 | 후보당 6~10분(절약이 작음). m W4A5에서 결정이 틀림 | 모델당 한 번 전체 BRECQ로 검사하는 방식으로 대체 |
| **calibration 이미지로 M 검사** (10-02) | BRECQ calibration 256장에서 flip 비교 | 시험하지 않고 설계 단계에서 배제 | BRECQ가 이미 맞춘 이미지라 해가 가려질 수 있다. calibration 밖 이미지를 쓴다 |

**정리:** 순위는 **학습(손실)으로 맞추는 게 아니라, 순위가 무너지는 구조적 지점을 양자화기 수준에서 지켜서** 보존한다.

---

## 8. 무엇을 주장할 수 있고, 무엇은 아닌가

**주장할 수 있는 것**
- open-vocab 검출기의 저비트 손상은 held-out 어휘 순위에서 가장 크게 드러난다(LVIS가 COCO보다 2~3배 민감).
- 손상은 텍스트 attention이 합쳐지는 concat conv와 초반 backbone에 집중된다. 원인은 attention 분기의 outlier와 공유 scale이다. s-v1, s-v2, m-v1에서 재현된다.
- 누수 없는 진단으로 그 지점을 찾아 지키고(①), 게이트 교환(⓪)과 공유 제약 이전(②)을 더하면 W4에서 LVIS 판정 flip이 약 30% 줄고 LVIS AP가 +0.8~2.5점 오른다(GPM, 확정 프로토콜 6 seed). 비용은 W4 모든 설정에서 +0.3%다(PM 변형은 +1.3%). 같은 크기의 무작위 보호는 효과가 거의 없다(+0.21점, 표 3a).
- 세 요소가 모두 기여한다(표 2). 게이트 교환이 단일 요소 중 가장 크고, 이전은 받쳐 주는 요소가 있을 때만 이득이다.
- QDrop 위에서도 같은 방식으로 개선되고(표 3b, GPM W4A5 +6.4점), 결과가 BRECQ + GPM과 같아진다. GPM을 쓰면 기반 PTQ의 선택이 중요하지 않다.
- calibration 어휘 기반 학습(PromptCal)은 held-out 순위를 해친다(음성 결과, W4A8·W4A6 6-seed에서 0/6).
- **텍스트 게이트 교환은 open-vocab 검출기의 텍스트 게이팅 구조를 이용한 양자화 전용 재배치다.** FP 등가이고 비트 비용이 0이며, 저비트(A6·A5)에서 BRECQ 대비 6/6 개선한다. 보호와 결합(GPM)하면 PM보다 작은 모델로 held-out 판정을 6/6 더 잘 지킨다.
- 학습 없는 PTQ로 W4A5까지 동작한다. 기존 PTQ(QDrop, AdaRound, PromptCal)는 같은 조건에서 크게 무너진다.
- (6 seed, W4A8·W4A5) attention과 contrastive matmul까지 8bit로 양자화해도 GPM의 이득과 G의 패턴이 유지된다. 즉 FP로 둔 연산이 결론을 만든 것이 아니다.
- (확정 프로토콜, 2 seed) **세 YOLO-World 모델(s, v2, m) 모두 BRECQ보다 좋아진다.** 손상 위치 진단(P)과 게이트 교환(G)은 크기·버전에 걸쳐 일반화된다. 특히 m에서는 G 하나로 W4A5 +2.3~3.2점이다.
- 이전(M)이 해로운 모델(m)은 calibration 밖 COCO flip 검사로 미리 걸러진다. 평균 임베딩 오차가 아니라 판정 기준이어야 걸러진다.

**주장하면 안 되는 것**
- "순위 보존을 위해 새로 설계한 선택 기준/손실": 선택 기준은 출력 MSE와 같은 결과를 내고, 이전은 범용 기법이다. 새로움은 게이트 교환(구조)과 분석에 있다.
- "GPM이 모든 비트에서 PM보다 낫다": W4A8에서는 PM 변형이 크기를 1% 더 쓰고 LVIS AP로 약 0.3점 앞선다. 방법은 GPM으로 고정하고, 이 트레이드오프를 그대로 보고한다.
- "W8A8에서 AP가 크게 오른다": W8A8의 LVIS AP 개선은 +0.31점으로 작다. 판정 충실도(flip −24%, lost −29%) 개선이 주된 효과라고 쓴다.
- "6 seed 모두에서 BRECQ를 이긴다"(W4A5): seed 0에서는 기준선이 붕괴해서 차이가 부풀려진다. 붕괴 seed를 밝히고 seed 0 제외 값을 함께 쓴다.
- "W4A4를 PTQ로 열었다": 열지 못했다.
- "held-out 어휘에만 특화된 개선": COCO도 비슷한 비율로 좋아진다.
- "실제 엣지 기기에서의 가속": 아직 시뮬레이션뿐이다.
- "이전(M)은 항상 도움이 된다": m에서는 해롭다. 검사로 끄는 규칙이 방법의 일부다.
- "M 검사가 모든 해를 잡는다": v2 W8A8의 LVIS 전용 악화는 잡지 못한다.

---

## 9. 남은 검증 (10-03 기준)

**끝난 것**
- ~~W4A8 게이트 교환 6 seed~~: W4A8은 PM, 저비트는 GPM.
- ~~attention·contrastive 8bit 6 seed~~ (§3.3): 결론 유지. s W4A5 seed 0에서 BRECQ 붕괴(1/6).
- ~~m 불안정 원인과 A16~~ (§3.4).
- ~~확정 프로토콜 2 seed 점검~~ (runs/154, LSQ 버그 수정 후 재실행):
  - v2: W4A8 PM +6.54 / +6.39, GPM +5.71 / +5.95. W4A5 PM +7.32 / +7.86, GPM +5.66 / +6.13. LVIS AP는 PM이, LVIS_flip은 GPM이 낫다.
  - m: P +1.00 / +1.74(A8), +2.35 / +1.49(A5). G = GP +0.77 / +1.50(A8), **+3.17 / +2.26(A5)**. PM·GPM은 BRECQ보다 나쁘다(M이 원인).
  - v2 W8A8 M: LVIS AP +0.71 / +0.47, LVIS_flip 5.0 → 8.1 / 8.7%. 새 프로토콜에서도 같다.
- ~~M 채택 규칙~~ (§4.5): 모델당 검사 1회.
- ~~표 1 (s, 확정 프로토콜, 6 seed)~~ (runs/158, 10-03): §1과 결과 문서 §1.

**남은 것**
1. ~~표 2·3~~ (runs/158, 10-03): §1과 결과 문서 §2·§3.
2. ~~모델 크기 계산~~ (10-03): attention Linear weight를 포함하도록 고쳤다(§6).
3. **v2·m 3 seed 확장:** 최종 구성(v2 PM/GPM, m GP)만.
4. **YOLOE-v8s:** 구조가 다른 OVOD로의 일반화. G를 적용할 수 없으므로 P(+M 검사)만 본다. 코드 지원이 먼저 필요하다.
5. **YOLO-World-L / QATMA 조건(첫·마지막 FP):** L40S 필요.
6. **SmoothQuant/AWQ 정식 구현, Reg-PTQ와 비교.**
7. **목표 엣지 기기에서 실제 배포 검증:** 게이트 교환 fusion 포함.
8. **s W4A5 seed 0의 기준선 붕괴:** 확정 프로토콜에서도 같은 seed에서 BRECQ(COCO 19.3)와 QDrop(9.8)이 함께 무너진다. 원인 conv는 특정하지 못했다(§3.3 진단). 붕괴 seed를 밝혀 보고한다.
9. **W8A8에서 M 단독 효과가 줄어든 원인:** 확정 프로토콜의 어느 변경(attention 8bit / A16)이 영향을 주는지 확인하면 좋다(선택). GPM의 효과는 크다.
10. **나중에:** 이전 고려 보호 배분(A안), 임베딩 방향 편향 보정(방향 3), TF32 영향 확인.
