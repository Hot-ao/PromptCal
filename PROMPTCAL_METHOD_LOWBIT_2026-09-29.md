# 저비트 PTQ 방법 — 구조와 작동 원리 (2026-09-29, 09-30 갱신)

**대상 독자:** 이 방법을 이어서 구현·실험하거나 논문을 쓰는 공저자.

**목적:** "무엇을 하는 방법인가, 모델의 어디를 어떻게 바꾸는가, 왜 그게 통하는가"를 한 문서에서 설명한다.

**관련 문서**
- 실험 수치와 경위는 [`docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md`](docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md)(6-seed 확정, 게이트 교환)와 [`docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-29.md`](docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-29.md)(원인 분석)에 있다(run 번호로 추적).
- **09-30 변경:** 텍스트 게이트 교환(⓪, §4.4)을 방법에 추가했다. 저비트(A6·A5)에서는 GPM이 최종 구성이다.
- 이전 설계(Combined = BRECQ + s_mult margin 학습)는 [`PROMPTCAL_PAPER_DESIGN_2026-09-28.md`](PROMPTCAL_PAPER_DESIGN_2026-09-28.md)에 있다. 이 문서의 방법은 그 설계를 대체한다. 이유는 §7에 있다.

---

## 1. 한눈에 보기

**문제.** open-vocabulary 검출기(YOLO-World)를 저비트로 양자화하면 AP보다 먼저 **held-out 어휘의 판정 순위**가 무너진다. held-out 어휘는 calibration 때 보지 않은 LVIS 1203클래스 같은 것이다.
- W8A8 BRECQ에서도 LVIS 판정의 약 4%가 FP와 다른 클래스로 바뀐다.
- W4A8에서는 18%가 바뀐다.

**관찰.** 이 손상은 모델 전체에 고르게 퍼져 있지 않다. 두 곳에 몰려 있다.
1. **텍스트 융합 지점:** neck의 `C2fAttn` 블록에서 텍스트 가이드 attention 출력이 concat으로 합쳐진 직후의 1×1 conv. 특히 `12.cv2`다.
2. **초반 backbone:** stem 직후 conv(`1`)와 첫 C2f(`2`, `4`)의 concat 경계.

**방법.** 같은 절차를 모든 비트 설정에 적용한다.

```
 ⓪ 게이트 교환(Gate commute) : C2fAttn의 텍스트 게이트를 1×1 conv 뒤로 옮기는 FP 등가 재배치 (비트 비용 0)   [09-30 추가]
 ① 보호(Protect)   : 누수 없는 진단으로 고른 소수 conv의 weight를 8bit로 유지 (예산 = weight 파라미터의 1.5%)
                     ⓪을 쓰면 C2fAttn cv2(12.cv2 등)는 목록에서 빠진다 -> 실제 보호는 1, 2.cv2, 4.cv1 (+0.3%)
 ② 이전(Migrate)   : 입력 채널별 scale s를 생산 conv → 소비 conv weight로 옮김 (배포 제약을 지키는 공유 s)
 ③ 재구성(BRECQ)   : 표준 블록 재구성 PTQ (rounding + activation step 학습). QDrop으로 바꿔도 된다
```

- 이름: ①+② = **PM**, ⓪+①+② = **GPM** (조건 접미사, §5).
- W8A8에서는 모든 weight가 이미 8bit라 ①이 규칙상 아무것도 하지 않는다(비용 0). ②가 주로 기여한다.
- W4A8에서는 ①이 주로 기여한다. PM이 GPM보다 낫다(6 seed: GPM의 LVIS AP 0/6).
- **W4A6·W4A5에서는 GPM이 최선이다:** PM보다 작은 모델로 held-out 판정 지표가 6/6 더 좋다.

**결과 요약 (head 포함, 6 seed, BRECQ → 우리)**

| 설정 | 방법 | 크기 | COCO AP | LVIS AP | LVIS_flip |
|---|---|---|---|---|---|
| W8A8 | ② + ③ | +0% | 36.26 → 36.80 | 0.2533 → 0.2565 | 3.91 → 2.63% |
| W4A8 | PM | +1.3% | 33.19 → 34.80 | 0.2239 → 0.2375 | 18.30 → 11.99% |
| W4A6 | **GPM** | **+0.3%** | 32.11 → 34.12 | 0.2147 → 0.2306 | 21.31 → 14.66% |
| W4A5 | **GPM** | **+0.3%** | 29.90 → 32.84 | 0.1958 → 0.2228 | 28.47 → 19.69% |

- FP32는 COCO AP 36.80, LVIS AP 0.2589다.
- 모든 행이 6/6 seed 개선이다.
- 같은 크기에서 무작위로 conv를 골라 보호하면 효과가 없다.
- QDrop 위에서도 같은 크기로 개선된다.

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
       cv4[l] : ContrastiveHead           → 클래스 점수 = τ · cos(x, t_j) + b   (conv 없음)

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
- **첫/마지막 레이어:** stem 첫 conv와 head cv2/cv3 마지막 1×1은 8bit로 둔다(`set_first_last_bits`, 저비트 표준 관례).
- **범위 밖:** attention 안의 Linear/matmul, DFL, CLIP, conv 출력(다음 conv 입력에서 양자화됨)은 FP다.
- 실제 정수 커널이 아니라 "양자화 → 곧바로 역양자화"로 정밀도 손실만 재현한다.

### 3.2 BRECQ (③)
`pipeline/quant/brecq.py::optimize_brecq`, conv 모듈은 `AdaRoundQuantConv2d`(`pipeline/quant/adaround.py`)다.

- **weight rounding 학습:** 원소마다 `W_int = floor(W/Δ) + h(α)`로 둔다. `h(α) = clamp(sigmoid(α)·1.2 − 0.1, 0, 1)`이고, 올림/내림을 연속 변수로 학습한다.
- **재구성 단위:** 블록 출력 오차 `‖f_block(q) − f_block(fp)‖²`를 줄이도록 α와 activation step(LSQ)을 함께 2000 iter 학습한다.
  - 입력 q는 앞 블록들이 이미 양자화된 상태의 실제 입력이다. 그래서 누적 오차가 보상된다.
- **이 저장소의 설정:** batch 1, α와 LSQ 공동 최적화, neck도 블록 단위. 공식 설정과 다른 점은 BASELINE_STATUS에 기록돼 있다.
- **다른 단계와의 관계:** BRECQ는 ①과 ②가 바꾼 모델을 그대로 받는다.
  - AdaRoundQuantConv2d는 QuantConv2d의 `w_bits`, 이미 계산된 weight scale, `mig` 버퍼를 물려받는다.
  - ①·②와 BRECQ는 코드상 서로 간섭하지 않는다.

---

## 4. 방법의 작동 원리

### 4.0 빌드 순서 (`pipeline/run_comparison.py::build`)

```
 ⓪ gate_commute_all(m.model)                  (+G) C2fAttn cv2를 cv2_main + head별 cv2_side로 분리, 게이트를 뒤로
 wrap_convs(W,A 비트)                         모든 conv → QuantConv2d
 set_first_last_bits(8)                       stem·head 마지막 = 8bit
 ① set_conv_wbits(보호 목록, 8)               보호 conv의 weight 비트만 8로   ← calibrate 전이어야 scale이 8bit 기준
 model.to(device)
 ② search_and_apply_tied(...)                 채널 계보 추적 → 공유 s 탐색 → W←W·diag(s), x←x/s
 calibrate(...)                               activation 범위·weight scale 확정 (이미 이전된 값 기준)
 ③ convert_to_adaround + optimize_brecq       rounding·LSQ 재구성 (+G면 FP 기준 모델도 같은 구조로 변환해 넘김)
```

- 보호 목록은 `main()`에서 한 번 정해 모든 조건에 똑같이 넘긴다(`--protect-budget`).
- 짝비교에서 이 옵션들은 run 안의 모든 조건(brecq, combined 등)에 동일하게 적용된다.

### 4.1 ① 보호: 어느 conv를 8bit로 둘 것인가

**진단** (`pipeline/diag_w4_sensitivity.py::rank_convs_leakfree`)
1. calibration 이미지(train2017 앞 200장)에서 FP 모델의 region 임베딩과 COCO 점수를 저장한다.
2. naive W8A8 모델에서 **conv 하나만** weight를 4bit로 내리고, 나머지는 8bit로 둔 채 다시 흘린다.
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
- 결과는 `configs/protect_cache/`에 모델별로 캐시한다.

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
- ②는 activation의 채널 편차를 여유가 있는 8bit weight로 옮긴다. 그래서 비트를 늘리지 않고 같은 효과의 상당 부분을 얻는다(LVIS_flip −30~33%).
- W4A8에서는 weight에도 여유가 없어 이득이 작다. ①이 큰 손상을 막은 뒤에 소폭 더한다(LVIS AP +0.23~0.25점).

### 4.3 두 단계의 역할 분담 (비트별)

| | 주 병목 | 주로 기여하는 단계 | 근거 (run) |
|---|---|---|---|
| W8A8 | activation (채널 편차, cv3 경로) | ② 이전 | 135, 147 |
| W4A8 | weight (텍스트 융합 conv, 초반 backbone) | ① 보호 | 132 vs 134, 141, 147 |
| W4A6·W4A5 | weight + activation (텍스트 융합 conv가 이전의 부담까지 받음) | ⓪ 게이트 교환 + ① + ② | 147 (G-*) |

**이전의 부작용과 그 해소:** A6·A5에서 이전(②) 단독은 BRECQ보다 나쁘다(W4A6 0/6).
- 이전은 취약성을 없애지 않고 옮긴다. `runs/148`에서 12.cv2의 민감도는 줄고, 1.conv의 민감도는 3.5~4배 는다.
- 보호(①)나 게이트 교환(⓪)이 취약한 conv를 받쳐 줄 때만 이전이 이득이 된다. W4A6 G → GM은 LVIS AP +0.31점, W4A5는 +0.65점이다.

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
- **효과 (naive, 12번 블록만 W4):** LVIS flip이 68.8%에서 **9.7%**로 줄었다. 분기별 scale 분리만 하면 18.5%다. 게이트를 뒤로 옮긴 것 자체가 핵심이다.
- **BRECQ와의 연동:**
  - neck/head는 conv 단위로 FP conv 출력을 목표로 재구성하므로, FP 기준 모델도 같은 구조로 변환해 넘긴다(FP 등가라 목표는 같다).
  - 새 conv(`cv2_main`, `cv2_side`)는 일반 conv처럼 AdaRound/LSQ를 받는다.
- **이전(②)과의 연동:**
  - 채널 계보 추적은 새 conv를 그대로 소비자로 인식한다.
  - C2fAttn 출력은 사용자 정의 forward에서 나와 계보가 끊긴다. 그래서 그 출력을 받는 채널에는 이전을 적용하지 않는다(s=1, 안전).
- **보호(①)와의 연동:** 게이트 교환이 C2fAttn cv2를 대신 다루므로, 보호 목록에서 해당 conv를 뺀다(`+G` 규칙). 남는 보호 대상은 초반 backbone 3개이고, 크기는 +0.3%다.
- **배포:** 1×1 conv 두 개(main, side), 위치별 곱, 덧셈, SiLU. 모두 표준 연산이고 FP 연산량은 같다. 추가 비트는 없다. 실제 엔진에서 fusion되는지는 미확인이다.

---

## 5. 실행 방법

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # 필수: --device N = nvidia-smi N
COMMON="--model yolov8s-world.pt --deterministic --calib 256 --no-skip-head --conditions brecq \
        --protect-budget 0.015 --mig-alphas 0,0.25,0.5,0.75,1 --mig-tied"

# W4A8 (보호 + 이전 + BRECQ)
.venv/bin/python pipeline/run_comparison.py $COMMON --w-bits 4 --a-bits 8 --first-last-bits 8 --seed 0 --device 4
# W8A8 (보호는 자동 no-op, 이전 + BRECQ)
.venv/bin/python pipeline/run_comparison.py $COMMON --w-bits 8 --a-bits 8 --seed 0 --device 4

# 조건 접미사로 한 run에서 짝비교 (09-30 권장 방식; 조건마다 RNG를 복원하므로 순서 무관)
.venv/bin/python pipeline/run_comparison.py --model yolov8s-world.pt --deterministic --calib 256 --no-skip-head \
    --first-last-bits 8 --w-bits 4 --a-bits 6 --conditions naive,brecq,brecq+PM,brecq+GPM,qdrop,qdrop+PM --seed 0 --device 4
```

| 조건 접미사 | 의미 |
|---|---|
| `+P` | 보호 (누수 없는 진단, `--protect-criterion`, 예산 `--protect-budget` 또는 기본 1.5%) |
| `+M` | 공유 제약 scale 이전 (`--mig-alphas`가 없으면 0,0.25,0.5,0.75,1) |
| `+G` | 텍스트 게이트 교환. P/R/H와 함께 쓰면 C2fAttn cv2는 보호 목록에서 빠짐 |
| `+R` / `+H` | 같은 예산의 무작위 보호 / HAWQ식(출력 MSE) 보호 (대조군) |

| 옵션 | 의미 | 기본 |
|---|---|---|
| `--protect-budget` | 보호 예산(양자화 weight 파라미터 비율). 0이면 끔 | 0 |
| `--protect-criterion` | 선택 기준(`coco` 확정; `emb`, `cls`, `box`는 ablation) | coco |
| `--protect-images` / `--protect-cache` | 진단 이미지 수 / 캐시 위치 | 200 / `configs/protect_cache` |
| `--mig-alphas` / `--mig-tied` / `--mig-images` | 이전 α 후보 / 공유 제약 버전 / 탐색 이미지 수 | 끔 / 끔 / 8 |
| `--hi-wbit-convs`, `--hi-wbit-blocks` | 보호 목록 수동 지정(진단 재현용) | 끔 |

모든 옵션은 기본값이 꺼짐이고, 꺼져 있으면 기존 코드와 bit-identical이다(검증함).

**코드 위치**

| 파일 | 역할 |
|---|---|
| `pipeline/run_comparison.py` | 빌드 순서(§4.0), CLI |
| `pipeline/diag_w4_sensitivity.py` | ① 진단(`rank_convs_leakfree`), 선택(`select_protected`), 분석용 진단 CLI |
| `pipeline/quant/quant_model.py` | `set_conv_wbits`, `set_block_wbits`, `set_first_last_bits` |
| `pipeline/quant/channel_graph.py` | ② 채널 계보 추적, 공유 그룹, s 탐색, 배포형 등가성 검사 |
| `pipeline/quant/migrate.py` | ② 독립 s 버전(상한, ablation), 채널 단위 W8(ablation) |
| `pipeline/quant/fake_quant.py` | `QuantConv2d` (`mig`, `hi_cols`, 비트별 scale) |
| `pipeline/quant/fusion_quant.py` | ⓪ 게이트 교환(`gate_commute`), 분기 인식 concat 양자화 프로토타입(기각) |
| `pipeline/quant/adaround.py`, `brecq.py` | ③ BRECQ |

`src/quant/`는 `pipeline/quant/`의 사본으로 동기화해 둔다.

---

## 6. 배포 형태와 비용 (엣지 관점)

| 구성요소 | 배포 형태 | 추가 비용 | 확인 상태 |
|---|---|---|---|
| ① 보호 | 해당 conv만 INT8 weight (레이어별 정밀도) | 크기 +1.3% (W4A8), 연산 형태 동일 | 시뮬레이션만 |
| ② 이전 | 생산 conv의 SiLU 뒤 채널별 곱셈(1/s), 소비 conv는 W·s를 표준 커널로 | 채널별 곱셈 1회. conv에 fusion되면 거의 0 | 배포형 FP 등가성은 확인. 실제 엔진 fusion은 미확인 |
| ⓪ 게이트 교환 | C2fAttn cv2를 1×1 conv 두 개 + 위치별 곱 + 덧셈으로 | 없음 (FP 연산량 동일) | FP 등가성 확인. 엔진 fusion 미확인 |
| ③ BRECQ | 학습된 rounding이 반영된 정적 weight | 없음 | – |

- **PTQ 제작 비용:** 서버에서 1회 수행한다. GPU 메모리 약 15GB, run당 약 20분이다. 기기는 완성된 모델만 받는다.
- **추론 모델 크기 (s):** W4 PM 6.22 MiB, W4 GPM 6.16 MiB, W8A8 12.08 MiB. FP32는 약 49 MiB다.
- **FP로 남는 연산:** attention의 Linear/matmul, DFL. 어휘가 고정되면 텍스트 임베딩은 미리 계산해 두므로 기기에서 CLIP을 돌릴 필요는 없다.
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
| **새로운 선택 기준** | 임베딩 방향 오차 등 "순위 전용" 기준 | 출력 MSE와 순위 상관 0.99. 데이터 기반 기준은 모두 같은 conv를 고른다 | 기준은 기여가 아니다. 가장 단순한 COCO flip으로 고정 |
| **W4A4** | 소수 conv만 A8 | conv의 76%를 A8로 되돌려도 붕괴. A5부터 풀림 | per-tensor A4는 이 모델의 절벽. 한계로 명시 |
| **분기 인식 concat 양자화** (09-30) | concat 입력을 생산자별 구간으로 나눠 구간마다 activation·weight scale | A4 붕괴 해결 실패(LVIS flip 95% 이상). 효과는 weight 쪽(그룹 양자화)뿐 | A4 손상은 concat을 넘어 전체에 퍼져 있다. 기존 기법과 같아 기각 |
| **이전 고려 보호 배분** (A안, 09-30) | 이전 적용 뒤의 모델로 보호 대상 진단 | 보호 목록의 마지막 한 자리만 바뀜 | 기대 이득이 작아 보류. 대신 "이전은 취약성을 옮긴다"는 메커니즘을 얻음 |

**정리:** 순위는 **학습(손실)으로 맞추는 게 아니라, 순위가 무너지는 구조적 지점을 양자화기 수준에서 지켜서** 보존한다.

---

## 8. 무엇을 주장할 수 있고, 무엇은 아닌가

**주장할 수 있는 것**
- open-vocab 검출기의 저비트 손상은 held-out 어휘 순위에서 가장 크게 드러난다(LVIS가 COCO보다 2~3배 민감).
- 손상은 텍스트 attention이 합쳐지는 concat conv와 초반 backbone에 집중된다. 원인은 attention 분기의 outlier와 공유 scale이다. s-v1, s-v2, m-v1에서 재현된다.
- 누수 없는 진단으로 그 지점을 찾아 지키고 공유 제약 scale 이전을 더하면 순위 지표가 30% 이상 개선된다. 비용은 W8A8 0%, W4A8 +1.3%다. 같은 크기의 무작위 보호는 효과가 없다.
- calibration 어휘 기반 학습(PromptCal)은 held-out 순위를 해친다(음성 결과, W4A8·W4A6 6-seed에서 0/6).
- **텍스트 게이트 교환은 open-vocab 검출기의 텍스트 게이팅 구조를 이용한 양자화 전용 재배치다.** FP 등가이고 비트 비용이 0이며, 저비트(A6·A5)에서 BRECQ 대비 6/6 개선한다. 보호와 결합(GPM)하면 PM보다 작은 모델로 held-out 판정을 6/6 더 잘 지킨다.
- 학습 없는 PTQ로 W4A5까지 동작한다. 기존 PTQ(QDrop, AdaRound, PromptCal)는 같은 조건에서 크게 무너진다.

**주장하면 안 되는 것**
- "순위 보존을 위해 새로 설계한 선택 기준/손실": 선택 기준은 출력 MSE와 같은 결과를 내고, 이전은 범용 기법이다. 새로움은 게이트 교환(구조)과 분석에 있다.
- "GPM이 모든 비트에서 PM보다 낫다": W4A8에서는 PM이 낫다(6 seed).
- "W4A4를 PTQ로 열었다": 열지 못했다.
- "held-out 어휘에만 특화된 개선": COCO도 비슷한 비율로 좋아진다.
- "실제 엣지 기기에서의 가속": 아직 시뮬레이션뿐이다.

---

## 9. 남은 검증 (상세는 결과 문서 09-30 §9)
1. ~~W4A8 게이트 교환 6 seed~~ 완료: W4A8은 PM, 저비트는 GPM.
2. **일반화:** YOLOv8m-World, YOLOv8s-WorldV2에서 BRECQ vs PM vs GPM.
3. **W4A8에서 G + 12.cv2 보호 조합.**
4. **QATMA와 같은 조건 비교:** attention 8bit, 첫/마지막 FP, YOLO-World-L.
5. **SmoothQuant/AWQ 정식 구현, Reg-PTQ와 비교.**
6. **목표 엣지 기기에서 실제 배포 검증:** 게이트 교환 fusion 포함.
7. **TF32 conv가 FP 기준과 flip 지표에 주는 영향 확인.**
8. **나중에:** 이전 고려 보호 배분(A안), 임베딩 방향 편향 보정(방향 3).
