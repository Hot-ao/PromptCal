# Baseline(AdaRound/BRECQ/QDrop) 구현 대 공식 구조 상세 비교 (2026-09-23)

이 문서는 Combined(제안 방법)의 성능과 무관하게, **AdaRound/QDrop/BRECQ 세 baseline이
각 원 논문·공식 코드의 구조를 얼마나 충실히 재현했는지**만 정리한다. 09-21~09-23
세션에서 진행한 baseline 정합성 작업(weight/activation calibration, LSQ 독립 구현,
COCO 프로토콜 이식 등)의 최종 상태를 항목별로 기록한다.

참고 소스:
- AdaRound: Nagel et al., *Up or Down? Adaptive Rounding for Post-Training
  Quantization*, ICML 2020 (arXiv 2004.10568). 공식 코드 공개 안 됨 — AIMET
  (`quic/aimet`) 문서로 하이퍼파라미터 대조.
- BRECQ: Li et al., *BRECQ: Pushing the Limit of Post-Training Quantization
  by Block Reconstruction*, ICLR 2021. 공식 코드 `yhhhli/BRECQ` (GitHub).
- QDrop: Wei et al., *QDrop: Randomly Dropping Quantization for Extremely
  Low-bit Post-Training Quantization*, ICLR 2022. 공식 코드 `wimh966/QDrop`
  (GitHub), 논문 본문 + 부록 E(ImageNet/COCO 실험 설정).

우리 구현 파일: `src/quant/adaround.py`, `src/quant/brecq.py`,
`pipeline/run_comparison.py` (`pipeline/quant/`에 동기화된 사본 존재).

---

## 0. 세 방법이 공유하는 것

| 항목 | 공식 | 우리 구현 | 상태 |
|---|---|---|---|
| 양자화 대상 | 논문마다 다름(분류 모델 전체) | YOLO-World의 vision 경로 Conv2d 전부(DFL 제외), text encoder는 원래 conv가 아니라 자동 제외 | 구조상 불가피한 이식(아키텍처가 다름) |
| weight 8bit 양자화 그리드 | 방법별로 다름(아래 참고) | `AdaRoundQuantConv2d`/`QuantConv2d`가 공통 프레임 사용, `w_quant_mode`로 방법별 분기 | 아래 표 참고 |
| activation 8bit 양자화 | 방법별로 다름 | `ActObserver`(`src/quant/fake_quant.py`), `method="minmax"/"mse"` | 아래 표 참고 |
| bias correction, CLE 등 전처리 | 일부 논문이 비교 대상으로만 언급(AdaRound Table 8) | 없음(적용 안 함) | 의도적 범위 제외 — 우리 방법의 대상이 아님 |

---

## 1. AdaRound (Nagel et al. ICML'20)

### 1.1 재구성 알고리즘

| 항목 | 논문/AIMET 공식값 | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 반올림 파라미터화 | rectified sigmoid: `h(α) = clip(σ(α)(ζ-γ)+γ, 0, 1)`, γ=-0.1, ζ=1.1 | 동일(`GAMMA,ZETA = -0.1,1.1`, `h_alpha()`) | ✅ 일치 | `adaround.py:73` |
| 재구성 단위 | layer-wise(conv/linear 각각) | layer-wise(conv마다) | ✅ 일치 | `optimize_adaround()` |
| 목적함수 | 비대칭 재구성 MSE + activation 함수 포함(Table 4) + round 정규화 | `lp_rec_loss`(비대칭, ReLU 포함된 입력으로 캡처) + `reg_loss` | ✅ 일치 | `adaround.py:77-90`, claim13에서 공식 `lp_loss` 정규화(channel-sum+mean) 대조 완료 |
| 정규화 항 | `1-\|2h(α)-1\|^β`, β는 20→2 선형 감소, warmup 20%는 β=20 고정 | 동일(`reg_loss`, `temp_decay`) | ✅ 일치 | `adaround.py:93-101` |
| optimizer | Adam, 기본 하이퍼파라미터(논문 "Adam optimizer with default hyper-parameters") | Adam, `lr=1e-3`(PyTorch Adam 기본값과 동일) | ✅ 일치 | `DEFAULT_LR=1e-3` |
| reg_weight | 논문 미세조정, AIMET 기본 `default_reg_param=0.01` | `DEFAULT_REG_WEIGHT=0.01` | ✅ 일치 | AIMET 문서 대조(claim13) |
| warmup | AIMET `default_warm_start=0.2` | `DEFAULT_WARMUP=0.2` | ✅ 일치 | |
| iters / 이미지 수 | 논문 주요 표: 2048장·20k iters(4bit), AIMET 기본 `default_num_iterations=10000`; Fig.4: 256장으로도 FP32 대비 2% 이내 | calib=256장, `--recon-iters-ada` 기본 **1000** | ⚠️ **iters 미검증** | calib 256은 논문 Fig.4 근거 있음. iters=1000은 BRECQ에서 확인한 수렴 진단(2000 vs 20000 차이 없음, `runs/87_brecq_iters_check`)을 그대로 가져온 것 — **AdaRound 자체로 수렴 확인한 적 없음** |
| batch size | 32(공식/AIMET) | `batch=1`(`optimize_adaround`의 `batch` 인자 기본값) | ⚠️ 다름 | 미검증 — 이 차이 자체의 결과 영향 확인 안 함 |

### 1.2 Weight quantization grid

| 항목 | 논문 | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 대칭/비대칭 | **대칭**(symmetric), zero_point 없음 | 대칭(`w_asym=False` when `w_quant_mode="adaround"`) | ✅ 일치 | `adaround.py:154` |
| granularity | 본문: **레이어 전체 스칼라 하나**. Table 7 각주: 일부 결과는 "더 유리한 per-channel" 사용 | **채널별**(`mse_weight_scale_symmetric_channelwise`) | ⚠️ 본문과 다름, 각주 허용 범위 | 09-21 최초엔 본문 그대로(레이어 전체) 구현 → 이 모델은 채널 간 weight 최대값이 최대 12배 차이 나서 정식 스케일에서 COCO_AP -1.62, Heval_flip 5.3%→21.5% 손상 실측(`runs/89_weight_mse`) → Table 7 각주 근거로 채널별로 전환 |
| scale 결정 방법 | `s`를 `‖W-W̄‖²_F` 최소화(=MSE)로 결정, 사전에(재구성 전) 1회 고정 | `mse_weight_scale_symmetric_channelwise`: 채널별로 80개 후보 범위를 grid search, L_p(2.4) 오차 최소화 | ✅ 취지 일치(공식 BRECQ와 동일한 grid search 방식을 재사용, 채널별 L_p 탐색이라는 점만 원논문이 명시한 정확한 알고리즘과 다를 수 있음) | `fake_quant.py: mse_weight_scale_symmetric_channelwise` |
| weight bit-width | 4bit(논문 주실험), 8bit(활성화 비교용) | 8bit(우리 설정 전체) | 설정 차이, 문제 아님 | W8은 논문도 "activation quant해도 거의 안 무너짐" 확인(Table 7 하단) |

### 1.3 Activation quantization

| 항목 | 논문 | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 학습 여부 | **없음**(순수 weight-rounding 방법. LSQ 없음) | 없음(`adaround_learn_act_scale` 기본 **False**) | ✅ 일치 | claim12 |
| calibration 방식(8bit 비교 실험) | "관측된 min/max로 scale 설정" | **min-max**(`args.torch-seed`와 무관하게 AdaRound 모드는 `calibrate(... act_observer="minmax")`로 강제 재보정) | ✅ 일치 | `run_comparison.py` build() adaround 분기 — 다른 방법(BRECQ/QDrop/naive/Combined)은 mse가 기본이지만 AdaRound만 예외 처리 |

### 1.4 종합 판정
핵심 알고리즘(반올림 파라미터화, loss, 정규화, optimizer 설정)은 **정확히 일치**.
weight granularity(채널별 vs 논문 본문의 레이어 전체)는 논문이 각주로 허용한 변형이며
실측 근거로 문서화됨. **iters=1000·batch=1은 미검증 상태로 남아 있음** — 공식(10k~20k
iters, batch 32) 대비 훨씬 작은 예산이고, 이게 결과에 실질적 영향을 주는지 AdaRound
자체로는 확인한 적이 없다(BRECQ 진단을 빌려 쓴 가정).

---

## 2. BRECQ (Li et al. ICLR'21)

### 2.1 재구성 알고리즘

| 항목 | 공식(`yhhhli/BRECQ`) | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 재구성 단위 | **block-wise**(ResNet의 BasicBlock/Bottleneck 등), 첫/마지막 레이어만 layer-wise | block-wise(YOLO의 C2f/C2fAttn 등 top-level 블록), head는 layer-wise(원래도), neck도 **09-23 현재 block-wise로 되돌림** | ⚠️ neck 처리가 QDrop이 인용한 BRECQ 방식과 다름(아래 3장 참고) | `brecq.py: optimize_brecq()` |
| 반올림 | AdaRound와 동일한 `AdaRoundQuantizer`(h(α)) | 동일(`AdaRoundQuantConv2d` 공용) | ✅ 일치 | |
| loss(weight 단계) | `lp_loss(pred,target,p=2,reduction='none')`: 채널축 sum, 나머지 mean | `lp_rec_loss(p=2)` 동일 정규화 | ✅ 일치(claim13에서 공식 소스 대조 완료) | `adaround.py: lp_rec_loss` |
| optimizer(weight 단계) | `Adam(opt_params)` 기본 lr | `DEFAULT_LR=1e-3` | ✅ 일치 | |
| reg_weight | `--weight` 기본 **0.01** | `DEFAULT_REG_WEIGHT=0.01` | ✅ 일치 | |
| warmup / b_range | `warmup=0.2`(run script 관례), `b_range=(20,2)` | 동일 | ✅ 일치 | |
| **학습 순서** | **순차 2단계**: (1) `act_quant=False`로 alpha만, (2) weight hard 고정 후 activation(LSQ) `iters_a=5000`, `lr=4e-4`, cosine, L_p(2.4) 별도 학습 | **공동 최적화**(`brecq_two_stage` 기본 **False**, 09-23 되돌림) | ❌ **공식과 다름(의도적으로 되돌림)** | `_brecq_act_stage()`로 2단계 구현은 돼 있고 `--brecq-two-stage`로 켤 수 있으나 기본 꺼짐. 사유: 정식 스케일 1-seed 테스트(`runs/85_brecq_two_stage`)에서 act_iters=5000 추가 비용이 6-seed 병렬 실행을 감당 못 할 만큼 늘림(09-22 밤) |
| iters(weight 단계) | ImageNet 표: **20000** | 2000(`recon_iters_strong` 기본값) | ⚠️ 공식보다 작음, 근거 있음 | 1-seed 정식 스케일 수렴 진단(`runs/87_brecq_iters_check`): iters 2000 vs 20000 결과 차이 노이즈 수준(COCO_AP 동일, lost 172 vs 185) — **BRECQ 자체로 검증됨** |
| calibration 이미지 수 | ImageNet 1024장(BRECQ 자체 COCO 절 없음) | 256장 | 아래 3장 참고(QDrop 인용 절차 기준으로는 일치) | |
| batch size(재구성 스텝) | 32(ImageNet) | `brecq_batch` 기본 **1**(09-23 되돌림) | ⚠️ 공식보다 작음 | 1-seed 실측(`runs/86_batch2_skiphead`)으로 batch 1↔2 차이가 노이즈 수준임을 확인했으나, 6-seed 병렬 실행 시간 비용 때문에 1로 되돌림(09-23) |

### 2.2 Weight quantization grid

| 항목 | 공식 | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 대칭/비대칭 | **비대칭**(zero_point 있음) | 비대칭(`w_asym=True` when `w_quant_mode="brecq"`) | ✅ 일치 | `adaround.py:154-159` |
| granularity | **채널별**(공식 실행 스크립트가 `--channel_wise` 사용, README 예시 명령 포함) | 채널별 | ✅ 일치 | `mse_weight_scale_asym_channelwise` |
| scale 결정 | `scale_method='mse'`: `x_min,x_max`를 (1-0.01i)배로 줄여가며 L_p(2.4) 오차 최소화, 채널별로 독립 탐색 | 동일 알고리즘을 벡터화(80개 후보, p=2.4) | ✅ 일치 | `fake_quant.py: mse_weight_scale_asym_channelwise`, 공식 `UniformAffineQuantizer.init_quantization_scale` 소스 직접 대조 |
| 재구성 중 delta 고정 여부 | alpha 학습 동안 delta(scale)는 고정, alpha만 학습 | 동일(calibrate() 때 1회 계산 후 재사용, `qconv.w_scale`을 `AdaRoundQuantConv2d`가 그대로 재사용) | ✅ 일치 | `adaround.py:155-159` |

### 2.3 Activation quantization

| 항목 | 공식 | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 학습 여부(LSQ) | **있음**: alpha와 activation step size를 공동 최적화(원 논문 취지), 공식 코드는 위 2단계로 분리 실행 | 있음(`qdrop_brecq_learn_act_scale` 기본 **True**) | ✅ 있음(단, 공동/순차 여부는 위 표 참고) | claim12/15 |
| 파라미터화 | 절대 scale(delta) 학습 파라미터, LSQ grad_factor 포함 | **09-21부터 독립 구현** `LSQActQuant`(절대 delta, `grad_factor=1/sqrt(numel*qmax)`) — Combined의 `s_mult`(배율)와 코드/파라미터 완전 분리 | ✅ 일치 | `adaround.py: LSQActQuant`, `_grad_scale` |
| lr | `--lr` 기본 **4e-4** | `DEFAULT_ACT_LR_BRECQ=4e-4` | ✅ 일치 | |
| scheduler | CosineAnnealingLR | 동일 | ✅ 일치 | |
| activation calibration(초기값) | `scale_method='mse'` | MSE observer(`act_observer` 기본 **mse**) | ✅ 일치 | `fake_quant.py: ActObserver._mse_range` |

### 2.4 종합 판정
weight quantization(비대칭·채널별·MSE)과 activation LSQ(독립 구현·공식 lr)는 **정확히
일치**. 남은 gap은 전부 **09-22 밤 시간 비용 문제로 의도적으로 되돌린 3가지**(2단계
분리, batch 2, iters 20000)와, **BRECQ 자체에 COCO 절이 없어 QDrop이 서술한 COCO
프로토콜을 빌려 썼다는 구조적 한계**(3장 참고)다. iters=2000은 BRECQ 자체 수렴
진단으로 뒷받침되지만, 2단계·batch 2는 성능 검증만 했지(2단계는 더 나쁨, batch는
무해함) "공식 그대로 두면 결과가 달라지는가"를 최종 확정 스케일에서 직접 비교하진
않았다(1-seed 스모크/정식 혼재).

---

## 3. QDrop (Wei et al. ICLR'22)

### 3.1 재구성 알고리즘

| 항목 | 공식(`wimh966/QDrop`, 논문 부록 E) | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| 기반 | BRECQ의 block-wise 재구성 + activation quantization drop | 동일(`optimize_brecq(qdrop_prob=0.5)`) | ✅ 일치 | |
| drop 위치 | (a) block 내부 activation quantizer, (b) block 입력(input_prob) | 동일 두 곳(`AdaRoundQuantConv2d.qdrop_prob`, `_mix()`) | ✅ 일치 | claim13에서 공식 구조 대조 |
| drop 확률 | 기본 0.5 | `qdrop_prob=0.5`(run_comparison.py 기본) | ✅ 일치 | |
| 학습 순서 | **공동 최적화**(alpha, activation step size를 함께) — BRECQ와 달리 QDrop은 2단계 아님, "we learn the weight and activation parameters together" | 공동 최적화(`two_stage` 파라미터는 QDrop 모드에 아예 적용 안 됨, `assert qdrop_prob==0` 가드) | ✅ 일치 | `brecq.py: optimize_brecq` docstring |
| calibration 이미지 수(COCO) | **256장**(논문 부록 E, "256 training samples taken from MS COCO") | 256장 | ✅ 일치 | 논문 원문 직접 확인(WebFetch) |
| batch size(COCO) | **2**(논문 부록 E, "batch size is set to 2") | `brecq_batch` 기본 **1**(09-23 되돌림) | ❌ **공식과 다름(의도적으로 되돌림)** | `runs/86_batch2_skiphead`에서 1↔2 결과가 노이즈 수준임을 실측했으나, 시간 비용 때문에 1로 되돌림. **QDrop 논문이 명시적으로 COCO에서 2를 쓴다고 적은 값**이라 나머지 두 항목보다 "공식 문장과 직접 어긋남"이 더 뚜렷 |
| head/neck 재구성 방식(COCO) | **head 미양자화**, backbone은 block-wise, **neck은 layer-wise**("we didn't quantize the head but applied block reconstruction to backbone and layer reconstruction to neck like BRECQ") | head 미양자화는 유지(`skip_head` 기본 **True**), **neck layer-wise는 09-23 되돌림**(`neck_layerwise` 기본 **False**, 전부 block-wise) | ❌ **공식과 다름(의도적으로 되돌림)** | neck_layerwise 켜면 재구성 대상이 17→35개로 늘어 6-seed 병렬 실행 시간이 감당 못 할 만큼 늘어남(09-22 밤 실측) |
| iters(COCO) | 논문: "다른 설정은 분류 실험과 동일" = **20000** | 2000 | ⚠️ 공식보다 작음, **QDrop 자체로 미검증** | BRECQ 진단(`runs/87`)을 빌려 씀. QDrop은 drop 메커니즘이 있어 수렴 양상이 BRECQ와 다를 수 있는데 직접 확인 안 함 |

### 3.2 Weight quantization grid

| 항목 | 공식(config yaml) | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| quantizer | `AdaRoundFakeQuantize`, `observer: MSEObserver`, `ch_axis: 0`(채널별), `symmetric: False`(비대칭) | `AdaRoundQuantConv2d(w_quant_mode="brecq")` — 채널별 비대칭 MSE | ✅ 일치 | 공식 config yaml 직접 확인 |

### 3.3 Activation quantization

| 항목 | 공식 | 우리 구현 | 상태 | 근거 |
|---|---|---|---|---|
| quantizer | `LSQFakeQuantize`, `observer: MSEObserver` | `ActObserver(method="mse")` + `LSQActQuant` | ✅ 일치 | |
| lr(ImageNet, COCO도 동일하게 사용) | **4e-5** | `DEFAULT_ACT_LR_QDROP=4e-5` | ✅ 일치 | 논문 부록 E 원문 직접 확인 |
| grad scaling | LSQ 공식(`1/sqrt(numel*qmax)`) | 동일 공식 구현(`_grad_scale`) | ✅ 일치 | |

### 3.4 종합 판정
drop 메커니즘, 공동 최적화 구조, weight/activation calibration, LSQ 전부 **정확히
일치**. 남은 gap 셋(batch, neck layer-wise, iters)은 **QDrop 논문 부록 E가 직접,
구체적으로 명시한 COCO 실험 설정**인데, 09-22 밤 시간 비용 문제로 되돌려서 지금 기본값과
어긋난다. 세 baseline 중 **QDrop이 논문 원문과의 괴리가 가장 뚜렷하다** — BRECQ는
애초에 COCO 절이 없어 "빌려 쓴 기준과 다르다" 정도지만, QDrop은 자기 논문 문장과 직접
다른 셈이다.

---

## 4. FP32 대비 AP 타당성 (claim17 계열, 6-seed 확정 수치)

| | FP32 | naive | AdaRound | QDrop | BRECQ |
|---|---|---|---|---|---|
| COCO_AP | 36.80 | 36.62(-0.18) | 36.63(-0.17) | 36.80(-0.00) | 36.73(-0.07) |
| LVIS_AP | 0.2589 | 0.2554(-0.0035) | 0.2586(-0.0003) | 0.2557(-0.0032) | 0.2554(-0.0035) |

W8A8 기준 FP32 대비 손상이 COCO_AP 0.2점, LVIS_AP 0.004 이내로 전부 작다. claim17에서
weight quantization은 W8에서 거의 무손실(W8A32 손상 -0.05)임을 확인했고, activation
MSE observer로 손상 대부분을 잡았기 때문에 이 정도 수준은 문헌 기준으로도 정상 범위다.
naive조차 FP32에 가깝고, 재구성 방법들이 naive보다 낫거나 비슷하다 — baseline이
붕괴됐다거나 부풀려졌다는 징후는 없다.

---

## 5. 종합: 지금 무엇이 남았나

### 확정적으로 일치(추가 작업 불필요)
- 재구성 loss 정규화, lr, reg_weight, warmup, b_range(세 방법 공통)
- weight quantization 방식(AdaRound: 대칭 채널별 MSE / BRECQ·QDrop: 비대칭 채널별 MSE)
- activation calibration(AdaRound: min-max / BRECQ·QDrop: MSE observer)
- activation LSQ 독립 구현 + 방법별 공식 lr(BRECQ 4e-4, QDrop 4e-5)
- QDrop의 drop 메커니즘(위치 2곳, 확률 0.5)
- BRECQ/QDrop 공통: calibration 256장(QDrop 논문 COCO 절과 정확히 일치)

### 의도적으로 되돌려 놓은 것 (공식 문서화, 기본값 = 미적용)
| 항목 | 공식 | 지금 기본값 | 켜는 플래그 |
|---|---|---|---|
| BRECQ 2단계 분리 | 2단계 | 공동 최적화 | `--brecq-two-stage` |
| batch size(BRECQ/QDrop 재구성 스텝) | 2(QDrop COCO 절) | 1 | `--brecq-batch 2` |
| neck layer-wise 재구성(BRECQ/QDrop) | layer-wise(QDrop COCO 절) | block-wise | `--neck-layerwise` |

### 미검증 (확인 안 하고 넘어간 것)
- AdaRound·QDrop 자체의 iters 수렴(BRECQ 진단을 가정으로 차용)
- AdaRound batch=1(공식 32)의 영향
- 되돌린 세 항목이 "성능이 아니라 순수 계산 결과"에 미치는 영향을 최종 확정 스케일(6-seed)에서 직접 비교한 적 없음(1-seed 스모크/정식 혼재)

이 문서는 baseline 구현 상태의 스냅샷이며, 위 "미검증"·"의도적으로 되돌림" 항목을
어떻게 처리할지(재검증 후 채택 / 영구히 각주로 남김 / 그대로 유지)는 후속 논의가
필요하다.
