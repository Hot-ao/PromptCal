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

> **09-23 밤 갱신 — 근거 재측정.** 이 문서 초판이 근거로 인용한 ablation
> `runs/85`~`runs/88`은 전부 커밋 `9125d1d`(09-23 17:24, weight 양자화를 채널별
> 비대칭 MSE로 바꾸고 LSQ를 독립 구현으로 분리) **이전** 코드에서 측정된 것이다.
> 그 전까지 weight는 채널별 max-abs 대칭 한 줄이었고(`git show 9125d1d^:src/quant/fake_quant.py`),
> activation MSE observer도 이미지별 탐색 평균이었다. 즉 **지금 baseline이 쓰지 않는
> 양자화 격자 위에서 내린 판단**이었다. 교차 확인: 같은 seed0인데 `runs/88`과
> `runs/91`의 **naive** 행이 다르다(lost 252 vs 272) — naive는 되돌린 설정들의
> 영향을 전혀 안 받으므로 코드가 달라졌다는 직접 증거다.
>
> 그래서 09-23 밤에 현재 코드로 1-seed 재측정을 돌렸다(`runs/92`~`runs/95`,
> seed 0, calib 256, full scale). 결과는 **§6**에 정리했고, 아래 각 절의 판정도
> 그에 맞춰 고쳤다. `runs/92_env_recheck`로 **baseline 4개가 `runs/91` seed0과
> 모든 지표에서 bit-identical함을 먼저 확인**했으므로(단 Combined는 예외, §6.5)
> 새 측정치를 기존 6-seed 분포와 직접 비교할 수 있다.

---

## 0. 세 방법이 공유하는 것

| 항목 | 공식 | 우리 구현 | 상태 |
|---|---|---|---|
| 양자화 대상 | 논문마다 다름(분류 모델 전체) | YOLO-World의 vision 경로 Conv2d 전부(DFL 제외, head는 `skip_head` 기본 True로 제외) | 구조상 불가피한 이식(아키텍처가 다름) |
| ~~text encoder 자동 제외~~ | — | **초판 서술 오류(09-23 수정)**: CLIP text tower는 Linear뿐이라 실제로 빠지지만, `set_classes()`가 `WorldModel.clip_model`에 캐싱하는 **CLIP vision tower의 patch-embed conv**(`clip_model.model.visual.conv1`, 3→768 k32, weight 2.36M)가 `wrap_convs`에 같이 잡혀 있었다 | ❌ → ✅ 수정됨(`skip_names={"clip_model"}`) |
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
| iters / 이미지 수 | 논문 주요 표: 2048장·20k iters(4bit), AIMET 기본 `default_num_iterations=10000`; Fig.4: 256장으로도 FP32 대비 2% 이내 | calib=256장, `--recon-iters-ada` 기본 **1000** | ❌ **교란변수 — §6.1 참고** | iters=1000은 QDrop/BRECQ(2000)의 절반이라 "방법 차이"와 "예산 차이"가 섞인다(`run_comparison.py`가 실행 중 경고까지 띄운다). 09-23 재측정: 2000으로 맞추면 lost 273→244, Heval_flip 5.99%→4.99% |
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
| calibration 방식(8bit 비교 실험) | "관측된 min/max로 scale 설정" | **min-max**(AdaRound 모드만 `calibrate(... act_observer="minmax")`로 재보정) | ✅ **논문 충실, 그대로 유지 권장 — §7.2** | 다른 넷은 MSE observer지만, 09-24 6-seed 측정 결과 이건 **불이익이 아니라 트레이드오프**다: MSE로 바꾸면 COCO는 좋아지고(AP +0.108, lost −24) **LVIS는 6/6 전부 나빠진다**(LVIS_AP −0.0035, LVIS_lost +185 = +29%). 논문의 held-out vocabulary 논지가 LVIS 쪽이므로 as-published(min-max)가 AdaRound에게 유리한 쪽이다. 격리 플래그 `--adaround-act-observer {minmax,mse}`(기본 minmax) |

### 1.4 종합 판정
핵심 알고리즘(반올림 파라미터화, loss, 정규화, optimizer 설정)은 **정확히 일치**.
weight granularity(채널별 vs 논문 본문의 레이어 전체)는 논문이 각주로 허용한 변형이며
실측 근거로 문서화됨.

**재측정 결과(§7)**: 남은 두 gap의 성격이 서로 다르다.
- **iters 1000은 진짜 교란변수였다.** 2000으로 맞추니 decision 지표가 실제로 개선된다
  (6-seed: Heval_flip 5.93→5.14%, Top1_flip 0.553→0.472%, lost 263→234). 변경 안 한
  조건들의 RNG 드리프트 대비 약 7배 크기다(§7.1). **`runs/97`을 AdaRound 확정값으로 채택.**
- **min-max observer는 교란변수가 아니라 트레이드오프였다**(§7.2). MSE로 바꾸면 COCO는
  일관되게 좋아지고 LVIS는 일관되게 나빠진다(둘 다 6/6). as-published 유지가 맞다.

batch=1(공식 32)은 여전히 미검증.

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
| **학습 순서** | **순차 2단계**: (1) `act_quant=False`로 alpha만, (2) weight hard 고정 후 activation(LSQ) `iters_a=5000`, `lr=4e-4`, cosine, L_p(2.4) 별도 학습 | **공동 최적화**(`brecq_two_stage` 기본 **False**, 09-23 되돌림) | ❌ **공식과 다름(의도적으로 되돌림)** | `_brecq_act_stage()`로 2단계 구현은 돼 있고 `--brecq-two-stage`로 켤 수 있으나 기본 꺼짐. **09-23 재측정으로 근거가 바뀜**: 이제 시간 문제가 아니라 **2단계가 실제로 더 나쁘다**. 현재 코드·확정 스케일에서 Heval_flip 3.70%→**5.10%**, Top1_flip 0.37%→**0.51%**, lost 180→**210**으로 셋 다 6-seed 범위 밖(§6.3) |
| iters(weight 단계) | ImageNet 표: **20000** | 2000(`recon_iters_strong` 기본값) | ⚠️ 공식보다 작음, **현재 코드로 재검증됨** | `runs/87`은 구 코드 + 확정과 다른 설정(two_stage=True, batch=2)이라 무효. 09-23 현재 코드·확정 설정으로 재측정(`runs/93_iters_recheck`): 20000으로 올려도 COCO_AP 36.73→36.76, Heval_flip 3.70%→3.51%, lost 180→155로 **전부 6-seed 범위 안**(비용은 6.2배). 단 "수렴했다"는 근거는 아니다 — nearest 대비 flip이 5.20%→**12.87%**라 alpha는 2000에서 전혀 수렴하지 않았고, 그런데도 최종 지표가 노이즈 안이라는 뜻이다(§6.2) |
| calibration 이미지 수 | ImageNet 1024장(BRECQ 자체 COCO 절 없음) | 256장 | 아래 3장 참고(QDrop 인용 절차 기준으로는 일치) | |
| batch size(재구성 스텝) | 32(ImageNet) | `brecq_batch` 기본 **1** | ⚠️ 공식보다 작음 | `runs/86`은 구 코드라 무효. 09-23 재측정에서는 batch 2를 neck layer-wise와 함께 켜서 QDrop 쪽은 무해~약간 유리, BRECQ 쪽은 2단계 때문에 악화였다(§6.3) — batch 단독 효과는 여전히 분리 안 됨 |

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
일치**. 구조적 한계는 그대로다 — **BRECQ 자체에 COCO 절이 없어 QDrop이 서술한 COCO
프로토콜을 빌려 썼다**(3장 참고).

**09-23 재측정 이후 남은 gap의 성격이 바뀌었다**(§6):
- **iters 2000**: 현재 코드·확정 설정에서 20000과 비교해 최종 지표가 전부 6-seed 범위
  안. 유지해도 되며, 이제 근거가 "구 코드 진단 차용"이 아니라 **BRECQ 자체·현재 코드
  직접 측정**이다. 다만 서술은 "수렴해서"가 아니라 "수렴하지 않았는데도 최종 지표가
  노이즈 안이라서"로 정확히 적어야 한다.
- **2단계 분리**: 시간이 아니라 **성능 때문에** 안 쓴다. 공식대로 켜면 decision 지표
  셋이 6-seed 범위 밖으로 악화된다. 논문에는 "의도적 이탈 + 실측 근거"로 명시할 것.
- **batch 2**: 단독 효과는 아직 분리 안 됨(§6.3에서 neck layer-wise와 묶여 있었다).

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
| batch size(COCO) | **2**(논문 부록 E, "batch size is set to 2") | `brecq_batch` 기본 **1** | ❌ 공식과 다름(의도적으로 되돌림) | 09-23 재측정에서도 **차이가 노이즈로 설명된다**(§6.3): QDrop 기준 COCO_AP·Heval_flip은 6-seed 범위 안, Top1_flip·lost만 경계에서 간신히 밖인데 **RNG 스트림 위치까지 다르다**. 빌드 비용만 2.7배 — **성능상 켤 이유 없음.** 남는 논거는 "논문 부록 E 문장과의 충실도" 하나뿐(서술 문제) |
| head/neck 재구성 방식(COCO) | **head 미양자화**, backbone은 block-wise, **neck은 layer-wise**("we didn't quantize the head but applied block reconstruction to backbone and layer reconstruction to neck like BRECQ") | head 미양자화는 유지(`skip_head` 기본 **True**), neck은 block-wise(`neck_layerwise` 기본 **False**) | ❌ 공식과 다름(의도적으로 되돌림) | 위 batch 항목과 같은 판정 — §6.3에서 둘을 묶어 측정했고 결과 차이가 노이즈로 설명된다. 재구성 대상 17→35개로 빌드 2.7배. **성능상 켤 이유 없음** |
| iters(COCO) | 논문: "다른 설정은 분류 실험과 동일" = **20000** | 2000 | ⚠️ 공식보다 작음, **QDrop 자체로는 여전히 미검증** | 09-23 재측정은 BRECQ 쪽만 20000을 돌렸다(§6.2). QDrop은 drop 메커니즘 때문에 수렴 양상이 다를 수 있는데 직접 확인 안 함 — 남은 미검증 항목 |

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
구체적으로 명시한 COCO 실험 설정**이라, 세 baseline 중 QDrop이 논문 원문과의 괴리가
가장 뚜렷했다.

**09-23 재측정 결과**(§6.3): batch 2 + neck layer-wise를 공식대로 켜도 **결과 차이가
노이즈로 설명된다** — 4개 지표 중 둘은 6-seed 범위 안, 둘은 경계에서 간신히 밖인데
비교 두 팔의 RNG 스트림 위치까지 다르다. 즉 **성능을 근거로 켤 이유도, 끄고 있는 것을
문제 삼을 이유도 없다**(초판의 "노이즈 수준" 판단이 현재 코드에서도 유지된다).
비용만 빌드 2.7배(2922s→7762s)다.

따라서 이 둘은 **성능 문제가 아니라 서술 문제**로 남는다: 논문에 "QDrop 부록 E의 COCO
프로토콜을 따랐다"고 쓸 수 없고, "batch 1·block-wise로 바꿨으며 그 차이는 노이즈 수준임을
실측했다"고 각주로 밝혀야 한다. iters 20000은 QDrop 자체로는 여전히 미검증이다.

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
**붕괴됐거나 부풀려졌다는 징후는 없다**.

> **09-23 재측정 보정(§6.1)**: 붕괴는 아니지만 **AdaRound 열은 과소평가돼 있다.**
> 이 표의 AdaRound는 iters 1000 + min-max observer로 측정된 것인데, 둘 다 다른
> 조건과 다른 설정이다. QDrop/BRECQ와 같은 조건(iters 2000 + MSE observer)으로
> 맞추면 seed0에서 COCO_AP 36.63→**36.81**로 표 안의 어떤 조건보다 높아진다.
> 이 표를 논문에 그대로 쓰려면 AdaRound 설정 차이를 각주로 반드시 밝힐 것.
>
> 모델 크기를 함께 인용했다면 **12.13 MiB → 9.88 MiB로 정정**해야 한다(§6.4).

---

## 5. 종합: 지금 무엇이 남았나

### 확정적으로 일치(추가 작업 불필요)
- 재구성 loss 정규화, lr, reg_weight, warmup, b_range(세 방법 공통)
- weight quantization 방식(AdaRound: 대칭 채널별 MSE / BRECQ·QDrop: 비대칭 채널별 MSE)
- activation LSQ 독립 구현 + 방법별 공식 lr(BRECQ 4e-4, QDrop 4e-5)
- QDrop의 drop 메커니즘(위치 2곳, 확률 0.5)
- BRECQ/QDrop 공통: calibration 256장(QDrop 논문 COCO 절과 정확히 일치)

### 의도적 이탈로 유지 (현재 코드로 재확인됨, 논문에 각주로 명시할 것)
| 항목 | 공식 | 지금 기본값 | 근거 |
|---|---|---|---|
| batch size(재구성 스텝) | 2(QDrop COCO 절) | 1 | 차이가 **노이즈로 설명됨**(§6.3). 비용만 2.7배 → 켤 이유 없음. 각주로 "노이즈 수준임을 실측" 명시 |
| neck layer-wise 재구성 | layer-wise(QDrop COCO 절) | block-wise | 위와 동일(§6.3에서 묶어 측정) |
| BRECQ 2단계 분리 | 2단계 | 공동 최적화 | 공식대로 켜면 Heval_flip 3.70%→5.10%, lost 180→210으로 **6-seed 범위 밖 악화**(§6.3). 노이즈로 설명 안 됨 |
| iters(BRECQ/QDrop) | 20000 | 2000 | BRECQ 기준 20000과 최종 지표 차이가 6-seed 범위 안, 비용만 6.2배(§6.2) |

### 해결됨 (09-24 6-seed, §7)
| 항목 | 조치 | 결과 |
|---|---|---|
| AdaRound iters 1000 | `--recon-iters-ada 2000`으로 통일, 6-seed 재측정(`runs/97`) | **교란변수 맞았음.** decision 지표 실제 개선 → `runs/97`을 확정값으로 채택 |
| AdaRound만 min-max observer | MSE 통일판 6-seed 병행 측정(`runs/98`) | **교란변수 아니었음.** COCO↔LVIS 트레이드오프 → **as-published(min-max) 유지**, MSE판은 ablation으로 병기 |

### 미검증으로 남은 것
- **QDrop 자체의 iters 수렴** — 09-23 재측정은 BRECQ만 20000을 돌렸다. drop 메커니즘 때문에 수렴 양상이 다를 수 있음
- **AdaRound batch=1**(공식 32)의 영향
- **batch 2 단독 효과** — §6.3에서 neck layer-wise와 묶여 측정됐다
- **6-seed 확인** — §6은 전부 1-seed다. 판정은 `runs/91` 6-seed 분포 대비 범위 안/밖으로만 했다

---

## 6. 09-23 밤 재측정 (현재 코드, seed 0, calib 256, full scale)

초판이 인용한 `runs/85`~`runs/88`이 커밋 `9125d1d` 이전 코드(채널별 max-abs 대칭 weight,
이미지별 MSE observer 평균)에서 측정된 것이라, 현재 코드로 다시 잰 결과다.
환경도 재구축했다(`.venv`: ultralytics 8.4.121 / torch 2.10.0+cu128 /
torchvision 0.25.0+cu128 / numpy 1.26.4 — 기존 editable 설치가 가리키던
`/home/taeho/Mamba-YOLO`가 삭제돼 실행 자체가 불가능한 상태였다).

> **비교할 때 주의 — RNG 스트림 위치.** `torch.manual_seed`는 실행 시작에 한 번만
> 걸리고 조건은 `--conditions` 순서대로 순차 빌드되므로, **조건 목록이 다르면 같은
> 조건이라도 다른 난수를 받는다.** 예: qdrop은 `runs/92`(naive,adaround,qdrop,brecq,
> combined)에서 3번째, `runs/95`(naive,qdrop,brecq)에서 2번째다. 이 재추첨만으로도
> 6-seed 분포만큼의 변동이 생긴다.
>
> 이게 실제로 그렇다는 증거: **항상 1번째로 빌드되는 naive는 `runs/92`~`runs/95`
> 네 run에서 모든 지표가 bit-identical**(AP 36.68 / Heval_flip 5.34% / lost 272)인
> 반면, 뒤 순서의 조건들은 그렇지 않다. 따라서 **6-seed 범위를 살짝 벗어나는 정도의
> 차이는 설정 효과로 읽으면 안 된다.**

판정 기준이 되는 `runs/91` 6-seed 분포:

| 조건 | COCO_AP | Heval_flip | Top1_flip | lost |
|---|---|---|---|---|
| naive | 36.57~36.68 | 5.34~6.80% | 0.59~0.70% | 272~307 |
| adaround | 36.60~36.67 | 4.90~7.12% | 0.52~0.59% | 242~273 |
| qdrop | 36.77~36.84 | 3.47~4.68% | 0.38~0.42% | 181~190 |
| brecq | 36.71~36.77 | 3.07~4.24% | 0.35~0.37% | 155~180 |
| combined | 36.56~36.74 | 4.79~6.43% | 0.51~0.59% | 226~275 |

### 6.0 환경 재현 검증 (`runs/92_env_recheck`)
`runs/91` seed0과 동일 설정으로 재실행. **naive/AdaRound/QDrop/BRECQ는 COCO·LVIS
모든 지표가 소수점까지 동일**(LVIS_AP는 4자리까지). 아래 측정치를 기존 6-seed와
직접 비교해도 된다. **Combined만 예외 — §6.5.**

### 6.1 AdaRound 교란변수 격리 (`runs/94_adaround_confound`) — **1-seed 예비, §7이 대체함**

| AdaRound 설정 | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP | LVIS lost |
|---|---|---|---|---|---|---|
| naive(기준) | 36.68 | 5.34% | 0.59% | 272 | 0.2556 | 902 |
| **확정값**(iters 1000 + min-max) | 36.63 | 5.99% | 0.58% | 273 | 0.2595 | 741 |
| iters 2000(예산만 정렬) | 36.61 | 4.99% | 0.47% | 244 | 0.2566 | 620 |
| **iters 2000 + MSE observer** | **36.81** | **4.15%** | **0.42%** | **193** | 0.2542 | 698 |
| *(참고) qdrop / brecq* | 36.77 / 36.73 | 3.97 / 3.70% | 0.39 / 0.37% | 183 / 180 | | |

COCO_AP·Top1_flip·lost 전부 6-seed 범위를 크게 벗어난다(COCO_AP 36.81은 최댓값
36.67보다 +0.14, std 0.025 기준 약 7σ). **COCO_AP는 observer가 지배적**(36.61→36.81),
**decision 지표는 둘 다 기여**(5.99→4.99→4.15). 단 **LVIS_AP는 반대 방향**
(0.2595→0.2542)이라 트레이드오프가 있다.

> **이 절은 1-seed 예비 결과다.** 09-24 6-seed(`runs/97`,`runs/98`)로 다시 재면 결론이
> 갈린다 — iters는 교란변수가 맞았지만, **observer는 교란변수가 아니라 COCO↔LVIS
> 트레이드오프였다.** 여기서 본 COCO_AP 36.81은 6-seed 평균으로는 36.757이고
> QDrop(36.807)을 넘지 못한다. **§7을 기준으로 삼을 것.**

### 6.2 BRECQ iters 2000 vs 20000 (`runs/93_iters_recheck`, 확정 설정)

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS lost | LVIS_AP | nearest 대비 flip | 빌드 |
|---|---|---|---|---|---|---|---|---|
| iters 2000(확정) | 36.73 | 3.70% | 0.37% | 180 | 622 | 0.2554 | 5.20% | 1914s |
| iters 20000 | 36.76 | 3.51% | 0.33% | 155 | 630 | 0.2542 | **12.87%** | 11836s |

거의 전부 6-seed 범위 안(Top1_flip 0.33만 최솟값 0.35보다 살짝 아래). 두 팔의 RNG
스트림 위치도 다르므로(§6 머리말) 이 정도 차이는 설정 효과로 읽을 수 없다.
**비용 6.2배 대비 얻는 것 없음 → 2000 유지.**

다만 근거의 성격을 정확히 적을 것: nearest 대비 flip이 **5.2%→12.9%로 2.5배**다.
**alpha는 2000에서 수렴하지 않았다.** "수렴했으니 충분"이 아니라 "수렴하지 않았는데도
최종 지표가 노이즈 안"이 맞는 서술이다.

### 6.3 공식 COCO 설정 vs 확정 설정 (`runs/95_official_defaults`)

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS lost | 빌드 |
|---|---|---|---|---|---|---|
| **QDrop** 확정(batch 1, block-wise) | 36.77 | 3.97% | 0.39% | 183 | 617 | 2922s |
| **QDrop** 공식(batch 2 + neck layer-wise) | 36.83 | 3.96% | 0.37% | **172** | 627 | 7762s |
| **BRECQ** 확정(공동 최적화) | 36.73 | 3.70% | 0.37% | 180 | 622 | 1914s |
| **BRECQ** 공식(2단계 + batch 2 + neck) | 36.79 | **5.10%** | **0.51%** | **210** | 653 | 3594s |

- **QDrop**(batch 2 + neck layer-wise): COCO_AP 36.83과 Heval_flip 3.96%는 6-seed
  범위(36.77~36.84 / 3.47~4.68%) **안**, Top1_flip 0.37%와 lost 172만 경계에서
  각각 0.01pp·9(5%)만큼 밖이다. 거기에 두 팔의 **RNG 스트림 위치가 다르다**(위 주의
  참고) — 즉 **차이가 노이즈로 완전히 설명된다.** 초판의 "노이즈 수준" 판단이 현재
  코드에서도 유지된다. **성능을 근거로 켤 이유 없음**(§3.4).
  *(09-23 최초 정리에서 lost 172 하나를 근거로 "약간 유리 → 채택 권장"이라고 적었던
  것은 과대해석이었다. RNG 스트림 차이를 계산에 넣지 않았다.)*
- **BRECQ**(+ 2단계): decision 지표 셋이 6-seed 범위를 **크게** 벗어나 악화됐다
  (Heval_flip 3.70→5.10%는 범위 상한 4.24%보다 0.86pp 위). 이 크기는 RNG 재추첨으로
  설명되지 않는다. 그리고 **원인을 특정할 수 있다** — QDrop 팔은 `batch 2 + neck
  layer-wise`만 바꿔서 노이즈 수준이었으므로, 악화의 원인은 **2단계 분리 단독**이다.

### 6.4 CLIP 인코더가 양자화 대상에 섞여 있던 문제 (수정 완료)
`set_classes()`가 `WorldModel.clip_model`에 CLIP(ViT-B/32) 전체를 캐싱하는데,
`wrap_convs`가 `m.model` 하위를 무조건 재귀하면서 **CLIP vision tower의 patch-embed
conv**(`clip_model.model.visual.conv1`, 3→768 k32, weight 2,359,296)까지 감쌌다.

- detector forward에서 호출되지 않아 **정확도 영향은 없다**(RNG도 소비하지 않아
  수정 전후 결과는 모델 크기를 빼면 bit-identical).
- 매 run의 `[adaround][warn] 1/53 conv 입력 미포착` 경고의 정체가 이것이었다.
- **`quantized_weight_mib()`가 12.13 MiB로 보고했지만 실제 detector는 9.88 MiB**
  — 2.25 MiB(22.8%)가 허수였다. naive 포함 5개 조건 전부 동일하게 과대계상.
- 수정: `quant_model.ALWAYS_SKIP_NAMES = {"clip_model"}` — DFL과 같은 방식으로
  `wrap_convs` 안에서 막는다. `scripts/` 아래 호출처가 50곳 가까이 되는데 전부 같은
  문제를 갖고 있었으므로 호출처마다 고치지 않고 한곳에서 차단했다(`src/quant/`·
  `pipeline/quant/` 사본 동기화 유지). 수정 후 실측 **9.878 MiB**, CLIP 기여 0.
- **논문에 모델 크기를 인용했다면 12.13 → 9.88 MiB로 정정할 것.**

### 6.5 Combined만 같은 seed에서 재현되지 않는다 (새 발견, 미해결)

| | `runs/91` seed0 | `runs/92` 재현(동일 seed·코드) | 차이 |
|---|---|---|---|
| combined COCO_AP | 36.61 | 36.64 | +0.03 |
| combined Heval_flip | 4.79% | 5.26% | **+0.47pp** |
| combined lost | 226 | 239 | +13 |
| combined LVIS_AP | 0.2580 | 0.2586 | +0.0006 |

baseline 4개는 bit-identical인데 Combined만 흔들린다 →
`optimize_promptcal_scale_neighbor` 안에 비결정적 CUDA 연산(index/scatter backward의
atomicAdd 계열)이 있다는 뜻이다. `cudnn.deterministic=True`는 conv 알고리즘만 고정할
뿐 이걸 못 잡는다.

**왜 문제인가**: 변동폭 0.47pp가 Combined와 AdaRound의 격차(1.2pp)의 약 40%다. 즉
`runs/91`의 6-seed 평균에는 seed 분산과 **실행 분산이 섞여 있고**, Combined의 마진을
주장할 때 이 분산이 계산에 안 들어가 있다.

**조치**: `--deterministic` 플래그 추가(`torch.use_deterministic_algorithms(warn_only=True)`,
기본 꺼짐). 켜고 돌리면 경고로 비결정 연산 위치를 특정할 수 있다. 근본 해결 전까지는
Combined 수치에 실행 분산을 명시할 것.

---

## 7. 09-24 6-seed 확정 재측정 (`runs/97`, `runs/98`)

§6은 전부 1-seed 예비였다. AdaRound의 두 gap을 6-seed로 다시 쟀다. 두 run 모두
`--recon-iters-ada 2000`(QDrop/BRECQ와 예산 동일)이고, **`--adaround-act-observer`
하나만 다르다**(97=min-max=as-published, 98=mse=통일).

### 7.0 측정 무결성 (세 겹 확인)
- **naive / QDrop / BRECQ가 `runs/97`↔`runs/98` 6-seed 전부 bit-identical.**
  AdaRound의 min-max 재보정은 난수를 소비하지 않으므로 뒤에 빌드되는 조건들의 RNG
  스트림 위치가 그대로다 → **두 run의 차이가 AdaRound 열 하나로 완전히 격리됐다.**
- **naive 6-seed가 `runs/91`과도 완전 동일** → CLIP 수정(§6.4)을 `wrap_convs`로 옮긴
  리팩터까지 결과 중립임이 6-seed 규모로 확인됐다. 모델 크기 9.88 MiB, AdaRound 경고 0건.
- **baseline은 실행 분산이 0이다.** 즉 6-seed 분산 전체가 seed 효과(클래스 분할 +
  torch RNG)이고, 기준자(yardstick)로 쓰기에 적합하다.

### 7.1 AdaRound iters 1000 → 2000 (`runs/91` vs `runs/97`)

iters를 바꾸면 소비 난수가 달라져 뒤에 빌드되는 조건들의 스트림이 밀린다. 그 조건들은
방법이 전혀 안 바뀌었으므로 **RNG 드리프트의 음성 대조군** 역할을 한다:

| 지표 | **adaround**(실제 변경) | qdrop | brecq | combined | 판정 |
|---|---|---|---|---|---|
| COCO_AP | +0.018 | +0.010 | +0.023 | −0.003 | 드리프트와 **구분 불가** |
| Heval_flip | **−0.785** | +0.082 | +0.068 | +0.107 | 부호 반대, **~7배** |
| Top1_flip | **−0.082** | −0.002 | −0.012 | −0.007 | **~7배** |
| lost | **−28.7** | −4.3 | −2.5 | +3.0 | **~7배** |
| LVIS_AP | −0.0010 | −0.0000 | +0.0006 | −0.0004 | 드리프트 수준 |

**예산 정렬은 decision 지표를 실제로 개선한다(AP는 아님).** `runs/97`을 AdaRound
확정값으로 채택한다.

> **판정 기준 교정**: "6-seed 범위 밖"은 너무 약한 기준이다 — 아무것도 안 바꾼 qdrop의
> lost와 brecq의 Top1_flip도 평균이 기존 범위를 벗어났다. 앞으로는 **이 음성 대조군
> 대비 배수**로 판정한다.

### 7.2 AdaRound activation observer: min-max(as-published) vs MSE(통일)

| 지표 | min-max | MSE | Δ | MSE 우세 | (참고) qdrop / brecq |
|---|---|---|---|---|---|
| COCO_AP | 36.648 | **36.757** | +0.108 | **6/6** | 36.807 / 36.755 |
| COCO Heval_flip | 5.142 | **4.567** | −0.575 | **6/6** | 4.090 / 3.705 |
| COCO Top1_flip | 0.472 | **0.450** | −0.022 | 4/6 | 0.397 / 0.345 |
| COCO lost | 234 | **210** | −24.3 | 5/6 | 180 / 166 |
| LVIS_AP | **0.2577** | 0.2542 | −0.0035 | **0/6** | 0.2556 / 0.2560 |
| LVIS Top1_flip | **2.813** | 3.603 | +0.790 | **0/6** | 2.867 / 2.605 |
| LVIS lost | **639** | 824 | +185 (**+29%**) | **0/6** | 642 / 606 |

**교란변수가 아니라 트레이드오프다.** MSE observer는 COCO(=calibration vocabulary)를
일관되게 개선하고 LVIS(=held-out vocabulary)를 일관되게 악화시킨다. 방향이 6/6으로
갈리므로 우연이 아니다.

메커니즘은 그럴듯하다: MSE observer는 COCO train2017 분포에서 L_2.4 오차를 최소화하도록
activation 범위를 클리핑한다 — in-domain 해상도를 얻는 대신 outlier를 버린다. LVIS
1203개 vocabulary에서는 그 outlier가 중요해진다. **"calibration vocabulary에 맞춘
목적함수가 held-out vocabulary로 전이되지 않는다"는 이 문서 바깥(논문 본 논지)의
주장과 같은 현상이다.**

**결론 — as-published(min-max)를 AdaRound 확정값으로 유지한다.** 근거:
1. 원 논문이 명시한 설정이라 별도 변호가 필요 없다.
2. **불이익이 아니다.** min-max는 AdaRound에게 LVIS 3개 지표 전부에서 유리하고,
   LVIS_AP 0.2577은 **모든 baseline 중 1위**다(qdrop 0.2556, brecq 0.2560).
3. MSE판은 ablation으로 병기한다 — "COCO만 보면 MSE가 낫다(36.757, BRECQ와 동급)"를
   밝혀 두면 "COCO에서 AdaRound를 불리하게 뒀다"는 지적을 미리 막을 수 있다.

*(09-23 1-seed 예비(§6.1)에서 "MSE로 통일 권장"이라고 적었던 것은 COCO_AP 하나만 보고
내린 판단이었다. LVIS를 포함한 6-seed에서는 성립하지 않는다.)*

### 7.3 확정 baseline 표 (`runs/97`, 6-seed 평균)

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP |
|---|---|---|---|---|---|
| naive | 36.617 | 5.945% | 0.667% | 291 | 0.2554 |
| AdaRound | 36.648 | 5.142% | 0.472% | 234 | 0.2577 |
| QDrop | 36.807 | 4.090% | 0.397% | 180 | 0.2556 |
| BRECQ | **36.755** | **3.705%** | **0.345%** | **166** | 0.2560 |
| *(참고) Combined* | 36.642 | 5.602% | 0.527% | 248 | 0.2576 |

seed별 짝비교(같은 seed = 같은 클래스 분할)에서 Combined는 naive에만 전승이고,
AdaRound·QDrop·BRECQ에게는 decision 지표 **0/6 전패**다. 유일한 우세는 LVIS_AP
(qdrop 대비 +0.0020, brecq 대비 +0.0016, 둘 다 6/6)인데 AdaRound(0.2577)가 거의
같다. **넘어야 할 선은 BRECQ의 `Heval_flip 3.705% / lost 166`이다.**

### 7.4 Combined 비결정성 — §6.5의 결정적 확증
`runs/97`↔`runs/98`에서 naive/QDrop/BRECQ가 6/6 bit-identical인 가운데,
**Combined만 seed 2·4에서 결과가 달라졌다(2/6, 약 33%)**. RNG 스트림·코드·입력이
모두 동일함이 나머지 세 조건으로 증명되므로, 비결정성의 위치가
`optimize_promptcal_scale_neighbor` 내부로 확정된다.

**09-24 해결됨**: full scale에서 같은 seed로 Combined를 2회 빌드하면 기본 설정에서는
s_mult가 **52/52 conv 전부 다르고**(max|Δ| 2.48e-01), `--deterministic`
(`torch.use_deterministic_algorithms(True, warn_only=True)`)을 켜면 **0/52**로
완전히 재현된다. 비용은 빌드 +2~3%.

그리고 **`--deterministic`은 baseline 수치를 바꾸지 않는다** — AdaRound/BRECQ/QDrop을
켜고/끄고 비교하면 quant weight·LSQ delta가 전부 bit-identical이다. §7.3 확정 표는
그대로 유효하다.

→ **Combined를 다루는 실행에는 `--deterministic`을 켤 것.** 그러면 설계 A vs B를
같은 seed에서 짝비교할 때 차이가 전부 실재하는 값이 된다.

---

이 문서는 baseline 구현 상태의 스냅샷이다. §6은 전부 **1-seed**이고, 판정은 `runs/91`
6-seed 분포 대비 범위 안/밖으로만 했으며, 비교 두 팔의 RNG 스트림 위치가 다른 경우가
있다(§6 머리말). 따라서 **6-seed 범위를 크게 벗어난 것만 설정 효과로 읽었다**:
AdaRound 교란변수(§6.1)와 BRECQ 2단계(§6.3) 둘뿐이고, 나머지는 전부 노이즈로 분류했다.

§5의 "반드시 고쳐야 할 것"(AdaRound iters 2000 + MSE observer)을 헤드라인 표에
반영하려면 그 설정으로 6-seed를 다시 돌려야 한다. 이때는 조건 목록을 `runs/91`과
동일하게 유지해 RNG 스트림 위치를 맞출 것.
