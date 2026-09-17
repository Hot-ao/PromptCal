# PromptCal-PTQ 작동 원리 상세 설명 (2026-09-08 갱신)

`PROMPTCAL_METHOD_SPEC.md`는 세 번의 실패한 시도(alpha rounding만으로 margin을
맞추려던 접근)를 기록한 문서라 지금 방법을 설명하지 않는다. 이 문서는 **지금
실제로 동작하는** 파이프라인 전체 — 문제 정의부터 측정 도구, baseline, 제안
방법, 평가지표까지 — 를 하나로 정리한다. 코드 경로를 항상 같이 적는다.

> **09-17 주의**: 이 문서는 09-08 시점 설계(파이프라인 구조·측정 도구·평가지표
> 개념은 지금도 유효)를 기준으로 쓰였다. 이후 §5.2(s_mult per-channel→
> per-tensor)와 §5.4(margin_loss가 identity-aware+intrusion 탐지로 보강됨)가
> 설계 변경으로 **superseded**됐다 — 해당 절 예시 코드를 그대로 재현 코드로
> 쓰지 말 것. **지금 확정 설계와 하이퍼파라미터는
> [`PROMPTCAL_CURRENT_MODEL_V2.md`](PROMPTCAL_CURRENT_MODEL_V2.md)**를 볼 것,
> 이 문서는 "왜 이런 구조로 설계했는가"의 개념적 배경 설명으로만 참고할 것.

---

## 1. 무엇을 풀려는 문제인가

YOLO-World 같은 **open-vocabulary detector**는 고정된 클래스 목록이 아니라
임의의 텍스트 프롬프트(class name) 목록을 입력받아, 이미지의 각 region이 그
프롬프트들과 얼마나 유사한지를 점수로 매긴다. 이 점수가 곧 "이 region은 어떤
class인가"라는 **의사결정**이다.

이 detector를 배포하려고 8bit로 양자화(PTQ)하면 일어나는 문제:

- **총량 지표(AP)는 거의 안 변한다** (37.8 → 37.3, -0.5 정도).
- 하지만 **개별 region의 "1등 프롬프트가 무엇인가"라는 결정은 심하게 흔들린다** —
  특히 경쟁하는 두 프롬프트의 점수 차이(margin)가 작을 때, 의미적으로 가까운
  프롬프트 방향으로, 그리고 calibration에서 그 프롬프트를 안 봤을 때 심하다.

즉 **"양자화가 텐서를 얼마나 잘 복원하는가"(reconstruction)와 "양자화가
region-prompt 의사결정을 얼마나 보존하는가"(decision preservation)는 다른
문제**라는 게 이 연구 전체의 핵심 주장이다. AdaRound/BRECQ/QDrop처럼
reconstruction을 아주 잘하는 방법도 후자는 못 지킨다(§5 참고). 우리 방법은
후자를 직접 타깃한다.

---

## 2. 측정 도구: region–prompt 유사도 행렬 하나로 전부 계산한다

### 2.1 왜 cv4(ContrastiveHead)인가

YOLO-World의 detection head(`WorldDetect`)는 box regression(`cv2`)과
classification(`cv3` 경로에 붙는 `cv4`)을 분리해서 계산한다. `cv4`는 각 FPN
level에서 "region feature ↔ text embedding" contrastive 유사도를 계산하는
**ContrastiveHead**들의 ModuleList다. 레벨별로 `[B, num_prompts, H, W]`의
pre-sigmoid 유사도 맵을 낸다.

- 3개 level: stride 8/16/32 → 640 입력 기준 80×80, 40×40, 20×20 grid.
- flatten+concat하면 `[total_anchors(=8400), num_prompts]` 행렬 하나가 된다.
- 이 값은 **모델의 최종 classification logit 그 자체**다(검증됨: 전역 argmax ==
  모델 정식 예측). head 출력 튜플을 파싱하는 것보다 모호함이 없고, pre-sigmoid라
  margin 기반 지표에도 유리하다.

### 2.2 `src/harness.py` — `SimilarityHarness`

```python
class SimilarityHarness:
    def __init__(self, model, device):
        self._head = ...              # DetectionModel.model[-1] (WorldDetect)
        self._level_buf = {}
        for i, sub in enumerate(self._head.cv4):
            sub.register_forward_hook(lambda ...: self._level_buf[i] = out)

    def run_image(self, img_tensor, image_id):
        self.model(img_tensor)                    # forward 1회
        sim = assemble(level_buf)                 # [anchors, P]로 조립
        return SimilarityRecord(image_id, sim)
```

FP32 모델과 양자화 모델에 **각각** 이 하네스를 씌워서 같은 이미지를 넣고 두
`sim` 행렬을 비교하는 것이, 이 코드베이스 전체 실험(motivation, baseline
비교, 제안 방법 학습·평가)의 공통 연산이다. precision-agnostic — FP/양자화
어느 쪽에 씌우든 코드가 동일하다.

---

## 3. Fake-quantization 스킴 (모든 baseline의 공통 바닥)

실제 INT8 엔진이 아니라 **quantize→dequantize**로 정밀도만 int8로 떨어뜨리는
시뮬레이션이다(연구용 fake-quant — §8.3의 caveat 참고). `src/quant/fake_quant.py`.

- **Weight**: per-output-channel symmetric int8. `scale = max(|w|)/127`,
  `wq = round(w/scale)·scale`. weight는 고정값이라 정적으로 한 번 계산.
- **Activation**: per-tensor **asymmetric** int8. calibration 이미지들을 흘려
  `ActObserver`가 min/max를 관측(`observe`) → `freeze()`에서 scale/zero_point
  확정 → 이후 `quantize()`로 매 forward마다 round-clamp.
- `QuantConv2d`가 기존 `nn.Conv2d`를 감싸서 위 두 가지를 다 적용한다.
- `wrap_convs(model, 8, 8)`(`src/quant/quant_model.py`)가 vision 경로의 모든
  `Conv2d`를 `QuantConv2d`로 교체(`DFL`은 고정 가중치라 스킵). text encoder는
  offline 임베딩(`txt_feats`)이라 애초에 conv 형태로 안 걸림 → 자동 제외.
- `calibrate(model, calib_imgs)`가 calibration → freeze → quantized 모드
  전환까지 한 번에 수행.

여기까지가 **naive** baseline이다(추가 최적화 없이 round-to-nearest만).

---

## 4. Baseline PTQ 세 가지 (모두 이 fake-quant 위에 재구현)

### 4.1 AdaRound — `src/quant/adaround.py`

**아이디어**: naive는 `round(w/scale)`로 고정 반올림하지만, 어떤 가중치는
올림이 내림보다 layer 출력 오차를 덜 만든다. AdaRound는 이 "올림/내림" 결정을
학습 가능한 변수로 만든다.

```python
w_int = floor(w/scale) + h(alpha)     # h(alpha) ∈ [0,1], rectified sigmoid
```

`h(alpha) = clamp(sigmoid(alpha)·(ζ-γ) + γ, 0, 1)` (γ=-0.1, ζ=1.1). 학습 중엔
soft(연속값), 확정 시 `h≥0.5 → 1, else 0`으로 hard. 손실:

```
loss = MSE(pred, target) + reg_weight · reg(alpha, beta)
reg  = mean(1 - |2h(alpha)-1|^beta)     # h를 0/1로 미는 정규화, beta는 annealing
```

**순차/누적 오차 반영(중요, 09-06 수정)**: `target`은 항상 순수 FP forward에서
캡처한 이상적 목표. 하지만 `pred`의 입력(`q_in`)은 **quant_model 자신의 현재
상태**(이미 처리된 앞쪽 layer는 hardened, 아직 처리 안 된 뒤쪽은 soft/init)로
다시 forward해서 캡처한다. 최초 구현은 이 `q_in`도 FP에서 캡처해서 썼는데,
그러면 "앞선 layer가 전부 완벽한 FP"라는 잘못된 가정이 되어 AdaRound/BRECQ
원 절차의 핵심(누적 오차 보상)이 빠진다 — 발견 후 수정(§8.1 참고).

**QDrop**은 이 AdaRound 최적화 루프에 확률적 activation drop을 추가한 것뿐:
매 iteration 원소별 확률 `qdrop_prob`(기본 0.5)로만 activation을 양자화하고
나머지는 그대로 둬서("이 layer만 양자화 안 했다면"의 반사실), 다른 레이어의
activation 양자화 노이즈에 더 강건한 반올림을 학습시킨다. `optimize_adaround`의
`qdrop_prob` 인자로 켠다 — 별도 파일 없음.

### 4.2 BRECQ — `src/quant/brecq.py`

AdaRound가 conv 하나씩 독립적으로 출력을 맞추는 layer-wise 방식이라면, BRECQ는
**block**(YOLOv8의 C2f 등 아키텍처 단위) 전체를 통째로 재구성해서 layer 간
상관을 반영한다. 재구성 단위:
- backbone/neck의 top-level 블록: block 단위 joint 재구성. block 안의 모든
  `alpha`를 동시에 최적화, block 내부로 grad가 흐르도록 STE 사용.
- head(WorldDetect): 출력이 복잡(decode 포함)해서 내부 conv(`cv2`/`cv3`)를
  conv 단위로 재구성(=AdaRound와 동일). `cv4`는 애초에 양자화 안 함(측정
  기준점이라 순수 FP로 유지).

AdaRound와 동일한 누적 오차 반영 수정이 적용돼 있다(target=FP 출력,
input=quant_module 현재 상태).

### 4.3 세 baseline의 6-seed 결과

세 baseline 모두 naive 대비 AP가 거의 안 오른다(±0.05 이내, `MASTER_SUMMARY.md`
§5 표 참고) — reconstruction을 더 정교하게 해도 이 W8A8 설정 자체에 개선
여지가 원래 작다는 뜻. 이게 "제안 방법이 왜 필요한가"의 실증적 근거다.

---

## 5. 제안 방법 (Combined / PromptCal) — `src/quant/promptcal.py`

### 5.1 세 번 실패했던 접근과 그로부터 배운 것

최초 설계(`PROMPTCAL_METHOD_SPEC.md`에 기록)는 AdaRound의 `alpha`(반올림
방향)만 최적화해서 held-out margin을 지키려 했다. 세 번 다 실패:
- alpha는 결국 0/1로 **이산화(hard round)** 되어야 하는데, margin 손실은
  연속값 목적함수라 최적화 도중 h가 26~32%에서 어중간하게 멈추고, hard로
  전환하는 순간 다른 모델이 되어버렸다(soft→hard 불연속).
- margin(값)을 맞추는 것과 decision(순위/argmax)을 보존하는 것은 다른
  목적인데, 전자만 최적화했다.

**결론: 최적화 대상을 이산적인 rounding(alpha)에서 연속적인 값으로 옮겨야
한다.** 그래서 현재 방법은 weight rounding은 AdaRound로 이미 확정(hard)해
버려두고, **그 위에 activation quantization의 scale factor에 learnable
continuous multiplier를 하나 더 얹는다.**

### 5.2 `s_mult` — learnable activation scale multiplier (09-08 기준: per-channel 벡터)

**09-16에 다시 per-tensor 스칼라로 되돌아감**(baseline과의 activation
quantization 공정성 + 표준 INT8 엔진 배포 가능성 때문 — 성능 문제 아님,
`PROMPTCAL_CURRENT_MODEL_V2.md` §5.2 참고). 아래는 그 이전(per-channel)
설계가 왜 필요했는지의 역사적 배경으로는 여전히 유효하다.

`AdaRoundQuantConv2d`(`adaround.py`)의 필드:

```python
self.s_mult = nn.Parameter(torch.ones(self.conv.in_channels, device=self.conv.weight.device))
self.use_smult = False                          # True일 때만 적용

def _quantize_smult(self, x):
    mult = self.s_mult.clamp(min=0.1, max=10.0).view(1, -1, 1, 1)
    scale = self.a_obs.scale * mult
    x_s = x / scale
    x_r = x_s + (round(x_s) - x_s).detach()      # round만 STE, scale은 grad 유지
    x_c = clamp(x_r + zp, qmin, qmax)
    return (x_c - zp) * scale
```

`round()`만 straight-through estimator로 처리하고 `scale` 자체는 계산 그래프에
남겨서 `s_mult`로 grad가 흐르게 한다. 즉 "이 레이어의 activation quantization
격자 간격을 살짝 늘리거나 줄인다"를 연속적으로 학습하는 것 — round-to-nearest
자체는 그대로 두고, 그 격자의 눈금 크기만 조정한다.

**원래는 conv당 스칼라 하나(`torch.tensor(1.0)`, 0-dim)였다.** §7의 COCO-80
6-seed 결과는 이 스칼라 버전 기준이다. 09-07~09-08에 LVIS-1203 vocabulary로
일반성을 검증하다가 이 스칼라 설계 자체의 구조적 한계가 드러나서 **입력
채널별 벡터(`[in_channels]`)로 재설계**했다 — 무엇이 문제였고 왜 이렇게
바꿨는지는 §5.7에 전체 경위를 정리한다.

(사소하지만 재현 시 주의할 점: `torch.tensor(1.0)`처럼 0-dim 텐서는 PyTorch가
CUDA 텐서와의 연산에서 자동으로 device를 맞춰주는 예외 취급을 받아
`.to(device)` 없이도 에러가 안 났다. `torch.ones(N)`처럼 1-dim 이상이 되는
순간 이 예외가 사라져서 명시적으로 `device=...`를 안 주면 "Expected all
tensors to be on the same device" 에러가 난다 — 벡터화하면서 실제로 겪은
버그.)

### 5.3 S / H_cal / H_eval — 프롬프트 3분할 (실험 정직성의 핵심)

COCO-80 프롬프트를 seed마다 무작위로 섞어 3분할(`np.random.default_rng(seed)`):

| 그룹 | 개수 | 역할 |
|---|---|---|
| S (seen) | 40 | 최적화 대상(`prompt_idx`)으로 실제 사용 |
| H_cal | 20 | (현재 Combined 구현은 안 씀 — 아래 §5.6 참고) |
| H_eval | 20 | 최적화에 **한 번도 안 씀**. 오직 평가(H_eval_flip, H_eval subset AP)에만 사용 |

핵심 원칙: **측정할 걸 학습에 넣지 않는다.** H_eval은 순수 held-out이라
"완전히 안 본 프롬프트에서도 결정이 보존되는가"를 정직하게 잰다.

### 5.4 margin_loss — S 프롬프트의 경계를 FP와 맞춘다

**09-16에 identity-aware 비교 + intrusion 탐지가 추가됨**(아래 코드는 그
이전의 정렬-비교 버전) — 정렬 후 비교만 하면 top-1/top-2가 값을 맞바꾸는
flip이 일어나도 margin 패턴이 같으면 loss가 0이 되는 blind spot이 있었다.
확정 설계의 정확한 코드는 `PROMPTCAL_CURRENT_MODEL_V2.md` §5.3(a) 참고.

```python
def margin_loss(sim_q, sim_fp, k=5, boundary_w=3.0):
    fp_top, _ = sim_fp.topk(k+1, dim=-1)          # confident anchor × S 프롬프트만
    q_top, _  = sim_q.topk(k+1, dim=-1)
    fp_m = fp_top[:, :-1] - fp_top[:, 1:]         # 인접 pairwise margin [A, k]
    q_m  = q_top[:, :-1]  - q_top[:, 1:]
    w = [1,1,1,1,3]                                # 마지막(top-k 경계)에 가중
    return mean((q_m - fp_m)^2 · w)
```

confident anchor(FP 기준 `sigmoid(max) > 0.25`)에서만, S 프롬프트 열
(`sim[:, pidx]`)만 뽑아 계산한다. "1등과 500등의 차이"가 아니라 **top-k 경계의
순위 간격**을 FP와 맞추려는 것 — 09_phase1(RankSafe) 보고서의 통찰("순위 k와
k+1 사이 역전이 실제 detection에 큰 영향")을 반영한 설계.

### 5.5 neighbor loss(asymmetric hinge) — collateral shift 억제

**왜 필요한가.** `s_mult`는 **class-agnostic**하다 — 하나의 conv를 지나는
feature 전체에 곱해지는 스칼라라서, region feature와 80개 text embedding의
내적(=79개 다른 class의 점수) 전부에 똑같이 영향을 준다. S의 margin만 보고
`s_mult`를 조정해도 이 조정은 **S가 아닌 class(H_cal, H_eval 포함)의 절대
점수에도 그대로 새어나간다(collateral shift)**. 특히 text embedding상 S 클래스와
가까운 이웃 class일수록 이 흔들림을 더 크게 받는다(09-05 §40에서 확인: H_eval이
S와 가까운 seed일수록 held-out 성능이 더 나쁨).

**대응.** S의 각 class에 대해 text embedding 최근접 이웃(non-S) `neighbor_k`개를
뽑아(`text_neighbor_order` — FP text encoder 임베딩의 코사인 유사도 내림차순,
GT/held-out identity 불필요), 그 이웃들의 절대 유사도가 FP 대비 흔들리지 않게
붙잡는 항을 추가한다:

```python
diff = sim_q[:, neighbor_cols] - sim_fp[:, neighbor_cols]
nl = relu(diff)^2.mean()            # asymmetric: "커지는" 방향만 벌점
loss = margin_loss + neighbor_weight · nl
```

**symmetric(MSE) 대신 asymmetric(one-sided hinge)을 쓰는 이유**: 이웃의
유사도가 FP보다 **낮아지는** 방향은 원래 anchor의 target이 오히려 더 안전해지는
방향이라 벌점을 줄 이유가 없다. **높아지는** 방향(경쟁자가 강해져서 실제로
top-1을 빼앗을 위험이 커짐)만 억제한다. 41번 실험에서 symmetric MSE는
`neighbor_weight`를 스윕했을 때 seed마다 반응이 뒤바뀌는 불안정한 패턴을
보였다 — 이웃 쪽으로의 흔들림이 우연히 유익한 seed에서도 무차별적으로
억제해버렸기 때문으로 추정. asymmetric으로 바꾼 뒤 6-seed AP가 안정적으로
개선됐다(§6 참고).

### 5.6 전체 빌드 파이프라인 (`scripts/45_baseline_compare.py`의 `build()`)

```
1. wrap_convs(model, 8, 8) → calibrate(calib_imgs)      # naive 초기화
2. convert_to_adaround(model)                           # QuantConv2d → AdaRoundQuantConv2d
3. optimize_adaround(model, fp_model, calib, iters=1000) # weight rounding 확정(hard)
4. optimize_promptcal_scale_neighbor(model, fp_model, calib, pidx=S,
                                     iters=1500, k=5, neighbor_k=5,
                                     neighbor_weight=1.0, asymmetric=True,
                                     scale_reg_weight=20.0)   # 09-08 기준 기본값
   → 이 단계에서 rounding(alpha)은 완전히 고정(requires_grad=False),
     s_mult만 학습(use_smult=True)
```

즉 **"Combined" = AdaRound(weight rounding) + asymmetric neighbor-preserving
continuous activation scale(s_mult, per-channel) + (s_mult-1)² 정규화**. 두
최적화가 서로 다른 파라미터 집합(alpha vs s_mult)을 순차적으로 건드리기
때문에 서로 간섭하지 않는다. `scale_reg_weight`가 왜 필요해졌는지는 §5.7.

`optimize_promptcal_scale_neighbor_utility`(같은 파일)는 여기에 논문 §4.3
utility constraint(threshold-crossing + box consistency, `semantic_calib.py`의
`utility_refinement_terms`)를 2단계로 추가한 변형이다 — 검증 결과 기본
가중치에서는 중립적(뚜렷한 추가 이득 없음, `scripts/44_utility_ap_check.py`).
현재 6-seed 최종 비교(`45_baseline_compare.py`)는 utility 없는
`optimize_promptcal_scale_neighbor`만 사용한다.

### 5.7 LVIS 일반성 문제와 재설계 (09-07~09-08) — s_mult가 스칼라에서 벡터로 바뀐 이유

§7의 6-seed 결과는 전부 **COCO-80 vocabulary**(calibration에 쓴 40개 S
프롬프트가 최종 배포 vocabulary와 같은 80개 중 일부)에서 나온 것이다. 논문의
일반성 주장을 위해 훨씬 촘촘하고 큰 vocabulary(LVIS-1203, 논문이 "80개짜리
좁은 세계에서만 통하는 트릭이 아니다"를 보여야 하는 지점)에 그대로 배포했을
때도 이점이 유지되는지 검증하다가, 원래 설계(conv당 스칼라 s_mult)의
구조적 한계가 드러났다.

**5.7.1 무엇이 문제였나.** COCO-80에서 학습한 Combined 모델을 재학습 없이
LVIS-1203 vocabulary로 바꿔서 배포하면, 실제 LVIS AP가 **naive보다도
낮았다**(naive/AdaRound/QDrop/BRECQ/Combined 5개 조건 중 최악). flip이
늘어나는 게 "무해한 재배치"가 아니라 진짜 검출 품질 저하라는 뜻이었다.

**5.7.2 원인 진단.** §5.5에서 이미 짚었듯 `s_mult`는 **class-agnostic**하다
— 원래(스칼라) 버전은 conv 하나당 숫자 하나뿐이라, "S(COCO 40개 프롬프트)의
neighbor만 조준해서 억제한다"는 neighbor loss의 의도 자체가 **구조적으로
불가능**했다: 그 컨볼루션을 지나는 activation 전체에 스칼라 하나가 곱해지니,
"S의 이웃 ~35개 컬럼이 안 커지게" 만족시키는 유일한 방법은 그 conv의
activation scale 전체를 획일적으로 낮추는 것뿐이고, 그러면 calibration에서
아예 존재도 몰랐던 LVIS의 나머지 1100여개 class까지 무차별적으로 위축된다.
실제로 학습 후 s_mult 평균이 항상 1.0 미만(0.97~0.99)으로 수렴했다 — "전역
하향 편향"이 새어나가고 있었다는 직접 증거.

**5.7.3 인과관계 확정 (ablation).** `scripts/52_smult_ablation.py`: 이미
학습된 Combined 모델의 `s_mult`를 학습 후에 강제로 1.0으로 되돌리면
(`s_mult.fill_(1.0)`), 그 모델의 AP가 **AdaRound와 소수점까지 일치**했다
(`combined_smult1` AP == AdaRound AP). 즉 Combined와 AdaRound의 유일한 차이가
`s_mult`이고, LVIS 손상의 원인이 정확히 `s_mult`의 하향 편향이라는 걸 직접
증명한 것.

**5.7.4 "재학습 없는 이식" 탓이 아니라 설계 자체의 한계임을 확인.** 혹시
"COCO에서 학습해서 LVIS로 재학습 없이 옮긴" 것 자체가 문제였을 수도 있으니,
LVIS-1203을 **처음부터 native로 calibration/학습**한 대조 실험도 돌렸다.
결과: COCO-80 native(s_mult 평균 0.977) / COCO→LVIS 이식(0.988) /
LVIS-native(0.971) **세 시나리오 전부**에서 s_mult가 1.0 미만으로
수렴했다 — 재학습 여부와 무관하게 conv당 스칼라라는 파라미터화 자체의
한계로 확정.

**5.7.5 1차 대응: 정규화만으로는 부족.** 가장 싼 대응인 `(s_mult-1)²`
정규화(scale_reg_weight, §5.6 코드 참고)를 스칼라 버전에 추가하고
`scale_reg_weight=200`으로 6-seed 검증했더니, LVIS AP는 naive를 근소하게만
넘었지만(AdaRound/BRECQ에는 못 미침) 그 대가로 COCO-80에서 Combined의 핵심
결과였던 UPIR(0.142%, 5개 중 최선)이 0.225%로 baseline 수준까지 후퇴했다 —
"핵심 주장을 포기하고 LVIS에서 그저 그런 성적"이라는 나쁜 트레이드오프.
정규화 하나로는 class-agnostic이라는 근본 원인을 못 없앤다는 뜻.

**5.7.6 2차 대응(현재 채택): per-channel 벡터화 + 가벼운 정규화.** §5.2의
`s_mult`를 conv당 스칼라에서 **입력 채널별 벡터**(`[in_channels]`)로
바꿨다 — 채널마다 다른 값을 학습할 자유도를 주면, "S의 이웃만 죽이고 나머지는
안 건드린다"는 원래 의도를 최소한 부분적으로는 채널 단위로 분산시켜 달성할
여지가 생긴다. 벡터화만으로는 자유도가 커진 만큼 오히려 calibration set에
과적합하는 경향이 보였는데(단독으로는 여전히 LVIS baseline을 다 못 이김),
가벼운 정규화(`scale_reg_weight=20`, 스칼라 버전에 썼던 200보다 훨씬 작은
값 — 벡터는 채널마다 다른 값을 가지므로 훨씬 약한 벌점으로도 충분)를 같이
쓰니 지금까지 나온 Combined 변형 중 가장 균형 잡힌 프로파일이 나왔다:
COCO-80 AP 우위를 거의 그대로 유지하면서, LVIS AP가 처음으로 baseline들과
동급(naive와 사실상 동률, AdaRound/QDrop/BRECQ보다 우위)이 됐다.

**5.7.7 "논문 방식" 데이터로 최종 검증 (진행 중, 09-08).** calibration을
val2017 슬라이스(32~256장, 평가 데이터와 겹침)에서 **train2017 256장**으로,
LVIS 평가를 임시 484장 부분집합에서 **공식 `lvis_v1_minival.json`
(4809장, ultralytics 공식 배포 — "COCO val2017 ∩ LVIS val"과 정확히 일치함을
확인)** 으로 바꾼, 논문 실험 관행에 더 가까운 설정으로 재검증했다
(`scripts/58_full_baseline_official_data.py`). 3-seed 결과: COCO-80 AP가
naive 대비 **+2.54**(역대 최대 격차), masked H_eval_flip에서 **처음으로
5개 조건 중 1위**(이전까지 이 지표는 줄곧 Combined의 최대 약점이었음),
LVIS_flip·LVIS_lost도 5개 중 1위. LVIS AP만 naive와 근소하게 동률(뚜렷한
승리는 아님). 상세 표는 `MASTER_SUMMARY.md` §8 국면 I,
`PromptCal_PTQ_progress_2026-09-08.md` §5 참고.

**현재 상태(09-08, 진행 중)**: 이 결과에 쓴 `scale_reg_weight=20`은 예전
데이터 스케일(calib=32)에서 고른 값을 그대로 가져온 것이라, 이 새 데이터
스케일(calib=256)에서 재스윕(`scripts/59_rw_sweep_official_data.py`,
rw∈{10,20,30,50})과 6-seed 확장 검증이 진행 중.

---

## 6. 평가지표 — 무엇을, 왜 재는가 (`scripts/45_baseline_compare.py`)

### 6.1 AP (전체 / S subset / H_eval subset)

`ultralytics` model.val()의 pycocotools 기반 mAP50-95. `per_class` AP를
class index로 뽑아 S/H_eval 그룹으로 평균 내면 "학습에 쓴 프롬프트만" vs
"한 번도 안 쓴 프롬프트만"의 AP를 분리해서 볼 수 있다.

### 6.2 flip 지표 두 종류 — 반드시 구분해야 하는 이유

둘 다 "FP32와 양자화 모델의 top-1 결정이 같은가"를 재지만, **가정하는 배포
시나리오가 다르다**.

- **Top1_flip(표준, `standard_flip`)**: masking 없음. FP confident anchor에서
  raw top-1(전체 80class 중)이 quant와 같은지. **AP와 동일한 조건** — "정답
  프롬프트가 실제로 vocabulary 안에 있는" 정상 배포 상황을 그대로 잰다.
- **H_eval_flip(masked, `group_flip`, 우리가 만든 진단 지표)**: H_eval 20개
  컬럼을 **가린 뒤**, FP가 원래 H_eval class로 판정했던 confident anchor에서
  "H_eval을 뺀 나머지(S+H_cal, 60class) 중 누가 1등이 되는가"가 FP/quant 간에
  같은지. 이건 **"이 20개 class를 아예 모르는/안 쓰는 다른 사용자"라는
  반사실적(counterfactual) 시나리오**를 재는 것 — 실제 배포에서 그 20개
  프롬프트가 여전히 vocabulary에 있다면 이 시나리오 자체가 일어나지 않는다.

이 구분이 왜 중요한가: Combined는 masked H_eval_flip에서는 BRECQ보다 나쁘지만
(neighbor preservation이 S만 지키고 H_cal/H_eval은 전혀 보호하지 않으니 당연),
표준 Top1_flip에서는 BRECQ와 동률이다(§7의 6-seed 결과) — **같은 "flip"이라는
이름 아래 서로 다른 걸 재고 있었다는 게 이 프로젝트의 중요한 방법론적 발견**.

### 6.3 GT 기반 지표 (진짜 COCO annotation 사용)

여기까지의 모든 지표는 FP32 자신의 예측을 pseudo-GT로 쓴다. GT 지표는 다르다:
실제 `instances_val2017.json`의 정답 box/class를 쓴다.

**GT 앵커 매칭** (`build_gt_targets`): 공식 TaskAlignedAssigner를 그대로
구현하지 않고, 실용적 근사를 쓴다 — GT box를 letterbox 변환으로 640 좌표계에
옮긴 뒤, 3개 FPN level(stride 8/16/32)에서 그리드 셀 **중심이 box 안에 드는**
후보 anchor를 모으고, 그중 **FP32가 그 GT의 정답 class에 가장 confident한
anchor 하나**를 대표로 선택한다.

이렇게 GT-anchor 쌍이 정해지면:

- **GT_MRR**: 그 anchor에서 정답 class의 순위 역수(quant 자신의 랭킹 기준) 평균.
- **GT_R@1**: 정답 class가 quant에서 1등인 비율.
- **lost / gained**: FP에서 1등이었다가 quant에서 잃은 개수 / 그 반대(FP에서
  1등이 아니었는데 quant에서 얻은 개수).
- **UPIR**(Unseen-Prompt Intrusion Rate): FP에서 1등이었던 GT-anchor 중
  (정답 class가 H_eval이 아닌 경우만) quant의 1등이 **H_eval class로 바뀐**
  비율. "calibration에서 아예 안 본 프롬프트가 정답 위로 침입"하는 정도를
  직접 잰다 — 논문의 핵심 동기와 가장 직접 대응하는 지표.

이 지표 세트는 새로 발명한 게 아니라, 이전 팀원의 Phase 1(RankSafe) 보고서
(`09_phase1_final_report_ko_v2.pdf` §1.3/§18)가 표준으로 썼던 것을 그대로
가져온 것 — 그 보고서는 "정렬 개선만으로 성공을 선언하지 말고 AP와 GT 기반
semantic error가 함께 개선되는지 봐야 한다"고 명시적으로 경고한 바 있다.

### 6.4 비용 지표

- **calib 시간**: `build()` 호출 전체를 wall-clock으로 측정(naive는 calibrate만,
  나머지는 AdaRound/QDrop/BRECQ/Combined 최적화 루프까지 포함).
- **이론적 모델 크기**: 양자화 대상 conv weight 총 원소 수 × 1byte(8bit) → MiB.
  실제 INT8 엔진의 파일 크기가 아니라 "weight를 8bit로 저장하면"의 이론값.

---

## 7. 6-seed 최종 결과 요약

자세한 표와 판정은 `MASTER_SUMMARY.md` §6, 원본 로그는
`results_v1/diag/45_metrics_seed{0,1,2,4,5,7}_full.txt`, 서술은
`PromptCal_PTQ_progress_2026-09-06.md` §18 참고. 한 줄 요약:

> Combined는 AP·표준 Top1_flip·UPIR에서 QDrop/BRECQ를 포함한 모든 baseline을
> 이기거나 동등하다. GT_MRR/R@1은 BRECQ와 사실상 동률. masked H_eval_flip(우리가
> 만든 반사실적 진단)에서만 BRECQ보다 못하다 — 이건 실제 배포 조건과 다른
> 시나리오를 재는 지표라서 발생하는, 메커니즘이 밝혀진 한계다.

(이 §7 결과는 COCO-80 vocabulary·스칼라 s_mult 기준이다. LVIS-1203로
일반화했을 때 이 스칼라 설계가 어떻게 무너지고 어떻게 재설계했는지는 §5.7.)

---

## 8. 알아둬야 할 함정 / 교훈 (재현하거나 확장할 때 참고)

### 8.1 AdaRound/BRECQ의 "누적 오차 미반영" 버그

최초 구현은 layer/block의 재구성 **입력**을 `fp_module`에서만 캡처했다 — 즉
모든 layer가 "앞쪽이 전부 완벽한 FP"라고 가정하고 독립적으로 재구성됐고, 앞선
layer들의 실제 양자화 오차가 뒤쪽 layer 학습에 전혀 반영되지 않았다(AdaRound/
BRECQ 원 절차의 핵심인 누적 오차 보상이 빠짐). target(이상적 목표)은 FP
기준으로 두되, pred의 입력만 quant_module 자신의 현재 상태에서 별도로 다시
캡처하도록 수정(§4.1). **수정 후에도 baseline들의 AP는 거의 안 변했다** —
버그가 baseline이 약했던 원인이 아니라, W8A8 설정 자체가 원래 reconstruction
개선 여지가 작았던 것으로 해석하는 게 정확하다.

### 8.2 cuDNN 비결정성

동일한 설정인데도 실행마다 결과가 다르게 나오는 문제가 있었다 — cuDNN 알고리즘
선택의 비결정성 때문. `torch.manual_seed`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`로 해결(모든 스크립트 상단에 고정 적용).

### 8.3 fake-quant는 실제 배포 성능이 아니다

지금까지의 모든 latency/모델크기 수치는 clamp/round/dequant로 시뮬레이션한
연구용 fake-quant 기준이지 실제 INT8 엔진(TFLite/TensorRT 등)의 성능이 아니다.
Phase 1 보고서도 이 점을 명시적으로 경고한다(부록 B 해석 원칙 #4: "실제 engine을
만들지 않은 model-size/latency 추정은 deployment 주장으로 사용하지 않는다").
실배포 성능을 주장하려면 별도의 엔진 export 작업이 필요 — 현재 범위 밖.

### 8.4 GPU 인덱스 함정

CUDA의 기본 디바이스 열거 순서("가장 빠른 것부터")는 `nvidia-smi`의 PCI-bus
순서와 다를 수 있다(이 서버는 L40S 3개가 CUDA에서 먼저 잡히고, RTX4000 Ada
5개가 그 뒤 — `nvidia-smi`는 반대로 RTX4000이 index 0). 항상
`CUDA_DEVICE_ORDER=PCI_BUS_ID`를 `CUDA_VISIBLE_DEVICES`와 함께 설정해야
의도한 물리 GPU에 job이 간다.
