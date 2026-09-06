# PromptCal-PTQ 연구 진행 정리 — 2026-09-06

## 0. 문서 목적

[PromptCal_PTQ_progress_2026-09-05.md](PromptCal_PTQ_progress_2026-09-05.md)에서 direction-C(연속
activation scale)가 "유망하지만 노이즈가 심함"으로 끝났었다. 이 문서는 그 이후
(36~43번 스크립트, 논문 초안 `논문.txt` 확인 포함) 진행 내용을 정리한다.

핵심 줄거리: **flip 지표만으로는 승부가 안 갈렸지만, 실제 AP로 보면 지금까지 시도한
방법 중 처음으로 6개 seed 전부에서 AdaRound를 이기는 결과가 나왔다.**

---

## 1. 논문 초안 확인 (`논문.txt`)

사용자가 논문 초안 위치를 알려줘서 확인함. §4(Methodology)가 지금까지의 실험과
정확히 대응된다:

| 논문 §4 | 코드 |
|---|---|
| Region Filtering | `semantic_calib.py`의 confidence/margin 기반 reliable anchor 선택 |
| Prompt Selection (teacher-derived, GT 불필요, semantically competitive prompts) | `text_neighbor_order` 기반 근접 이웃 선택(30번, 41번) |
| Semantic Objective | margin_loss + neighbor preservation |
| Utility Constraints | threshold-crossing, box consistency (`semantic_calib.py`, 아직 재통합 안 함) |
| Optimization ("Scale/clipping/rounding parameter만 최적화, weight 고정") | AdaRound alpha(rounding) 또는 s_mult(scale) |

즉 30번(`semantic_calib.py`)은 논문 §4 전체의 첫 구현 시도였고 실패했었다(-16.3%).
41번(neighbor preservation)은 같은 §4.2 Prompt Selection 자리의 재시도이며, 이번엔
"경쟁 프롬프트를 상대 margin에만 쓴다"(30번) 대신 "경쟁 프롬프트의 절대 유사도
드리프트를 직접 억제한다"는 다른 메커니즘을 쓴다.

---

## 2. 39번 재확인: 결정성이 정말로 분산을 지웠는가

09-05 문서 §13에서 결정성(torch.manual_seed+cudnn.deterministic)을 도입했지만
"같은 seed를 여러 번 돌려도 다 같은 값만 나온다"는 문제가 있었다(진짜 랜덤성을
학습 루프가 소비하지 않아서). `scripts/37_variance_check.py`를 고쳐서 trial마다
`torch_seed + trial`로 서로 다르지만 재현 가능한 시드를 쓰도록 했다.

결과([39_seed0/1/2_seeded_full.txt](results_v1/diag/39_seed0_seeded_full.txt)):
seed 0/1 모두 std=0.00(6개 trial이 거의 완전히 동일), seed 2도 마찬가지. **즉 결정성
자체는 완벽하게 작동한다** — 남은 변동은 "seed(prompt split)가 다르면 결과가
다르다"는 것뿐이고, 그건 노이즈가 아니라 실제 신호다.

---

## 3. 40번: 왜 seed마다 유불리가 갈리는가

학습 없이 FP 모델만으로 각 seed의 H_eval 구성을 분석. 39번(symmetric neighbor
없는 순수 scale) 기준:

```
seed |  결과 | avg_sim(H_eval->S) | n_confident
   0 |  악화 |              0.8449 |        3819
   1 |  악화 |              0.8428 |        4080
   2 |  개선 |              0.8358 |        3517
```

H_eval이 S와 text embedding상 가까울수록(유사도 높을수록) 악화되는 패턴 관찰.
해석: S의 margin을 맞추려는 s_mult 조정이 class-agnostic해서, S와 가까운 class로
"collateral shift"가 전파된다.

---

## 4. 41번: Neighbor Preservation (symmetric MSE) — 절반의 성공

`src/quant/promptcal.py`에 `optimize_promptcal_scale_neighbor` 추가: S 각 class의
text-embedding 최근접 이웃(비-S) `neighbor_k`개의 유사도가 FP 대비 흔들리지 않도록
MSE로 붙잡아둔다.

seed 0/1/2, neighbor_weight=1.0 결과:
```
seed |   AdaRound |  Scale-only | +neighbor(symmetric)
   0 |     7.59%  |      7.36%  |   8.12% (악화!)
   1 |     8.92%  |      9.00%  |   8.26% (개선)
   2 |     9.64%  |      9.10%  |   8.90% (개선)
```

seed 0에서 오히려 악화 — symmetric MSE가 "우연히 유익했던 흔들림"까지 무차별
억제했기 때문으로 추정. neighbor_weight를 0.3/0.5로 낮춰도 깔끔하게 해결 안 됨
(seed마다 반응 방향이 달라 다이얼로 조절되지 않음). 이 과정에서 **GPU 6과 GPU7이
같은 모델(RTX 4000 Ada)인데도 완전히 동일한 설정에서 미세하게 다른 결과**를
낸다는 것도 확인(결정성은 "같은 물리 GPU 안에서"만 보장).

---

## 5. 41번: Asymmetric hinge — seed0 문제 해결, 그러나 완벽하진 않음

`asymmetric=True` 옵션 추가: `relu(sim_q - sim_fp)^2`만 사용(경쟁자가 FP보다
강해지는 방향만 억제, 약해지는 방향은 자유). semantic_calib.py의 threshold-crossing
hinge와 같은 스타일.

seed 0/1/2 (neighbor_weight=1.0, GPU7 고정):
```
seed |   AdaRound |  Scale-only | symmetric | asymmetric
   0 |     7.59%  |      7.36%  |   8.12%   |   7.49% (AdaRound도 이김)
   1 |     8.92%  |      9.00%  |   8.26%   |   9.02% (근소 패, +0.10pp)
   2 |     9.64%  |      9.10%  |   8.90%   |   8.93% (승)
```

seed 0~9까지 10개 seed로 확장(GPU 4~7 병렬):
```
6승 4패, 평균 격차 -0.23pp, std 0.69pp (통계적으로 확정하기엔 이름)
seed 7이 최악(+1.21pp 악화)
```

42번(seed별 분석, gap 포함) 재확인: **avg_sim(H_eval->S)와 gap의 상관계수가
symmetric일 때보다 크게 약해짐(+0.16)** — 원래 진단한 메커니즘(collateral shift)이
asymmetric으로 실제로 완화됐다는 뜻. 다만 seed7처럼 아직 설명 안 되는 변동이 남음.
n_confident와의 상관은 -0.43(약함, seed 6/9의 "person" 이상치가 왜곡 요인).

---

## 6. 43번: AP 검증 — 지금까지 가장 깨끗한 결과

flip만으로는 승부가 안 갈리자, 논문 §3.4의 원칙("rank 보존이 AP를 보장하지 않음")을
따라 실제 pycocotools AP를 확인(`scripts/43_ap_check.py`, `model.val()` 재사용).
비교축은 여전히 AdaRound(reconstruction baseline) vs Combined(AdaRound weight +
asymmetric neighbor scale).

전체 80-class mAP50-95 (AdaRound=34.96, FP32=36.80 고정):
```
seed |  Combined | 격차
   0 |    36.36  | +1.40
   1 |    36.47  | +1.51
   2 |    36.47  | +1.51
   4 |    36.39  | +1.43
   5 |    36.54  | +1.58
   7 |    36.51  | +1.55
```
6개 seed 평균 +1.50, 범위 +1.40~+1.58 — 매우 안정적. FP32에 근접.

H_eval(calibration에 전혀 안 쓴 20개 class) subset mAP50-95:
```
seed | flip 결과 | AdaRound H_eval | Combined H_eval | 격차
   0 |    승    |     32.43       |     33.50       | +1.07
   1 |    패    |     30.43       |     31.81       | +1.38
   2 |    승    |     35.37       |     37.14       | +1.77
   4 |    패    |     39.87       |     41.27       | +1.40
   5 |    승    |     40.02       |     41.87       | +1.85
   7 |  패(최악) |     34.45       |     36.28       | +1.83
```
**6개 seed 전부 Combined 승리(6/6), 평균 +1.55.** flip에서 최악이었던 seed 7이
AP에서는 오히려 개선폭이 가장 큰 축에 속함 — flip과 AP가 방향조차 다를 수 있다는
논문 §3.4의 경고를 정확히 보여주는 사례.

**결론**: flip은 seed-dependent로 불안정했지만(6승 4패, 통계적으로 약함), AP는
6전 6승으로 훨씬 깨끗하고 일관됨. 논문 핵심 주장("제안 방법이 AdaRound보다
held-out vocabulary에서 일관되게 우수")을 현재까지 중 가장 설득력 있게 뒷받침.

---

## 7. GPU 인덱싱 사고 및 수정

43번 seed 0~7 병렬 실행 중, `CUDA_VISIBLE_DEVICES=0`으로 지정한 작업이 실제로는
nvidia-smi 기준 GPU 1(다른 사용자가 88% 사용 중인 L40S)에 올라간 것을 사용자가
발견. 원인: **CUDA의 기본 GPU 열거 순서가 nvidia-smi의 PCI 버스 순서와 다르다**
(CUDA는 L40S 3개를 0/1/2로 먼저 놓고 RTX4000 Ada 5개를 3/4/5/6/7로 놓는데,
nvidia-smi는 PCI 순서대로 RTX4000이 0, L40S가 1~3, RTX4000이 4~7).

```python
torch.cuda.get_device_properties(i).uuid  # 로 확인
```

즉시 해당 프로세스를 kill하고, 나머지(seed 1/4/5/7, CUDA index 4/5/6/7)는 우연히
identity mapping이라 문제없었음을 UUID로 확인. 이후 모든 실행에
`CUDA_DEVICE_ORDER=PCI_BUS_ID`를 `CUDA_VISIBLE_DEVICES`와 함께 지정하도록 변경—
이러면 CUDA 인덱스가 nvidia-smi 인덱스와 항상 일치한다.

**주의**: 09-05 문서 및 이 문서 초반 실험들 중 일부(29~38 등)는 이 수정 이전에
`CUDA_VISIBLE_DEVICES=1` 등을 썼는데, 그것들이 정확히 어떤 물리 GPU였는지는
재확인하지 않았다. 다만 핵심 비교(35_deterministic, 39, 41 asymmetric, 43)는
모두 GPU 6/7(또는 correctly-mapped 4/5)을 명시적으로 고정해서 진행했으므로
결론 자체에는 영향 없음.

---

## 8. 09-06 결론 및 다음 단계 후보

```
39: 결정성 자체는 완벽 작동(seed 재현), 남은 변동은 진짜 seed-dependent 신호
40: collateral shift 메커니즘 확인(H_eval-S 임베딩 유사도 상관)
41: asymmetric neighbor preservation으로 메커니즘은 완화됐지만 flip 기준
    승률은 아직 불안정(6/10)
42: 상관 재확인 -> 메커니즘 완화는 확인, 잔여 변동 원인은 미상
43: AP 기준으로는 6/6 완승, 평균 +1.5(전체) / +1.55(H_eval subset) --
    지금까지 가장 강력한 증거
```

다음 후보(미착수):
1. AP를 더 많은 seed(예: 10개)로 확장해서 통계적 안정성 재확인.
2. 논문 §Evaluation Protocol의 다른 지표(GT rank, boundary inversion)도 확인.
3. 다른 detector(OWLv2)로 방법이 이식되는지 확인(논문 §Generalization
   "Cross-Architecture Generalization").
4. 논문 §4.3 Utility-Constrained Refinement(threshold-crossing, box consistency)를
   재통합해서 추가 개선 여지가 있는지 확인.

---

## 9. 44번: Utility-Constrained Refinement 재통합 — 중립적

논문 §4.3(threshold-crossing + box consistency, `semantic_calib.py`의
`utility_refinement_terms`)을 `optimize_promptcal_scale_neighbor_utility`로
Combined 위에 재통합해서(`scripts/44_utility_ap_check.py`) 6 seed(0,1,2,4,5,7)에서
확인했다.

```text
Combined vs Combined+Utility 격차 (6 seed)
  전체 mAP  평균 +0.05 (최대 편차 ±0.17)
  H_eval mAP 평균 +0.06 (최대 편차 ±0.17)
```

기본 가중치(thresh_w=1.0, box_w=0.5)에서는 **사실상 중립적** — 있으나 없으나 거의
같다. neighbor preservation(semantic calibration)이 이미 개선 여지의 대부분을
가져간 것으로 보이며, 이 세팅에서 utility constraint를 추가로 튜닝해도 큰 이득은
기대하기 어려워 보인다(가중치를 세게 올려서 재확인하는 것도 가능하지만 후순위로 미룸).

---

## 10. Baseline 비교 착수: QDrop, BRECQ

지금까지 AdaRound 하나만 baseline이었다. 논문 Related Work에 언급된 QDrop
(activation drop으로 강건한 rounding), BRECQ(block 단위 joint reconstruction)와도
비교해야 baseline 비교가 완성된다. 코드베이스에 이미 구현이 있었다
(`src/quant/brecq.py`, `src/quant/adaround.py`의 `qdrop_prob`).

`scripts/45_baseline_compare.py`(seed 2)로 첫 확인:

```text
전체 mAP50-95: FP32=36.80 naive=35.06 AdaRound=34.96 QDrop=35.06 BRECQ=34.98 Combined=36.47
```

**QDrop, BRECQ가 AdaRound보다 더 정교한 방법인데도 naive와 거의 구분이 안 됐다**
(34.96~35.06 좁은 범위). Combined만 확실히 앞섰다(+1.4 이상).

---

## 11. 발견: AdaRound/BRECQ의 "누적 오차 미반영" 버그

QDrop/BRECQ가 이론상 기대만큼 효과가 없는 게 이상해서 구현을 재검토했다.
`optimize_adaround`(및 `optimize_brecq`)는 **모든 layer/block의 재구성 입력을
fp_module에서만 캡처**하고 있었다 — 즉 10번째 layer를 최적화할 때도 "앞의 9개
layer가 이미 양자화돼서 생긴 오차가 섞인 진짜 입력"이 아니라 "앞이 전부 완벽한
FP였다고 가정한 입력"을 썼다. 원래 AdaRound/BRECQ 절차의 핵심(순서대로 처리하며
이미 처리된 앞쪽 layer의 실제 양자화 오차를 뒤쪽 layer가 보상하도록 학습)이
빠져 있었던 것.

**수정**: `optimize_adaround`, `optimize_brecq` 둘 다 target(이상적 목표)은
그대로 FP 기준으로 캡처하되, pred의 입력(`q_in`)은 quant_module 자체의 현재
상태(이미 처리된 앞쪽 layer는 hardened, 아직 처리 안 된 뒤쪽은 soft/init)로
별도로 다시 캡처하도록 고쳤다(asymmetric reconstruction: target=FP, input=실제
누적 오차 반영). QDrop의 drop도 이제 q_in 기준으로 동작하도록 함께 수정.

---

## 12. 수정 후 재검증: 버그는 진짜였지만 "숨겨진 개선"은 없었다

수정 후 seed 2 재실행(`45_fixed_seed2_full.txt`):

```text
                수정 전                수정 후
naive         | 35.06                | 35.06
AdaRound      | 34.96                | 35.05
QDrop         | 35.06                | 35.08
BRECQ         | 34.98                | 35.01
Combined      | 36.47                | 36.38
```

**AdaRound/QDrop/BRECQ는 수정 후에도 naive와 여전히 거의 구분이 안 된다**(오히려
더 좁게 뭉침). 즉 "누적 오차 미반영이 이 baseline들을 실제보다 약하게 만들었다"는
가설은 기각됐다 — 버그는 실재했고 고쳤지만, 그게 원인이 아니었다.

더 설득력 있는 설명: **W8A8은 naive quantization도 이미 FP32에 꽤 가까운("쉬운")
세팅이라, reconstruction 기반 방법들이 개선할 여지 자체가 거의 없다.** 이 방법들은
보통 INT4 이하 저비트에서 진가를 발휘한다. Combined는 수정 후에도 거의 그대로
유지(-0.09, noise 범위) — 예전 버그가 Combined 결과를 부풀린 것도 아니었다.

결과적으로 이건 논문 주장을 더 깔끔하게 만든다: **"reconstruction을 아무리
정교하게(버그를 고쳐도) 해도 이 세팅에서는 다 같은 지점(~35.0~35.1)에 갇히는데,
semantic-aware 방법만 그 천장을 뚫는다."**

---

## 13. 6-seed baseline 비교 (AP): Combined 6/6 완승

수정판으로 seed 0,1,2,4,5,7 전체를 재실행(`45_fixed_seed{N}_full.txt`).

H_eval subset mAP50-95:

```text
seed |  naive | AdaRound |  QDrop |  BRECQ | Combined | Combined 격차(최고 baseline 대비)
   0 |  32.61 |    32.66 |  32.68 |  32.39 |    33.64 | +0.96
   1 |  30.47 |    30.47 |  30.52 |  30.41 |    31.65 | +1.13
   2 |  35.77 |    35.70 |  35.65 |  35.61 |    37.23 | +1.46
   4 |  40.20 |    39.98 |  40.02 |  39.96 |    41.33 | +1.13
   5 |  40.21 |    40.23 |  40.11 |  40.20 |    41.72 | +1.49
   7 |  34.54 |    34.35 |  34.49 |  34.32 |    36.32 | +1.78
평균 |  35.63 |    35.57 |  35.58 |  35.48 |    36.98 | +1.35
```

**6/6 전승, 평균 +1.35.** naive/AdaRound/QDrop/BRECQ는 seed 안에서 항상 0.2~0.4점
이내로 서로 구분이 안 되는데, Combined는 그 넷 중 최고보다도 항상 최소 +0.96점
이상 앞선다. QDrop, BRECQ까지 포함해서 baseline 비교가 통계적으로 탄탄해졌다.

---

## 14. flip을 다시 추가했더니 정반대 결과가 나왔다

AP만 재고 flip을 안 잰 것을 사용자가 지적함 — `scripts/45_baseline_compare.py`에
`group_flip`(H_eval flip)을 다시 추가해서 같은 6 seed에서 재실행.

```text
H_eval flip (%)
seed |  naive | AdaRound |  QDrop |  BRECQ | Combined
   0 |   8.06 |     7.96 |   7.80 |   6.57 |     8.56
   1 |  10.05 |     8.58 |   8.26 |   8.36 |     9.90
   2 |   9.35 |     8.79 |   8.81 |   8.33 |     9.07
   4 |  10.37 |     9.45 |   9.53 |   8.82 |     9.82
   5 |  11.21 |    10.33 |  10.13 |   9.81 |    10.49
   7 |   9.67 |     8.76 |   8.55 |   8.08 |     9.94
평균 |   9.79 |     8.98 |   8.85 |   8.33 |     9.63
```

**AP 순위와 정반대다.** naive가 항상 flip 최악, **BRECQ가 6개 seed 전부 flip
최선**, Combined는 naive와 비슷한 수준(오히려 4/6 seed는 naive보다 근소하게
낫지만, AdaRound/QDrop/BRECQ보다는 뚜렷이 나쁘다). 같은 6 seed, 같은 모델들인데
"AP 1등"과 "flip 1등"이 정확히 뒤바뀐 것.

---

## 15. flip과 AP가 왜 갈리는가 — 메커니즘 정리 (사용자와의 논의로 도출)

### 15.1 flip 계산이 실제로 무엇을 재는지

`group_flip`은 (1) FP가 confident하게 H_eval을 1등으로 뽑은 anchor만 골라서,
(2) **H_eval 컬럼을 통째로 가린 뒤**, (3) 남은 **S+H_cal 60개** 안에서 새 1등
(argmax)이 FP와 양자화 모델에서 같은지를 비교한다. 즉 "정답(H_eval)을 vocabulary에서
지웠다면 승격됐을 2등이, FP와 양자화 모델에서 같은가"를 재는 지표다.

### 15.2 Combined가 왜 flip에서 밀리는가

margin_loss/neighbor_loss는 학습 중 **S 컬럼만 슬라이스해서** 계산되므로 H_cal
컬럼을 아예 본 적이 없다. 그런데 실제 조정 대상인 `s_mult`는 conv 하나를 지나는
시각 특징 전체에 곱해지는 **class-agnostic 스칼라**라서, S를 위해 조정한 결과가
공유 특징을 통해 H_cal(및 다른 모든 클래스)의 절대 점수에도 그대로 새어나간다.
flip이 순위를 비교하는 S+H_cal 60개 중 H_cal 20개는 이 누출을 전혀 방어받지
못하므로, "정답을 지운 뒤의 2등 경쟁"이 흔들리기 쉽다. 반대로 AdaRound/QDrop/
BRECQ(특히 block 단위로 상관까지 잡는 BRECQ)는 특정 클래스를 겨냥하지 않고
전 클래스에 고르게 FP를 재현하려 하므로 이 좁은 지표에서 오히려 안정적이다.

### 15.3 그런데도 AP는 왜 개선되는가

flip이 재는 "정답을 지운 뒤의 2등"은 **실제 배포에서는 벌어지지 않는 상황**이다
(정답이 vocabulary에 실제로 있으면 지워질 일이 없다). AP는 "정답이 실제로 있는
상황에서 그 정답을 confident하고 정확하게 찾아내는가"만 본다. Combined는 S
margin을 다듬는 과정의 전반적 효과로 **정답 자체의 유사도/confidence가 FP에
가깝게 유지**되고(그 부수효과가 H_eval에도 일부 번지는 것으로 보임 -- 40번의
collateral shift와 같은 메커니즘), 이게 AP 개선으로 이어진다. 반면 "정답이 없을
때의 가상의 2등"이 안정적인가는 AP와 무관하다.

### 15.4 flip을 재는 이유 자체는 타당하다 -- 단, 다른 배포 시나리오를 잰다

flip이 가정하는 "정답 클래스가 vocabulary에서 빠진 상황"은 사실 실제로 일어날 수
있다 -- open-vocabulary detection은 **같은 양자화 모델을 서로 다른 사용자가
서로 다른 vocabulary로 질의**할 수 있다는 게 핵심 전제이므로, "zebra를 아예 안
쓰는 사용자"는 실재하는 시나리오다. 문제는 **지금 AP는 "80개 다 포함한 사용자"
기준으로, flip은 "H_eval을 뺀 사용자" 기준으로 재고 있어서, 서로 다른 두 배포
조건을 하나의 표에서 비교하고 있었다**는 점이다. 이게 이번 반전의 근본 원인이다.

### 15.5 논문 서술에 대한 함의 (미결정)

이 발견 자체가 논문 §3.4("rank 보존이 AP를 보장하지 않음")를 실증하는 좋은
사례가 될 수 있다. 다만 어떻게 다룰지는 아직 결정 안 됨. 후보:
1. flip은 §3(Motivation)에서 "reconstruction만으로는 안 보이는 손상이 있다"는
   문제 제기용으로만 쓰고, method 평가/main results는 AP(특히 H_eval subset AP)
   중심으로 명확히 분리해서 서술.
2. §Evaluation Protocol의 다른 지표(GT rank, boundary inversion)를 추가로 구현해서
   flip만의 특이 현상인지, "결정 구조" 계열 지표 전체의 문제인지 구분.
3. "좁은 반사실적 지표와 실제 배포 지표가 괴리될 수 있다"는 것 자체를 분석
   섹션의 발견으로 정직하게 다루기.

---

## 16. 09-06 최종 결론

```
9  : Utility-Constrained Refinement 재통합 -> 중립적(기본 가중치 기준)
10 : QDrop/BRECQ 첫 비교(seed 2) -> naive와 거의 구분 안 됨
11 : AdaRound/BRECQ의 누적 오차 미반영 버그 발견, 수정
12 : 수정 후에도 baseline들 그대로 -> W8A8은 원래 reconstruction 개선 여지가 작음
13 : 6-seed AP 비교 -> Combined가 QDrop/BRECQ 포함 전부를 6/6으로 이김(평균 +1.35)
14 : flip 재추가 -> AP와 정반대(BRECQ가 flip 최선, Combined는 naive 수준)
15 : 메커니즘 규명 -> flip은 "정답이 vocabulary에서 빠진 다른 사용자" 시나리오,
     AP는 "정답이 실제로 있는 상황" 시나리오. 서로 다른 배포 조건을 재고 있었음.
```

09-06 세션 전체를 통해: (1) 방법론 핵심(AdaRound weight + asymmetric neighbor
scale)은 AP 기준으로 QDrop/BRECQ까지 포함해 통계적으로 탄탄하게 검증됐고, (2)
그 과정에서 발견한 flip-AP 괴리는 버그가 아니라 **서로 다른 배포 시나리오를
측정하고 있었다는 개념적 문제**임을 확인했다. 논문에서 이걸 어떻게 서술할지는
아직 미결정 -- 다음 세션에서 결정 필요.

---

## 17. 코드 구조 (09-05 문서 이후 추가분)

```text
scripts/40_seed_analysis.py       -- seed별 H_eval 구성/임베딩 유사도/anchor 수 분석
scripts/41_neighbor_preserve.py   -- neighbor preservation(symmetric/asymmetric) 검증
scripts/43_ap_check.py            -- pycocotools AP 검증(전체 + S/H_eval subset)
scripts/44_utility_ap_check.py    -- Utility-Constrained Refinement 재통합 검증
scripts/45_baseline_compare.py    -- AdaRound/QDrop/BRECQ/Combined AP+flip 비교

src/quant/promptcal.py            -- optimize_promptcal_scale_neighbor 추가
                                      (asymmetric 옵션 포함), eval_hook 지원.
                                      optimize_promptcal_scale_neighbor_utility 추가
                                      (§4.3 utility constraint 재통합)
src/quant/adaround.py             -- optimize_adaround: 누적 오차 반영(순차 재구성)로
                                      수정 -- target은 FP 기준, pred 입력은 quant_module
                                      현재 상태에서 별도 캡처. qdrop_prob도 q_in 기준.
src/quant/brecq.py                -- optimize_brecq: 동일한 누적 오차 반영 수정
                                      (target=FP 출력, input=quant_module 현재 상태)

configs/coco_local.yaml           -- train 필드 손상 수정(오래된 잔재)

results_v1/diag/39_seed{0,1,2}_seeded_full.txt  -- per-trial seed 재현성 확인
results_v1/diag/40_full.txt                     -- seed 구성 분석(10 seed)
results_v1/diag/41_full.txt, 41_w03/w05_full.txt -- neighbor_weight 스윕
results_v1/diag/41_asym_w10_full.txt, 41_asym_seed{3~9}_full.txt -- asymmetric 10-seed
results_v1/diag/42_full.txt                     -- 상관관계 재분석
results_v1/diag/43_full.txt, 43_seed{0,1,4,5,7}_full.txt -- AP 검증(6 seed)
results_v1/diag/44_full.txt, 44_seed{0,1,4,5,7}_full.txt -- Utility 재통합 검증(6 seed)
results_v1/diag/45_full.txt                     -- QDrop/BRECQ 첫 비교(seed 2, 버그 수정 전)
results_v1/diag/45_fixed_seed{0,1,2,4,5,7}_full.txt -- 누적 오차 수정 후 AP 재검증(6 seed)
results_v1/diag/45_flip_seed{0,1,2,4,5,7}_full.txt -- flip까지 포함한 최종 baseline 비교(6 seed)
```
