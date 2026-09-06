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

## 9. 코드 구조 (09-05 문서 이후 추가분)

```text
scripts/40_seed_analysis.py       -- seed별 H_eval 구성/임베딩 유사도/anchor 수 분석
scripts/41_neighbor_preserve.py   -- neighbor preservation(symmetric/asymmetric) 검증
scripts/43_ap_check.py            -- pycocotools AP 검증(전체 + S/H_eval subset)

src/quant/promptcal.py            -- optimize_promptcal_scale_neighbor 추가
                                      (asymmetric 옵션 포함), eval_hook 지원

configs/coco_local.yaml           -- train 필드 손상 수정(오래된 잔재)

results_v1/diag/39_seed{0,1,2}_seeded_full.txt  -- per-trial seed 재현성 확인
results_v1/diag/40_full.txt                     -- seed 구성 분석(10 seed)
results_v1/diag/41_full.txt, 41_w03/w05_full.txt -- neighbor_weight 스윕
results_v1/diag/41_asym_w10_full.txt, 41_asym_seed{3~9}_full.txt -- asymmetric 10-seed
results_v1/diag/42_full.txt                     -- 상관관계 재분석
results_v1/diag/43_full.txt, 43_seed{0,1,4,5,7}_full.txt -- AP 검증(6 seed)
```
