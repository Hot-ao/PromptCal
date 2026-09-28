# PromptCal 논문 설계 — 방법·근거·결과·한계 (2026-09-28)

**대상 독자**: `논문.txt`(논문 아웃라인)를 쓰고 있는 공저자. 아웃라인의 각 절이 **지금 코드에서
무엇으로 구현돼 있고, 어떤 측정이 받쳐주며, 무엇이 아직 비어 있는지**를 한 문서에서 보게 하는 게 목적이다.
이전 문서(`docs/PROMPTCAL_CURRENT_MODEL_V2/V3.md`, `docs/PROMPTCAL_METHOD_SPEC.md`)는 옛 설계
(alpha 학습, naive 기반)를 서술하고 있어 **이 문서로 대체한다.** 수치의 출처는 항상 `runs/` 로그이며,
판정의 경위는 `docs/PROMPTCAL_CLAIMS_2026-09-15.md`의 claim 번호로 따라갈 수 있다.

---

## 0. 한 장 요약

**방법**: FP32 YOLO-World를 W8A8로 양자화할 때, (1) BRECQ로 weight rounding을 재구성한 뒤
(2) **conv당 스칼라 activation-scale 배율 `s_mult`(52개, head 포함 시 70개)만** 추가로 학습한다.
이 2단계의 목적함수가 tensor 재구성이 아니라 **region–prompt similarity의 top-k 순위 경계**다.
weight, rounding, BRECQ가 배운 것은 전부 동결이고, `s_mult`는 activation scale에 흡수되므로 추론 오버헤드는 0이다.

**데이터가 지지하는 주장** (전부 6-seed, 같은 run 안 짝비교):

| | head 제외(QDrop 프로토콜) | head 포함(fully quantized) |
|---|---|---|
| LVIS_AP vs BRECQ | **6/6**, +0.0019 | **6/6**, +0.0041 |
| COCO_AP vs BRECQ | 3/6, 0.000 (무차) | **6/6**, +0.29 |
| LVIS_lost vs BRECQ | 6/6, −161 | 5/6, −144 |
| LVIS_flip vs BRECQ | 6/6, −0.50 | 3/6, −0.11 |
| Heval_flip / lost(COCO) vs BRECQ | 5/6 악화 / 무차 | **0/6 / 0/6 (악화)** |

**정직한 한 줄**: 재구성 기반 PTQ는 OVOD에서 flip/lost는 줄이지만 **LVIS_AP는 못 올린다**(head 포함 시
BRECQ·QDrop·AdaRound 모두 naive 이하). 우리 항을 얹으면 **AP와 LVIS 검출 손실은 일관되게 좋아지지만
그 크기는 작고**(LVIS_AP +0.4 AP점, COCO_AP +0.3 AP점), **COCO top-1 안정성 지표는 오히려 나빠진다.**
"decision을 더 잘 보존한다"가 아니라 "open-vocab AP를 올리고 LVIS 검출 손실을 줄이되 COCO 안정성 일부를
내준다"가 데이터에 맞는 서술이다. 방법의 동작 범위는 **W8A8에 한정**된다(§7).

---

## 1. 문제 설정

### 1.1 OVOD의 decision 구조 (논문 §Preliminary)

YOLO-World는 각 region 특징 `x`와 텍스트 임베딩 `w_j`(prompt j)의 유사도로 점수를 낸다
(`ContrastiveHead`, `head.cv4`):

```
sim[r, j] = logit_scale.exp() · cos( x̂_r , ŵ_j ) + bias        (pre-sigmoid)
score[r, j] = σ(sim[r, j])
```

한 region의 결정은 절대 logit이 아니라 **배포 시 주어진 prompt 집합 안에서의 상대 순위**다. `cv4`에는 conv가
없다(normalize → einsum → 스칼라 2개). 즉 결정이 만들어지는 지점 자체는 양자화 대상이 아니고, 그 **앞단이
만드는 region 임베딩(`cv3` 분기)의 오차**가 순위를 흔든다.

### 1.2 측정 지표 (논문 §Evaluation Protocol)

| 종류 | 지표 | 정의 |
|---|---|---|
| detection utility | `COCO_AP`, `LVIS_AP/APr/APc/APf` | 공식 val2017 5000장 / LVIS minival 4809장 |
| decision damage | `Top1_flip` | FP의 top-1 prompt가 양자화 후 바뀐 region 비율(표준, AP와 같은 배포 조건) |
| | `Heval_flip` | H_eval 20개 prompt를 마스킹하고 나머지 위에서 top-1이 바뀐 비율(**진단용**, calib 이미지 × COCO-80) |
| | `lost / gained / lateral` | FP가 맞힌 것을 잃음 / 새로 맞힘 / 틀린 채로 다른 class로 이동 (GT 기준) |
| | `LVIS_flip`, `LVIS_lost` | 위를 LVIS 1203 class에서 |
| | `GT_MRR`, `UPIR` | GT 순위의 평균 역순위 / 무관 prompt 침입률 |

> **서술 주의**: `Heval_flip`은 이름에 held-out이 붙었지만 calibration 이미지 위에서 COCO-80 안의
> prompt로 잰다. **진짜 vocabulary shift는 LVIS(1203 class)** 다.

---

## 2. 방법 개요

```
FP32 YOLO-World (동결, teacher)
  │
  ├─ [양자화기 초기화]  W8: 채널별 비대칭, L_2.4 MSE 탐색   /  A8: per-tensor 비대칭, MSE observer
  │                     (calibration = COCO train2017 256장, 평가 데이터와 분리)
  │
  ├─ [Stage 1] BRECQ block-wise 재구성 (rounding alpha만, 2000 iter)     ← "strong reconstruction PTQ"
  │             LSQ는 켜지 않는다 (아래 Stage 2가 activation scale을 담당)
  │
  └─ [Stage 2] Semantic Calibration:  s_mult (conv당 스칼라, 52 / head 포함 70개)만 학습
                목적함수 = decision-level (margin + neighbor + scale_reg),  1500 iter
                └→ 학습 후 activation scale에 곱해서 흡수 → 추론 오버헤드 0
```

**BRECQ와의 관계를 정확히 말하면**: BRECQ 기준선은 `rounding alpha + activation step(LSQ)`을 *재구성 손실*로
공동 학습한다. 우리는 `rounding alpha`를 BRECQ로 얻고, **activation scale 손잡이를 같은 개수(conv당 1개)로
두되 목적함수만 decision-level로 바꾼 것**이다. 자유도는 같고 감독만 다르다 — 이게 목표로 하는 서술이다.

> ⚠️ **다만 "감독만 다르다"는 지금 완전히 격리돼 있지 않다.** 코드를 대조하면 Combined의 Stage 1과 기준선 BRECQ가
> 다른 점이 둘 있다(`run_comparison.py::build`).
> 1. **LSQ**: Combined Stage 1은 꺼짐(의도된 차이 — 그 자리를 `s_mult`가 대신함).
> 2. **neck 재구성 단위**: Combined Stage 1은 `optimize_brecq`의 **함수 기본값 `neck_layerwise=True`**를 그대로
>    받아 neck을 conv 단위로 재구성하고, 기준선 BRECQ는 CLI 기본값(`--neck-layerwise` 꺼짐)으로 **block 단위**다.
>    이 차이는 **의도된 것이 아니라 호출부에 인자가 안 넘어가서 생긴 것**이다. 기준선에 대해서는 batch 2와 묶어 측정해
>    "노이즈 수준"임을 확인했지만(`PROMPTCAL_BASELINE_FIDELITY` §6.3), Combined-vs-BRECQ 짝비교에서 단독으로
>    격리한 적은 없다. **§10의 대조 실험으로 닫아야 한다.**

---

## 3. Stage 2 상세 (논문 §Semantic Calibration / §Utility-Constrained Refinement)

코드: `src/quant/promptcal.py::optimize_promptcal_scale_neighbor` (`pipeline/quant/`에 동일 사본).

### 3.1 프롬프트 3분할 — 실험 정직성의 핵심

COCO-80을 seed마다 무작위 3분할한다(`run_comparison.py`의 `rng.permutation(80)`).

| 그룹 | 개수 | 역할 |
|---|---|---|
| **S** (seen) | 40 | calibration에서 margin 보존 대상 |
| **H_cal** | 20 | calibration에 포함하되 "안 본 prompt 흉내"로 쓰는 그룹 |
| **H_eval** | 20 | **학습에 절대 안 씀.** anchor 선정·neighbor 후보에서도 제외. 측정 전용 |

최적화는 `S ∪ H_cal`(60개)만 본다.

### 3.2 Region Filtering — reliable anchor (GT 불필요)

FP(teacher)가 `S ∪ H_cal` 컬럼에서 확신하는 region만 쓴다:
`anchor = { r : max_{j ∈ S∪H_cal} σ(sim_fp[r,j]) > 0.25 }`. anchor를 80열 전체로 고르면 FP가 H_eval을
1등으로 확신한 region이 섞여 H_eval이 새므로(claim1) **`S∪H_cal` 기준으로만** 고른다.

### 3.3 손실

```
L = L_margin(S) + w_cal · L_margin(H_cal) + w_nb · L_neighbor + w_reg · L_scale
    (w_cal = 1.0,  w_nb = 1.0,  w_reg = 1.0)
```

**(a) `L_margin(G)`** — top-k 경계 margin을 FP와 맞춘다 (`k = 5`, `boundary_w = 3`).
그룹 G의 컬럼 위에서 FP의 top-(k+1) 값 `f_1 ≥ … ≥ f_{k+1}`과 그 class `π_1..π_{k+1}`를 뽑고,
양자화 쪽은 **같은 class 위치**를 gather한다(`identity_aware`): `q_i = sim_q[π_i] (i ≤ k)`,
마지막 칸은 침입을 잡기 위해 `q_{k+1} = max_{j ∉ π_1..π_k} sim_q[j]`.

```
m^f_i = f_i − f_{i+1},   m^q_i = q_i − q_{i+1}        (i = 1..k)
L_margin = mean_{r, i}  w_i · ( m^q_i − m^f_i )²        w = [1, …, 1, 3]   (경계 칸 3배)
```

대칭 MSE다. 단측(FP보다 좁아진 margin만 벌점)으로 바꾸면 **오히려 나빠졌다**(§5) — 넓어진 margin을 억제하는
성질이 암묵적 앵커로 작동하는 것으로 보인다.

**(b) `L_neighbor`** — S의 각 class에 대해 FP 텍스트 인코더의 최근접 이웃(`neighbor_k = 5`, 비-S·비-H_eval)의
유사도가 **FP보다 커지는 방향만** 억제한다(단측 hinge):

```
L_neighbor = mean  relu( sim_q[r, N] − sim_fp[r, N] )²
```

teacher 신호(FP 텍스트 인코더)만 쓰고 GT를 안 쓴다(논문 §Prompt Selection과 일치).
⚠️ **한계**: `S(40)+H_cal(20)+H_eval(20)=COCO-80` 전부라서 이웃 후보 풀이 구조적으로 **H_cal과 정확히 같다**
(claim16). 즉 이 항은 "calibration에 없던 prompt"를 대변하지 못하고, 과적합 억제 담당이라는 원래 의도는
COCO-80 폐쇄 구조에서 무력화돼 있다.

**(c) `L_scale = mean_conv (s_mult − 1)²`** — 값싼 전역 하향 편향(모든 activation을 조금씩 줄여 hinge를 쉽게
만족시키는 지름길)을 막는다. 이게 없으면 s_mult가 1보다 작게 수렴해 calibration 밖 vocabulary에서 AP가 깎인다
(`PromptCal_PTQ_progress_2026-09-07.md` §7.3~7.7, `scripts/52_smult_ablation.py`로 인과 확인).

### 3.4 파라미터화와 최적화

```
s' = s_obs · clamp(s_mult, 0.1, 10)            conv당 스칼라(per-tensor), 초기값 1.0
round는 STE.   Adam lr 1e-2,  1500 iter,  gradient clip 1.0,  calib 256장 순환(it % n)
```

- **동결**: detector weight, BRECQ의 rounding, MSE observer scale. 학습되는 건 `s_mult` 52개(head 포함 70개)뿐.
- **per-tensor 유지**: baseline이 전부 per-tensor activation이고, 표준 INT8 커널이 per-channel activation
  dequant를 지원하지 않는다. channelwise `s_mult`는 **공정성(unisolated confound)과 배포 가능성** 두 이유로
  배제했다(claim4).
- **`--deterministic` 필수**: 없으면 같은 seed에서도 s_mult가 52/52 conv에서 달라진다(max|Δ| 0.25).

### 3.5 비용과 추론 오버헤드 (논문 §Complexity)

- 추론: `s_mult`가 activation scale에 곱으로 흡수 → **오버헤드 0**. 단 **실측 latency는 없다**(fake-quant
  시뮬레이션이라 실제 INT8 실행 코드가 저장소에 0줄, §9).
- calibration: Stage 1이 BRECQ weight 재구성이라 **BRECQ 기준선과 같은 수준**이다. head 포함 실측(RTX 4000 Ada,
  4~5잡 동시): BRECQ 빌드 1090~1130s, Combined 빌드 1250~1390s → **BRECQ 대비 +150~300s(약 +14~28%)**.
  이전에 쓰던 "BRECQ의 1/16~1/24 비용" 포지셔닝은 **기반이 naive였을 때의 수치라 더는 유효하지 않다**
  (`PAPER_EVIDENCE_MAP.md` 상단 배너 참고). 비용은 이 방법의 강점이 아니다.

---

## 4. 논문 아웃라인 ↔ 구현 대응 (`논문.txt` 기준)

| 논문 소절 | 구현 | 상태 |
|---|---|---|
| Quantizer Initialization — "strong reconstruction PTQ 또는 naive" | **BRECQ**(rounding만). naive 기반은 "재구성 + ranking"이라는 주장을 시험하지 못해 잘못된 기반이었다(claim20) | ✅ 서술을 "BRECQ 위에서"로 확정 |
| Region Filtering | FP confident anchor(`max σ > 0.25`), `S∪H_cal` 기준 | ✅ |
| Prompt Selection | FP 텍스트 인코더 최근접 이웃, GT 불필요 | ✅ 단 풀이 H_cal로 포화(§3.3-b) |
| Semantic Objective — "local reconstruction + semantic consistency" | `L_margin + L_neighbor + L_scale`. **local reconstruction 항은 Stage 1(BRECQ)이 담당** — Stage 2에 얹어 공동 최적화하는 시도는 기각(claim21) | ⚠️ "재구성은 1단계, semantic은 2단계"로 서술 |
| Utility Constraints (threshold crossing, box consistency) | 구현돼 있으나 **기본 꺼짐**. 측정상 순수 이득 없음(claim19). `l_thresh`는 인덱싱 버그로 한 번도 작동한 적 없었고 커밋 `5bc00d6`에서 수정 | ❌ 본문에서 빼거나 "시도했고 이득 없음"으로 |
| Optimization — "scale/clipping/rounding만" | **scale만.** rounding(claim14·18-a·21·22)과 clipping(claim18-b)은 시도 후 악화 | ⚠️ 서술을 "scale만"으로 정정 |
| Complexity — inference zero overhead | ✅ 흡수 가능. **latency 미측정** | ⚠️ |
| Precision-Induced Semantic Damage — FP32/FP16/FP8/INT8 ladder | W8A8·W8A6·W4A8 등만 있음. FP16/FP8 없음 | ❌ 미실시 |
| Prompt-Axis Shift — semantic hard negative vs random | 없음 | ❌ 미실시 |
| Reconstruction–Utility Misalignment | ✅ **observer min-max↔MSE에서 COCO와 LVIS가 6/6 seed 반대 방향**, head 포함 시 재구성 기반 3종이 flip은 줄이나 LVIS_AP는 naive 이하 | ✅ 가장 단단한 절 |
| Cross-Architecture (OWLv2) | **코드 0줄** | ❌ |
| Efficiency (latency/memory/INT8 실측) | 코드 0줄. model size만(9.88 MiB, head 포함 12.08 MiB) | ❌ |

---

## 5. 확정된 설계와 기각된 설계

**확정 (기본값)**: Stage 1 = BRECQ 2000 iter, Stage 2 = `s_mult`, `L_margin(대칭, identity-aware) + L_neighbor(단측) + L_scale`,
`--calib 256 --iters 1500 --lr 1e-2 --k 5 --neighbor-k 5 --scale-reg-weight 1.0 --cal-weight 1.0 --deterministic`.

**기각 (전부 플래그로 남아 있고 기본 꺼짐 = 끄면 bit-identical)**:

| 시도 | 결과 | 출처 |
|---|---|---|
| Stage 1을 생략(naive 기반) | "재구성 + ranking"을 시험 못 함 — 이후 실패 8건의 공통 원인 | claim20 |
| `aux_mse`, `neighbor_of_cal` | 6-seed 반박 | claim16 |
| **rounding alpha 학습**(W8A8) | 10/10 지표 악화 | claim18-a |
| activation 범위(클리핑) 확장 | 7/10 악화, 이득 0 | claim18-b |
| utility 두 항(`l_thresh`, `l_box`) | LVIS_APr 트레이드, 순이득 없음 | claim19 |
| 단측 margin, 랜덤 샘플링, per-group anchor, local recon, `region_dir`, `range_blend` | 측정상 이득 없음/악화 | claim18~20 |
| **W4A8 공동최적화**(`learn_alpha` + block recon) | **붕괴**(COCO_AP 33.5→18.5, 8/8 악화) | claim21 |
| **`alpha_bias`**(conv당 스칼라로 rounding 보정, 자유도 52개) | **붕괴**(COCO_AP 33.48→27.77, 8지표 0/3) | claim22 |
| channelwise `s_mult` | 공정성·배포성 이유로 배제 | claim4 |

**claim21·22의 교훈**: W4에서 rounding을 decision loss로 건드리면 자유도가 수백만이든 52개든 똑같이 붕괴한다
(BRECQ 반올림 대비 flip 6.2% vs 6.6%). 원인은 "앵커 대비 자유도 과다"가 아니라 **rounding 접근 자체**다.
claim21의 초기 진단(자유도 폭발)은 claim22가 직접 반증했다.

---

## 6. 결과 (W8A8, full probe, 6-seed 평균, 같은 seed·같은 GPU 모델)

### 6.1 head 제외 — QDrop COCO 프로토콜 (`runs/112`, AdaRound·QDrop은 `runs/97`)

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_flip | LVIS_lost | LVIS_AP | LVIS_APr |
|---|---|---|---|---|---|---|---|---|
| FP32 | 36.80 | – | – | – | – | – | 0.2589 | 0.1767 |
| naive | 36.617 | 5.945 | 0.667 | 291 | 4.467 | 985 | 0.2554 | 0.1777 |
| AdaRound(min-max) | 36.648 | 5.142 | 0.472 | 234 | 2.813 | 639 | 0.2577 | 0.1748 |
| QDrop | 36.807 | 4.090 | 0.397 | 180 | 2.867 | 642 | 0.2556 | 0.1774 |
| BRECQ | 36.753 | 3.802 | 0.367 | 182 | 2.723 | 622 | 0.2560 | 0.1780 |
| **ours** | 36.753 | 4.278 | 0.378 | 180 | **2.227** | **461** | **0.2579** | 0.1774 |

### 6.2 head 포함 — fully quantized (`runs/115`+`runs/121`+`runs/117`, claim23)

`WorldDetect` 전체(박스 회귀 `cv2` 9개 + region 임베딩 `cv3` 9개 = conv 18개) 양자화. AdaRound는 **MSE observer**.

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_flip | LVIS_lost | LVIS_AP | LVIS_APr |
|---|---|---|---|---|---|---|---|---|
| FP32 | 36.80 | – | – | – | – | – | 0.2589 | 0.1767 |
| naive | 36.36 | 7.94 | 0.74 | 332 | 5.39 | 1197 | 0.2540 | 0.1771 |
| AdaRound(MSE) | 36.50 | 7.01 | 0.55 | 261 | 4.64 | 1049 | 0.2521 | 0.1751 |
| QDrop | 36.54 | 6.35 | 0.49 | 220 | 3.88 | 840 | 0.2538 | 0.1805 |
| BRECQ | 36.29 | 6.52 | 0.48 | 231 | 3.97 | 861 | 0.2514 | 0.1784 |
| **ours** | **36.58** | 7.56 | 0.51 | 256 | **3.87** | **717** | **0.2555** | 0.1781 |

**ours vs BRECQ (같은 run 짝비교)**: COCO_AP 6/6 (+0.29, t=6.2), LVIS_AP 6/6 (+0.0041, t=8.4),
LVIS_lost 5/6 (−144, t=−4.2), LVIS_flip 3/6, LVIS_APr 3/6(무차), **Heval_flip 0/6 (+1.04), lost 0/6 (+25)**.
ours vs QDrop(가장 강한 baseline): COCO_AP 3/6(무차), LVIS_AP 6/6 (+0.0017), LVIS_lost 5/6 (−122), Heval_flip 1/6, lost 0/6.

**해석**
- head를 넣으면 **BRECQ는 naive보다도 나빠지고**(COCO_AP 36.29<36.36, LVIS_AP 0.2514<0.2540), 재구성 기반 3종은 flip/lost는
  줄이면서 LVIS_AP는 naive에 못 미친다. 우리는 두 AP 모두 5조건 중 1위 — FP32 대비 손상이
  COCO_AP −0.51→−0.22, LVIS_AP −0.0075→−0.0034.
- **절대 크기는 작다.** "BRECQ 손상의 55% 회복"은 분모(BRECQ의 손상)가 0.75 AP점밖에 안 돼서 나오는 수치다.
- **AdaRound의 min-max observer는 head 포함 시 붕괴**(COCO_AP 36.6→33.4)하고 MSE로 바꾸면 즉시 회복된다(runs/117 단일변수).
  observer 아티팩트라 우리 기여가 아니며, 표에는 MSE 버전을 쓴다.

### 6.3 Reg-PTQ와의 관계 (head 손실이 작은 이유)

Reg-PTQ(CVPR'24)가 보고하는 "head 양자화의 큰 손실"은 주로 **two-stage 검출기(Faster/Mask R-CNN)의 RoI+FC regressor**와
**W2A4~W4A4**에서 나온다. 같은 논문 표에서 one-stage(RetinaNet, YOLOF)는 BRECQ로도 W4A8에서 약 −0.6~−1.3 AP만 잃는다.
YOLO-World head는 FPN 각 레벨에 붙는 conv 기반 one-stage head이고 우리는 W8A8이라 손실이 작은 게 자연스럽다.
인용 시 "head 양자화는 늘 위험하다"가 아니라 "regression head 난이도는 구조와 bit-width에 좌우된다"로 스코프를 좁힐 것.

---

## 7. 동작 범위 (Limitations로 정직하게 기록)

| 설정 | 결과 |
|---|---|
| **W8A8** | 위 표. 작동 |
| W8A7 / W8A6 | 개선 지표 4/7 → **0/7**(naive 기반 Combined, 1-seed `--eval-cap 500` 스캔이라 참고용). 2비트 사이에 부호가 뒤집힌다 — `s_mult`는 "거의 맞는 출발점에 작은 배율 보정"을 전제 |
| W8A5 | BRECQ가 naive보다 나쁨(재구성 발산) |
| W8A4 / W8A3 | **baseline까지 전멸**(COCO_AP 0.0) — per-tensor 비대칭 activation 스킴의 한계 |
| **W4A8** | brecq는 33.5로 복구하는데 우리는 **0/3 악화**(`s_mult`만 −0.56). rounding에 접근하는 두 방법(claim21·22)은 붕괴 |
| W4A4 | BRECQ조차 붕괴(COCO_AP 0.73) |

`s_mult`는 **activation scale만** 건드리므로 손상의 주범이 weight rounding인 W4에서는 손이 닿지 않고, 주범이 activation인
W8A≤5에서는 baseline이 먼저 죽는다. 저비트로 가려면 rounding 접근이나 activation 양자화 스킴(per-channel, log-scale 등)을
바꿔야 하는데 전자는 두 번 실패했고 후자는 배포성 제약(claim4)과 충돌하며 논문의 기여 축이 양자화기 설계로 바뀐다.

---

## 8. baseline 공정성 프로토콜

- 조건: naive / AdaRound / QDrop / BRECQ / Combined. **`--conditions` 목록이 RNG 스트림 위치를 정하므로**
  비교할 run은 동일하게 유지한다.
- AdaRound: iters 2000(QDrop/BRECQ와 정렬). observer는 head 제외에선 논문대로 min-max(as-published, COCO↔LVIS 트레이드오프가
  실재), **head 포함에선 MSE**.
- BRECQ: 공동 최적화(alpha+LSQ). 공식의 2단계 분리는 성능이 악화돼 사용 안 함. QDrop: batch 1, neck block-wise
  (공식 batch 2·neck layer-wise는 노이즈 수준 차이에 비용 2.7배라 사용 안 함 — 각주로 밝힐 것).
- CLIP 인코더는 항상 제외. 모델 크기는 9.88 MiB(head 제외) / 12.08 MiB(head 포함).
- **GPU 종류를 섞지 않는다**(L40S와 RTX 4000 Ada는 커널이 달라 bit-identical이 깨진다). `CUDA_DEVICE_ORDER=PCI_BUS_ID` 필수.
- BRECQ는 `--deterministic`에서도 완전히 결정적이지 않다(`adaptive_max_pool2d_backward_cuda`, claim18-d) — 그래서
  방법 비교는 **같은 run 안 짝비교**로만 판정한다.

---

## 9. 논문이 주장할 수 있는 것 / 없는 것

**할 수 있다**
1. OVOD PTQ에서 재구성 충실도와 task utility는 어긋난다 — observer 선택이 COCO와 LVIS를 6/6 seed 반대로 움직이고, head 포함 시
   재구성 기반 3종은 flip/lost는 줄이나 LVIS_AP는 naive 이하다.
2. 그 위에 decision-level 목적함수로 학습한 **activation scale 배율 52~70개**만 얹으면 LVIS_AP와 LVIS 검출 손실이
   6/6(5/6) 일관되게 개선된다. 자유도는 baseline의 LSQ와 같지만, **"감독의 종류만 바뀐 효과"라고 단정하려면 Stage 1
neck 설정 차이(§2 경고)를 먼저 격리해야 한다.**
3. 추론 오버헤드 없음(구조상). 재학습 없음.

**할 수 없다**
- "decision을 더 잘 보존한다": Heval_flip·COCO lost는 0/6으로 악화, LVIS_flip은 head 포함 시 3/6.
- "큰 이득": 절대 효과는 LVIS_AP +0.4, COCO_AP +0.3 AP점.
- "저비트로 확장된다": W8A8에 한정.
- "비용이 싸다": Stage 1이 BRECQ라 BRECQ와 같은 수준.
- "다른 아키텍처에서도 된다"(OWLv2 코드 없음), "실제 INT8에서 빠르다"(측정 없음).

---

## 10. 열린 항목

**결정이 필요한 것**
- **헤드라인 프로토콜**: head 제외(QDrop 원 논문 프로토콜, LVIS 3지표 전승) vs head 포함(fully quantized, AP 이득이 2배지만
  flip/lost는 진다). OVOD 배포 관점에선 head 포함이 자연스럽지만 서사는 더 복잡해진다. 두 표를 병기하는 안도 가능.

**실험 (`docs/PAPER_EVIDENCE_MAP.md` 우선순위 기준)**
0. **Stage 1 격리 대조 (가장 싸고 가장 먼저)**: `build()`의 combined 분기가 `optimize_brecq`에 `neck_layerwise`(그리고
   `batch`)를 넘기도록 고쳐 기준선 BRECQ와 Stage 1 설정을 맞추고, 짝비교가 그대로 유지되는지 확인. 유지되면 §2의
   "감독만 다르다"가 성립하고, 사라지면 이득의 일부가 neck 재구성 단위 차이였다는 뜻이다. 현재 결과는 이 수정 **전** 코드로
   측정된 것이라, 수정하면 `--combined-stage1 brecq` 경로의 수치가 바뀌므로 플래그로 opt-in해서 기존 결과를 보존할 것.
1. 실제 INT8 배포 측정(latency/memory) — 코드 0줄
2. 두 번째 아키텍처(OWLv2) — 코드 0줄. 지금 결과가 "이 모델 특성 아니냐"는 반론에 약하다
3. semantic hard-negative vs random prompt 통제, FP16/FP8 ladder(§Motivation)
4. head 제외 프로토콜의 Heval_flip 악화 원인 규명(왜 AP는 오르는데 top-1 안정성은 내주는가)

---

## 11. 재현

```bash
export YOLO_AUTOINSTALL=false
# head 포함 W8A8 (claim23). head 제외는 --no-skip-head 와 --adaround-act-observer mse 를 빼면 된다.
.venv/bin/python pipeline/run_comparison.py \
  --model yolov8s-world.pt --device <gpu> --seed <0..5> --deterministic \
  --w-bits 8 --a-bits 8 --calib 256 --no-skip-head --adaround-act-observer mse \
  --conditions naive,adaround,qdrop,brecq,combined \
  --combined-stage1 brecq --combined-recon-iters 2000
```

환경·소요 시간·측정 신뢰성은 [`pipeline/BASELINE_STATUS.md`](pipeline/BASELINE_STATUS.md),
파일 지도는 [`pipeline/README.md`](pipeline/README.md), 판정 경위는
[`docs/PROMPTCAL_CLAIMS_2026-09-15.md`](docs/PROMPTCAL_CLAIMS_2026-09-15.md)(claim14~23).
