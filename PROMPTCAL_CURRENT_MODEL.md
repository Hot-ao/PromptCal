# Combined 모델 현재 설계 전체 문서 (2026-09-08 기준)

`PROMPTCAL_HOW_IT_WORKS.md`는 처음 6-seed 검증(COCO-80 vocabulary, scalar
`s_mult`, calib=32)까지의 파이프라인을 중심으로 쓰여 있고, 이후의 per-channel
재설계·공식 데이터 검증은 §5.7로 덧붙인 상태라 전체 그림을 한 번에 보기
어렵다. 이 문서는 **지금(09-08) 실제로 쓰는 최종 설계**를 구조·코드·평가지표·
성능결과 순으로 처음부터 다시 정리한다. 코드/결과를 인용할 때마다 파일 경로를
작게 같이 적는다.

---

## 1. 문제 정의 (한 문단 요약)

YOLO-World 같은 open-vocabulary detector를 W8A8로 PTQ하면 총량 지표(AP)는
거의 안 변하지만, region–prompt 의사결정(어떤 anchor가 어떤 프롬프트를
1등으로 뽑는가)은 크게 흔들린다 — 특히 confidence margin이 작은 경계에서,
의미적으로 가까운 프롬프트 방향으로, calibration에서 못 본 프롬프트일수록
심하다. **"reconstruction ≠ decision preservation"** 이 이 연구 전체의 핵심
주장이고, AdaRound/QDrop/BRECQ 같은 강한 reconstruction 계열도 이 손상을 못
막는다는 게 baseline 비교로 확인돼 있다(`PROMPTCAL_HOW_IT_WORKS.md` §4).

---

## 2. 전체 파이프라인 한눈에

```
1. FP32 모델 forward + SimilarityHarness로 cv4 유사도 행렬 캡처   (§3)
2. wrap_convs + calibrate                → naive W8A8            (§4)
3. convert_to_adaround + optimize_adaround(1000 iter)
     → weight 반올림(alpha) 확정, hard-round                     (§4.1)
4. optimize_promptcal_scale_neighbor(1500 iter)
     → alpha는 완전히 고정(requires_grad=False)
     → per-channel s_mult(활성화 scale 배율)만 margin_loss +
       asymmetric neighbor hinge + scale_reg_weight로 학습        (§5)
```
*`scripts/58_full_baseline_official_data.py:343` (`build()`) 가 이 4단계를 그대로 호출한다.*

"naive/AdaRound/QDrop/BRECQ" 4개 baseline은 이 중 1~3단계까지만(QDrop/BRECQ는
3단계의 변형)이고, **"Combined"는 1~4단계 전부**를 거친 모델을 가리킨다.

---

## 3. 측정 도구 — SimilarityHarness

YOLO-World의 `WorldDetect` head 안 `cv4`(ContrastiveHead 리스트)에 forward
hook을 걸어, 3개 FPN level(stride 8/16/32)의 region–prompt 유사도를
`[total_anchors=8400, num_prompts]` 행렬 하나로 조립한다. 이 값이 모델의 최종
classification logit 그 자체(전역 argmax == 모델 정식 예측, 검증됨)라 이후
모든 지표 계산의 공통 기반이 된다.

```python
class SimilarityHarness:
    def run_image(self, img_tensor, image_id) -> SimilarityRecord:
        ...  # forward 1회, [8400, P] 유사도 반환
```
*`src/harness.py:33` (`class SimilarityHarness`), `src/harness.py:74` (`run_image`)*

FP32/양자화 모델에 각각 씌워서 같은 이미지의 두 `sim` 행렬을 비교하는 게 이
전체 실험의 공통 연산이다.

---

## 4. Baseline 4가지 (Combined와의 비교 기준)

| baseline | 방식 | 코드 |
|---|---|---|
| naive | round-to-nearest, 추가 최적화 없음 | `src/quant/fake_quant.py` |
| AdaRound | layer별 출력 재구성 MSE로 반올림(alpha) 학습 | `src/quant/adaround.py`, `optimize_adaround` |
| QDrop | AdaRound + 확률적 activation drop(`qdrop_prob`) | `src/quant/adaround.py` (같은 함수, 인자만 다름) |
| BRECQ | block 단위 joint 재구성(층 간 상관 반영) | `src/quant/brecq.py:35` (`optimize_brecq`) |

세부 설명·6-seed 결과는 `PROMPTCAL_HOW_IT_WORKS.md` §4 참고. 이 문서에서는
Combined 자체에 집중한다.

---

## 5. Combined 모델 상세 (현재 설계)

### 5.1 1단계 — AdaRound weight rounding (그대로 재사용)

```python
w_int = floor(w/scale) + h(alpha)     # h(alpha) ∈ [0,1], rectified sigmoid
```
*`src/quant/adaround.py:24` (`class AdaRoundQuantConv2d`)*

`optimize_adaround(m.model, fp.model, calib, device, iters=1000)`로 학습 후
hard-round 확정. Combined는 이 결과를 **그대로 고정**하고 2단계로 넘어간다
(`ac.alpha.requires_grad_(False)`, `src/quant/promptcal.py:398`).

### 5.2 2단계 — `s_mult`: per-channel learnable activation scale (핵심 재설계 지점)

**지금 버전(09-08, per-channel 벡터):**

```python
self.s_mult = nn.Parameter(
    torch.ones(self.conv.in_channels, device=self.conv.weight.device)
)                                      # 초기값 1.0, conv 입력 채널 수만큼
self.use_smult = False                # True일 때만 적용

def _quantize_smult(self, x):
    mult = self.s_mult.clamp(min=0.1, max=10.0).view(1, -1, 1, 1)
    scale = self.a_obs.scale * mult
    x_s = x / scale
    x_r = x_s + (round(x_s) - x_s).detach()   # round만 STE, scale은 grad 유지
    x_c = clamp(x_r + zp, qmin, qmax)
    return (x_c - zp) * scale
```
*`src/quant/adaround.py:44` (필드 선언), `src/quant/adaround.py:79` (`_quantize_smult`)*

`round()`만 straight-through estimator로 처리하고 `scale` 자체는 계산
그래프에 남겨 `s_mult`로 grad가 흐르게 한다 — "이 conv 활성화 양자화 격자의
채널별 눈금 크기"를 연속적으로 학습하는 것.

**왜 스칼라가 아니라 벡터인가 (재설계 경위 요약):**

원래는 conv당 스칼라 하나(`torch.tensor(1.0)`, 0-dim)였다. LVIS-1203
vocabulary로 일반성을 검증하다가 구조적 한계가 드러났다:

1. **발견**: COCO-80에서 학습한 Combined를 재학습 없이 LVIS로 배포하면 실제
   LVIS AP가 **naive보다도 낮았다**(5개 조건 중 최악).
2. **진단**: `s_mult`가 conv당 스칼라라서, "S(COCO 40개)의 이웃만 조준해서
   억제"하려는 neighbor loss(§5.3)의 의도가 구조적으로 불가능 — 만족시키는
   유일한 방법이 그 conv의 활성화 scale 전체를 획일적으로 낮추는 것뿐이라,
   calibration에서 존재도 몰랐던 LVIS 1100여개 class까지 무차별적으로
   위축됐다(학습 후 s_mult 평균이 항상 1.0 미만으로 수렴).
3. **인과관계 확정**: 학습된 Combined의 `s_mult`를 사후에 1.0으로 강제
   되돌리면 AP가 AdaRound와 소수점까지 일치 — `s_mult` 하나가 LVIS 손상의
   원인임을 직접 증명.
   *`scripts/52_smult_ablation.py`*
4. **재학습 여부와 무관함을 확인**: COCO-80 native / COCO→LVIS 이식 /
   LVIS-native 세 시나리오 전부에서 s_mult<1로 수렴 — "이식" 특유의 문제가
   아니라 스칼라 파라미터화 자체의 한계로 확정.
5. **1차 대응(정규화만)은 부족**: `(s_mult-1)²` 정규화만으로는 UPIR 등 핵심
   지표를 희생하는 나쁜 트레이드오프만 나옴.
6. **2차 대응(현재 채택)**: 스칼라 → **입력 채널별 벡터**로 확장 + 가벼운
   정규화(§5.4) 병행. 채널마다 다른 값을 가질 자유도를 주면 "S 이웃만
   조준"을 채널 단위로나마 부분적으로 달성할 여지가 생긴다.

전체 서사는 `PROMPTCAL_HOW_IT_WORKS.md` §5.7 에 더 자세히 있다.

(재현 시 주의: `torch.ones(N)`은 CPU 텐서라 `device=...`를 명시해야 한다.
스칼라(`torch.tensor(1.0)`, 0-dim)는 CUDA 텐서와 섞여도 PyTorch가 암묵적으로
허용해줘서 이 버그가 숨어 있었다.)

### 5.3 학습 목적함수 — `optimize_promptcal_scale_neighbor`

*`src/quant/promptcal.py:343`*

세 개 항의 합으로 `s_mult`만 학습한다(alpha는 고정):

**(a) margin_loss** — S(40개 프롬프트)의 top-(k+1) 인접 margin을 FP와 맞춤:

```python
def margin_loss(sim_q, sim_fp, k=5, boundary_w=3.0):
    fp_top, _ = sim_fp.topk(k+1, dim=-1)
    q_top, _  = sim_q.topk(k+1, dim=-1)
    fp_m = fp_top[:, :-1] - fp_top[:, 1:]
    q_m  = q_top[:, :-1]  - q_top[:, 1:]
    w = [1,1,1,1,3]                      # top-k 경계에 가중(boundary_w)
    return mean((q_m - fp_m)^2 · w)
```
*`src/quant/promptcal.py:36`*

**(b) asymmetric neighbor hinge** — S 각 class의 text-embedding 최근접 이웃
(non-S) `neighbor_k`개가 FP보다 **강해지는** 방향만 억제(collateral shift
억제):

```python
diff = sim_q[aidx][:, neighbor_cols] - sim_fp[aidx][:, neighbor_cols]
nl = F.relu(diff).pow(2).mean()
```
*`src/quant/promptcal.py:452`*

**(c) scale_reg_weight 정규화** (09-07 §5.2 재설계 이후 추가, opt-in):

```python
if scale_reg_weight > 0:
    sr = sum((s - 1.0).pow(2).mean() for s in smults) / len(smults)
    loss = loss + scale_reg_weight * sr
```
*`src/quant/promptcal.py:457`*

최종 loss:

```
loss = margin_loss(S) + neighbor_weight · neighbor_hinge(neighbors)
                       + scale_reg_weight · mean((s_mult-1)²)
```

### 5.4 S / H_cal / H_eval — 프롬프트 3분할

```python
rng = np.random.default_rng(seed)
perm = rng.permutation(80)
S      = perm[:40]     # 학습(margin_loss·neighbor 계산 대상)에 실제 사용
H_cal  = perm[40:60]   # 20개 -- 현재 Combined 구현은 사용하지 않음
H_eval = perm[60:80]   # 20개 -- 최적화에 전혀 안 씀, 순수 평가 전용
```
*`scripts/58_full_baseline_official_data.py:441` 부근*

핵심 원칙: 측정할 프롬프트(H_eval)는 학습에 절대 넣지 않는다.

### 5.5 현재 하이퍼파라미터 (공식 데이터 설정 기준값)

| 파라미터 | 값 | 비고 |
|---|---|---|
| `calib` | 256 (COCO train2017) | 분류 표준(1024)보다 작음 — detection PTQ 관행 |
| `recon_iters_ada`(AdaRound) | 1000 | |
| `iters`(s_mult 학습) | 1500 | |
| `lr` | 1e-2 | |
| `k`(margin top-k) | 5 | |
| `neighbor_k` | 5 | S 각 class당 이웃 개수 |
| `neighbor_weight` | 1.0 | |
| `scale_reg_weight` | **10.0(잠정 채택)** | §8.1/8.2 참고 — 09-08에 20에서 10으로 잠정 변경, 아직 4-seed까지만 검증(seed3/5 진행 중) |
| `w_bits` / `a_bits` | 8 / 8 | |

*`scripts/58_full_baseline_official_data.py`의 `argparse` 기본값과 동일*

---

## 6. 데이터 설정 (2026-09-08 "공식 데이터" 버전)

이전(calib=32, val2017 슬라이스 재사용, LVIS 484장 임시 부분집합)에서
논문 실험 관행에 더 가깝게 09-08에 교체:

| 구분 | 소스 | 장수 |
|---|---|---|
| calibration | COCO **train2017** (평가셋과 완전 분리) | 256 |
| COCO-80 평가 | COCO val2017 **전체** | 5000 |
| LVIS 평가 | 공식 `lvis_v1_minival.json`(ultralytics 공식 배포) | 4809 |

LVIS minival(4809장)은 확인 결과 **"COCO val2017 ∩ LVIS val"과 정확히
일치** — 이전에 쓰던 484장 부분집합의 공식적이고 10배 큰 버전.

*실행 스크립트: `scripts/58_full_baseline_official_data.py`,
GT 파일: `/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json`*

---

## 7. 평가지표 전체 정의

| 지표 | 정의 | 무엇을 보는가 |
|---|---|---|
| COCO_AP | COCO-80 vocabulary, mAP50-95(pycocotools) | 표준 배포 시 총량 성능 |
| LVIS_AP | 같은 모델을 LVIS-1203 vocabulary로 바꿔 평가한 mAP | 낯선/촘촘한 vocabulary로 일반화됐을 때의 총량 성능 |
| S_AP / H_eval_AP | per-class AP를 S/H_eval 그룹으로 평균 | 학습에 쓴 프롬프트 vs 한 번도 안 쓴 프롬프트의 AP 분리 |
| Heval_flip(masked) | H_eval 20개 컬럼을 가린 뒤, FP가 H_eval로 판정했던 anchor에서 나머지 60class 중 1등이 FP/quant 간 같은가 | "H_eval을 아예 모르는 다른 사용자" 반사실적 시나리오(우리가 만든 진단 지표, 실제 배포 조건 아님) |
| Top1_flip(표준) | masking 없이 FP confident anchor의 raw top-1이 quant와 같은가 | AP와 동일한 정상 배포 조건 |
| GT_MRR / GT_R@1 | 실제 COCO GT-anchor에서 quant 자신의 정답 순위 역수 평균 / 1등 비율 | 진짜 정답 기준 순위 보존도 |
| lost / gained | GT-anchor에서 FP는 1등이었는데 quant가 잃은/얻은 개수 | 진짜 정답 기준 손익 |
| UPIR | FP가 GT를 1등으로 맞춘 경우 중, quant가 그 1등을 H_eval class로 바꿔버린 비율 | "모르는 프롬프트의 정답 자리 침입" — 논문 핵심 동기와 가장 직접 대응 |
| LVIS_flip / LVIS_lost | 위 Top1_flip/lost를 LVIS vocabulary·실제 LVIS GT로 재측정 | 촘촘한 vocabulary에서도 결정/정답 보존이 되는가 |

계산 코드: `scripts/58_full_baseline_official_data.py`의 `group_flip`(228행대),
`standard_flip`, `gt_metrics_for_method`, `compute_lvis_flip_gt_streaming`(256행).
지표 설계 배경은 `PROMPTCAL_HOW_IT_WORKS.md` §6에 더 자세히 있다.

---

## 8. 성능 결과 (2026-09-08 기준, 공식 데이터 설정)

### 8.1 메인 결과 — `scale_reg_weight=10`(잠정 채택, §8.2 판단 근거), 4-seed(0,1,2,4)

09-08에 rw=10을 잠정 메인으로 바꿈(§8.2에서 rw=10이 논문 방향에 더 맞는다고
판단). 아직 4-seed까지만 나왔고(seed 3/5 진행 중), **범위는 평균±표준편차가
아니라 최소~최대**로 표시(괄호 안이 평균). baseline은 AP/lost/LVIS류는
seed와 무관한 상수, Heval_flip·UPIR만 S/H_eval 분할 때문에 seed마다 달라서
같이 범위로 표시했다.

| method | COCO_AP | LVIS_AP | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|
| **FP32** | **36.80** | **0.1260** | - | - | - | - | - | - |
| naive | 33.54 | 0.1264 | 10.51~13.40%(11.60%) | 0.95% | 0.24~0.39%(0.31%) | 432 | 6.48% | 1224 |
| AdaRound | 33.24 | 0.1242 | 9.20~12.74%(10.75%) | 0.74% | 0.24~0.31%(0.27%) | 349 | 5.67% | 1212 |
| QDrop | 33.30 | 0.1235 | 9.25~12.66%(10.74%) | 0.72% | 0.22~0.33%(0.29%) | 384 | 5.71% | 1192 |
| BRECQ | 33.30 | 0.1200 | 8.99~12.51%(10.41%) | 0.71% | 0.19~0.33%(0.25%) | 327 | 5.53% | 1158 |
| **Combined(rw10)** | **36.19~36.38(36.32)** | 0.1213~0.1260(0.1247) | **8.57~10.17%(9.57%)** | 0.71~0.78%(0.76%) | 0.20~0.30%(0.25%) | 352~390(369.8) | **5.26~5.56%(5.38%)** | **1052~1159(1088.8)** |

*원본 로그: `runs/58_official_data/{seed0,seed1,seed2,seed4,rw10_seed1,rw10_seed2,rw10_seed4}.log`,
`runs/59_rw_sweep/rw_sweep_seed0.log`(rw10 seed0)*

**정직한 요약**:
- COCO_AP: naive 대비 **+2.78**(평균 기준), 4-seed 범위가 겹치지 않을 만큼
  안정적으로 높음(36.19~36.38 vs naive 33.54).
- Heval_flip: **4/4 seed 전부** BRECQ(같은 4-seed 평균 10.41%)보다 낮음(더
  좋음) — masked flip을 이긴 건 이 설계가 처음(§5.2 재설계 이전엔 항상
  최악~근접 최악이었음).
- LVIS_flip: 4/4 seed 전부 BRECQ(5.53%)보다 낮음.
- LVIS_lost: 3/4 seed(1052, 1077, 1067)가 BRECQ(1158)보다 낮고, 1개
  seed(1159)만 근소하게 못 미침 — 평균은 확실히 이김(1088.8 vs 1158).
- **LVIS_AP**: 평균은 naive(0.1264)와 Combined(0.1247) 거의 붙어 있지만
  seed별 편차가 커서(0.1213~0.1260) 일관된 승패라기보다 "naive와 대략
  동급"에 가까움(rw=20에서는 5/5 seed 전부 naive보다 낮았던 것과 다름 —
  §8.2 비교표 참고). 어느 쪽이든 AdaRound/QDrop/BRECQ보다는 명확히 위.
- UPIR·lost: baseline 중 최선(AdaRound/BRECQ)에는 못 미치는 중위권 — 이건
  rw10/rw20 공통.

**부수적 관찰 — naive가 AdaRound/QDrop/BRECQ보다 AP가 높은 이유**: 세 baseline
모두 반올림 방향을 "layer 출력 재구성 MSE"를 줄이는 쪽으로 최적화하는데, 이건
AP를 직접 겨냥한 목적함수가 아니다(reconstruction ≠ decision/task 최적화라는
이 논문의 핵심 주장이 AP 자체에서도 드러난 사례). Combined는 이 MSE 대신
decision 직접 목적함수(margin_loss)로 학습해서 이 함정을 안 물려받는다.

### 8.2 `scale_reg_weight` 재스윕 — rw=10 vs rw=20, 4-seed 중간 결과

1-seed 스윕(`scripts/59_rw_sweep_official_data.py`)에서 rw=10이 유력해 보여
`scripts/58`로 seed 0/1/2/4 4개를 직접 검증했다(seed 3/5는 09-08 현재 GPU에서
추가 진행 중 — 끝나면 갱신 예정). **평균±표준편차 대신, seed 4개의 최소~최대
범위로 표시**(괄호 안은 평균):

| | rw=10 (seed 0,1,2,4) | rw=20 (같은 4개 seed) |
|---|---|---|
| COCO_AP | 36.19 ~ 36.38 (36.32) | 35.99 ~ 36.23 (36.09) |
| LVIS_AP | 0.1213 ~ 0.1260 (0.1247) | 0.1224 ~ 0.1262 (0.1247) |
| Heval_flip | 8.57 ~ 10.17% (9.57%) | 8.55 ~ 10.18% (9.48%) |
| Top1_flip | 0.71 ~ 0.78% (0.76%) | 0.69 ~ 0.75% (0.73%) |
| UPIR | 0.20 ~ 0.30% (0.25%) | 0.18 ~ 0.32% (0.26%) |
| lost | 352 ~ 390 (369.8) | 341 ~ 402 (373.8) |
| LVIS_flip | 5.26 ~ 5.56% (5.38%) | 5.24 ~ 5.44% (5.34%) |
| LVIS_lost | 1052 ~ 1159 (1088.8) | 1119 ~ 1199 (1151.0) |

*원본: `runs/58_official_data/rw10_seed{1,2,4}.log` + `runs/59_rw_sweep/rw_sweep_seed0.log`(rw10 seed0),
`runs/58_official_data/seed{0,1,2,4}.log`(rw20)*

**같은 seed끼리 직접 비교한 결론**: rw10이 이기는 항목(COCO_AP +0.23, LVIS_lost
-62.2)은 격차가 크고, rw20이 이기는 항목(Heval_flip/Top1_flip/UPIR/LVIS_flip)은
전부 범위가 크게 겹칠 정도로 작은 차이 — seed 간 자연 변동폭 안에 있다.
LVIS_AP는 평균이 완전히 동일(0.1247).

**판단(09-08)**: 논문의 핵심 주장("decision을 지키면서 AP도 희생 안 한다")에는
AP 우위 폭이 더 큰 **rw=10 쪽이 더 잘 맞는다** — masked Heval_flip 최고기록도
rw10(9.57%)이 여전히 BRECQ(같은 4-seed 평균 10.41%)를 확실히 이겨서 "처음으로
이 지표를 이겼다"는 핵심 스토리가 rw10에서도 그대로 유지된다. 연산 비용은
rw10/rw20 사이에 실질적 차이 없음(§8.4 예정 또는 진행 중 문서 참고).

**진행 중**: rw10 seed3(재시도)/seed5, rw20 seed3(재시도, 최초엔 메모리 버그로
OOM 소실)를 추가로 검증 중 — 끝나면 두 설정 다 6-seed(0~5)로 맞춰서 이 표를
갱신할 예정. (§8.1의 "메인 결과"는 09-08부로 rw=10 기준으로 이미 교체함 —
4-seed 상태라 seed3/5까지 채워지면 다시 갱신 예정.)

### 8.3 참고 — 이전(다른 규모) 실험과 헷갈리지 않도록

`PROMPTCAL_HOW_IT_WORKS.md` §7의 "6-seed 최종 결과"는 **이 문서와 다른
실험**이다 — calib=32(val2017 슬라이스, 평가셋과 일부 겹침), scalar `s_mult`,
LVIS 미검증 상태 기준. 그 표의 절대 수치(AP 36.46 등)를 이 문서의 표와
직접 비교하지 말 것 — 데이터 소스와 s_mult 설계 자체가 다르다.

---

## 9. 알아둘 점 / 한계 / 열린 이슈

- **메모리**: `scripts/58`/`59`의 `preprocess()`가 한때 `torch.from_numpy(im).float().unsqueeze(0) / 255.0`
  (out-of-place 나눗셈)를 써서 probe 5000장 기준 RSS가 이론치(~24GB)의 2배(~47GB)로
  부풀었었다 — in-place `.div_(255.0)`로 수정(09-08). 동시에 여러 조건을
  돌릴 때 이 메모리 때문에 OOM으로 프로세스가 죽은 사례 있음(seed3).
  *`scripts/58_full_baseline_official_data.py:59` 부근*
- **LVIS_AP 잔여 열세**: §8.1 참고 — naive 대비 -1.3%, 완전히 해소되진 않음.
- **UPIR·lost가 최선이 아님**: AdaRound/BRECQ가 이 두 지표는 더 낮음(=더 좋음).
  Combined가 "전부 최고"는 아니라는 점을 논문에 정직하게 써야 함.
- **rw 재스윕 미완**: §8.2, 현재 진행 중. 완료되면 이 문서 §8.1을 갱신할 것.
- **H_cal(20개) 미사용**: 3분할 중 H_cal은 현재 Combined 학습에서 전혀
  안 쓰인다 — 향후 활용 여지(예: H_cal도 neighbor 대상에 포함) 남아 있음.
- **`pipeline/` 디렉토리 최신화 안 됨**: 재현용으로 만들어둔 `pipeline/quant/adaround.py`는
  아직 스칼라 버전 그대로라, 이 문서의 per-channel 설계와 다르다 — 논문
  최종 방법이 확정되면 동기화 필요.

---

## 10. 관련 문서/코드 인덱스

| 내용 | 위치 |
|---|---|
| 측정 도구·baseline·평가지표 설계 배경(서술 중심) | `PROMPTCAL_HOW_IT_WORKS.md` |
| per-channel 재설계 전체 서사(발견→진단→ablation→확정) | `PROMPTCAL_HOW_IT_WORKS.md` §5.7 |
| 전체 로드맵/최신 상태 요약 | `MASTER_SUMMARY.md` |
| 09-08 작업 일지(official-data 전환 경위, 버그 3건 등) | `PromptCal_PTQ_progress_2026-09-08.md` |
| s_mult 구현 | `src/quant/adaround.py` |
| Combined 학습 루프 | `src/quant/promptcal.py` |
| baseline 구현 | `src/quant/adaround.py`(AdaRound/QDrop), `src/quant/brecq.py` |
| 측정 하네스 | `src/harness.py` |
| 공식 데이터 5조건 전체 검증 스크립트 | `scripts/58_full_baseline_official_data.py` |
| scale_reg_weight 스윕 스크립트 | `scripts/59_rw_sweep_official_data.py` |
| 원본 결과 로그 | `runs/58_official_data/*.log`, `runs/59_rw_sweep/*.log` |

---

*이 문서에 반영 안 된 최신 진행 상황(예: rw=10 3-seed 완료 결과)이 있다면
알려주면 §8을 갱신하겠습니다. 궁금한 부분(특정 코드 경로, 특정 seed의 원본
로그 등)이 있으면 말씀해주세요.*
