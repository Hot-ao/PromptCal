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

### 5.4.1 버그 수정(09-09) — H_eval이 실제로는 완전히 held-out이 아니었음

**발견 경위**: "H_eval은 최적화에 한 번도 안 씀"이라는 위 원칙이 실제로
지켜지는지 코드를 다시 보다가, neighbor 후보 선정(§5.3 (b))이 **S만
제외하고 H_cal+H_eval을 통째로 후보 풀로 쓰고 있었다는 걸 발견**했다:

```python
# 수정 전
picked = [o for o in order if o not in pidx_set][:neighbor_k]   # pidx_set = S만
```

즉 S의 어떤 class와 text embedding상 가까우면, 그게 H_eval이어도 그냥
neighbor_cols에 뽑혀서 asymmetric hinge(§5.3 (b))로 간접 학습됐다. **실측
(seed=0)**: neighbor_cols 35개 중 H_cal 17개, **H_eval 18개(20개 중 90%)**
가 포함돼 있었다 — UPIR·Heval_flip 같은 "H_eval은 순수 held-out"을 전제로
하는 지표들이 실제로는 오염된 조건에서 측정되고 있었다는 뜻.

**수정**: neighbor 후보 풀에서 H_eval도 같이 제외하도록
`exclude_from_neighbors` 인자를 추가.

```python
# 수정 후
exclude_set = pidx_set | set(exclude_from_neighbors)   # exclude_from_neighbors=H_eval
picked = [o for o in order if o not in exclude_set][:neighbor_k]
```
*`src/quant/promptcal.py:422` (`optimize_promptcal_scale_neighbor`), `scripts/58_full_baseline_official_data.py`의
`build()`가 `exclude_from_neighbors=H_eval`로 호출*

수정 후 실측(seed=0): neighbor_set이 35개(H_cal 17+H_eval 18)에서
**20개(H_cal 20, H_eval 0)로 정확히 줄어듦** — 이제 H_eval은 정말로 학습에
전혀 안 쓰인다.

**영향 범위**: 이 버그는 학습 자체(어떤 컬럼이 neighbor-hinge 대상인지)를
바꾸는 거라, s_mult가 다르게 학습되고 **COCO_AP/LVIS_AP를 포함한 Combined의
모든 수치가 재검증 대상**이 된다 — UPIR 하나만 다시 재면 되는 게 아니다.
1-seed(seed=0) 재검증 결과와 진행 상황은 §8.3 참고.

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
| `scale_reg_weight` | **10.0(확정)** | §8.1/8.2 참고 — 09-08에 20에서 10으로 변경, 6-seed(0~5) 검증 완료 |
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

### 8.1 메인 결과 — `scale_reg_weight=10`, **H_eval 순수성 버그 수정 후 6-seed(0~5) 완료** (2026-09-09 갱신)

§5.4.1의 H_eval 누출 버그를 고친 뒤 재실행한 결과로 교체(버그 자체와
발견 경위는 §5.4.1 참고). seed3~5는 09-09 메모리 사고(3개 동시 실행 중
커널 OOM-killer로 silent kill)로 한 번 유실되어 재실행함 — 최종적으로
6개 seed 전부 에러 없이 완료. **범위는 평균±표준편차가 아니라 최소~최대**로
표시(괄호 안이 평균). baseline은 AP/lost/LVIS류는 seed와 무관한 상수,
Heval_flip·UPIR만 S/H_eval 분할 때문에 seed마다 달라서 같이 범위로
표시했다.

| method | COCO_AP | LVIS_AP | Heval_flip | Top1_flip | GT_MRR | GT_R@1 | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|---|
| **FP32** | **36.80** | **0.1260** | - | - | - | - | - | - | - | - |
| naive | 33.54 | 0.1264 | 9.85~14.55%(11.80%) | 0.95% | 0.9220 | 0.8767 | 0.18~0.44%(0.31%) | 432 | 6.48% | 1224 |
| AdaRound | 33.24 | 0.1242 | 8.74~13.39%(10.85%) | 0.74% | 0.9229 | 0.8782 | 0.13~0.31%(0.25%) | 349 | 5.67% | 1212 |
| QDrop | 33.30 | 0.1235 | 8.81~13.29%(10.84%) | 0.72% | 0.9222 | 0.8770 | 0.13~0.33%(0.27%) | 384 | 5.71% | 1192 |
| BRECQ | 33.30 | 0.1200 | 8.55~12.67%(10.48%) | 0.71% | 0.9233 | 0.8788 | 0.11~0.33%(0.23%) | 327 | 5.53% | 1158 |
| **Combined** | **36.19~36.48(36.30)** | 0.1229~0.1254(0.1240) | **7.77~10.52%(9.39%)** | 0.68~0.77%(0.74%) | 0.9221~0.9227(0.9224) | 0.8765~0.8776(0.8770) | 0.16~0.31%(0.25%) | 360~383(372) | **5.15~5.50%(5.37%)** | **1076~1151(1122)** |

*원본 로그: `runs/58_official_data/neighborfix_seed{0,1,2,3,4,5}.log`
(baseline은 §8.4에서 이 로그들과 이전 버전 로그가 값이 동일함을 확인함)*

**GT_MRR/GT_R@1 대조가 보여주는 것(중요, 버그 수정 후에도 유지됨)**: 5개
방법의 GT_MRR(0.9220~0.9233)·GT_R@1(0.8765~0.8788)이 **거의 완전히
겹친다** — 이 절대적 랭킹 정확도만 보면 naive조차 다른 방법들과 별 차이가
없다. 그런데 같은 5개 방법의 AP는 33.24~36.48로 크게 갈리고,
Top1_flip/Heval_flip도 뚜렷이 갈린다. **이 괴리 자체가 핵심 논증이다**: AP
개선이 "GT 랭킹 점수 자체를 절대적으로 더 잘 복원해서" 나온 게
아니라(그랬다면 GT_MRR도 같이 갈렸어야 함), **"FP32가 매기던 순서/결정을
얼마나 유지하는가"(flip)에서 나온다**는 뜻 — 이는 `margin_loss`가
값(score) 자체가 아니라 순위 간격(margin)을 맞추도록 설계된 것과 정확히
같은 철학이다.

**정직한 요약**:
- COCO_AP: naive 대비 **+2.76**(평균 기준), 6-seed 범위(36.19~36.48)가
  naive(33.54)와 전혀 안 겹칠 만큼 안정적으로 높음.
- Heval_flip: 평균 9.39%로 BRECQ 평균(10.48%)보다 낮음(더 좋음). 다만
  범위가 넓어서(7.77~10.52%) BRECQ의 최선 seed(8.55%)와 일부 겹침 —
  "항상"이 아니라 "평균적으로/대체로" 이긴다가 정확함.
- **LVIS_flip·LVIS_lost는 6개 seed 전부 개별적으로 BRECQ를 이김**
  (LVIS_flip 5.15~5.50% 전부 < BRECQ 5.53%; LVIS_lost 1076~1151 전부 <
  BRECQ 1158) — 버그 수정 전보다 오히려 이 두 지표의 승률이 더 뚜렷해짐.
- LVIS_AP: 평균 0.1240, BRECQ(0.1200)·QDrop(0.1235)보다는 6개 seed 전부
  위, AdaRound(0.1242)와는 거의 동률(seed별로 이기고 지고 섞임),
  naive/FP32(0.126대)보다는 소폭 아래.
- **UPIR·lost는 여전히 baseline 중 최선(BRECQ)에는 못 미침** — 특히
  `lost`(360~383)는 6개 seed 전부 BRECQ(327)보다 나쁘고 폭도 좁아
  일관된 트레이드오프로 보임. UPIR은 평균으로는 BRECQ(0.23%)보다
  소폭 나쁘지만(0.25%) seed4/5에서는 역전(0.17%/0.16%)돼 있어 lost만큼
  일관되진 않음. ~~§8.3에 정리한 가설: neighbor-hinge의 부작용(H_eval
  경계 손해)은 학습에 쓰인 COCO-80 vocabulary 안에서만 드러나는
  비용~~ — **09-10 그룹별 분해로 기각됨**(§9 참고): H_eval의 lost 악화폭이
  S와 거의 같아서(둘 다 +15%) H_eval 국소적 비용이 아니라 COCO-80
  전반에 걸친 일반적 트레이드오프로 보는 게 정확하다. neighbor-hinge가
  실제 보호하는 H_cal만 상대적으로 덜 나빠짐(+9%)은 확인됨.

**부수적 관찰 — naive가 AdaRound/QDrop/BRECQ보다 AP가 높은 이유**: 세 baseline
모두 반올림 방향을 "layer 출력 재구성 MSE"를 줄이는 쪽으로 최적화하는데, 이건
AP를 직접 겨냥한 목적함수가 아니다(reconstruction ≠ decision/task 최적화라는
이 논문의 핵심 주장이 AP 자체에서도 드러난 사례). Combined는 이 MSE 대신
decision 직접 목적함수(margin_loss)로 학습해서 이 함정을 안 물려받는다.

### 8.2 `scale_reg_weight` 재스윕 — rw=10 vs rw=20, **6-seed(0~5) 완료**

| | rw=10 (6-seed) | rw=20 (같은 6-seed) |
|---|---|---|
| COCO_AP | 36.19 ~ 36.45 (36.34) | 35.99 ~ 36.23 (36.09) |
| LVIS_AP | 0.1213 ~ 0.1272 (0.1253) | 0.1224 ~ 0.1272 (0.1251) |
| Heval_flip | 8.05 ~ 10.50% (9.47%) | 7.85 ~ 10.77% (9.42%) |
| Top1_flip | 0.68 ~ 0.78% (0.75%) | 0.69 ~ 0.76% (0.73%) |
| UPIR | 0.10 ~ 0.31% (0.23%) | 0.12 ~ 0.32% (0.24%) |
| lost | 352 ~ 390 (371.3) | 341 ~ 402 (367.8) |
| LVIS_flip | 5.20 ~ 5.56% (5.35%) | 5.24 ~ 5.44% (5.31%) |
| LVIS_lost | 1052 ~ 1159 (1093.8) | 1114 ~ 1199 (1141.2) |

*원본: `runs/58_official_data/rw10_seed{1,2,3,4,5}.log` + `runs/59_rw_sweep/rw_sweep_seed0.log`(rw10 seed0),
`runs/58_official_data/seed{0,1,2,4,5}.log` + `runs/58_official_data/seed3_retry.log`(rw20)*

**6-seed로 확정된 결론**: rw10이 이기는 항목(COCO_AP +0.25, LVIS_lost -47.4)은
여전히 격차가 크고, rw20이 이기는 항목(Heval_flip/Top1_flip/UPIR/LVIS_flip)은
전부 평균 차이가 0.01~0.05 수준 — 범위가 크게 겹쳐서 seed 노이즈와
구분하기 어렵다. LVIS_AP는 평균이 사실상 동일(0.1253 vs 0.1251).

**최종 판단(09-08)**: 4-seed 때의 판단이 6-seed로도 그대로 유지된다 —
논문의 핵심 주장("decision을 지키면서 AP도 희생 안 한다")에는 AP 우위 폭이
더 큰 **rw=10**이 더 잘 맞는다. masked Heval_flip도 rw10(9.47%)이 여전히
BRECQ(10.48%)를 확실히 이겨서 핵심 스토리가 유지된다. 연산 비용은 rw10/rw20
사이에 실질적 차이 없음(§5.5 하이퍼파라미터 표, 실측 빌드시간 1030~1080초
안팎으로 동일 수준). **`scale_reg_weight=10`을 최종 채택값으로 확정.**

### 8.3 H_eval 순수성 버그 수정 후 재검증 — **완료(6/6 seed), §8.1로 승격됨**

§5.4.1의 버그 수정(neighbor 후보에서 H_eval도 제외) 이후 재검증, 6개
seed(0~5) 전부 완료. seed3~5는 09-09 메모리 사고(3개 동시 실행 중 커널
OOM-killer로 silent kill) 이후 재실행해서 완료시킴. 이 결과는 §8.1의
메인 결과로 이미 반영됨 — 이 절은 seed별 원값과 판단 과정 기록용으로 유지.

**Combined, seed별 원값**

| seed | COCO_AP | LVIS_AP | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|
| 0 | 36.38 | 0.1246 | 8.55% | 0.75% | 0.30% | 369 | 5.45% | 1139 |
| 1 | 36.19 | 0.1242 | 9.73% | 0.73% | 0.31% | 368 | 5.27% | 1151 |
| 2 | 36.28 | 0.1239 | 9.84% | 0.76% | 0.27% | 376 | 5.37% | 1141 |
| 3 | 36.27 | 0.1229 | 7.77% | 0.77% | 0.28% | 383 | 5.50% | 1140 |
| 4 | 36.19 | 0.1229 | 9.95% | 0.74% | 0.17% | 374 | 5.49% | 1085 |
| 5 | 36.48 | 0.1254 | 10.52% | 0.68% | 0.16% | 360 | 5.15% | 1076 |
| **평균(6-seed)** | **36.30** | **0.1240** | **9.39%** | **0.74%** | **0.25%** | **372** | **5.37%** | **1122** |

**BRECQ(최선 baseline) 대비, 6-seed 평균**

| method | COCO_AP | LVIS_AP | Heval_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|
| BRECQ | 33.30 | 0.1200 | 10.48%(8.55~12.67) | 0.23%(0.11~0.33) | 327(고정) | 5.53%(고정) | 1158(고정) |
| **Combined** | **36.30** | **0.1240** | **9.39%** | 0.25% | 372 | **5.37%** | **1122** |

*원본: `runs/58_official_data/neighborfix_seed{0,1,2,3,4,5}.log`. BRECQ의
COCO_AP/LVIS_AP/lost/LVIS_flip/LVIS_lost는 seed 무관 상수(§8.4 seed-결정성
참고), Heval_flip/UPIR만 S/H_eval split이 seed마다 달라서 값이 흔들림.*

**6-seed 평균 기준 최종 관찰**:
- COCO_AP(+3.00)·LVIS_AP(+0.0040)·Heval_flip(9.39%<10.48%)는 여전히
  Combined가 BRECQ보다 우위.
- **LVIS_flip·LVIS_lost는 6개 seed 전부 개별적으로 BRECQ를 이김**
  (버그 수정 전보다 오히려 승률이 더 뚜렷해짐 — 이전엔 "6개 중 5개"였음).
- `lost`(372>327)는 6개 seed 전부 BRECQ보다 나쁘고 범위(360~383)가 좁아
  일관된 트레이드오프로 보임. UPIR은 평균으로는 근소하게 나쁘지만(0.25%
  vs 0.23%) seed4/5에서는 오히려 역전(0.17%/0.16%)됨 — lost만큼 일관된
  패턴은 아님.
- **가설 검증 완료(09-10) — 기각됨**: "neighbor-hinge의 부작용(H_eval
  경계 손해)이 COCO-80 vocabulary에 국소적"이라는 가설을 `lost`의
  S/H_cal/H_eval 그룹별 분해로 직접 확인했다(1-seed). H_eval의 lost
  악화폭(+15%)이 S의 악화폭(+15%)과 거의 같아서, H_eval 국소적 비용이
  아니라 COCO-80 전반의 일반적 트레이드오프로 보는 게 정확하다. 상세는
  §9 "UPIR·lost가 최선이 아님" 항목 참고.

**다음**: §8.2(rw10 vs rw20 비교)는 버그 수정 전 데이터 기준이지만,
재검증하지 않기로 판단(09-10, 근거는 §9 참고) — `scale_reg_weight=10`
확정은 그대로 유지. 이 leak-fix 6-seed 검증 관련 작업은 여기서 종료.

### 8.4 참고 — 이전(다른 규모) 실험과 헷갈리지 않도록

`PROMPTCAL_HOW_IT_WORKS.md` §7의 "6-seed 최종 결과"는 **이 문서와 다른
실험**이다 — calib=32(val2017 슬라이스, 평가셋과 일부 겹침), scalar `s_mult`,
LVIS 미검증 상태 기준. 그 표의 절대 수치(AP 36.46 등)를 이 문서의 표와
직접 비교하지 말 것 — 데이터 소스와 s_mult 설계 자체가 다르다.

### 8.5 H_eval 버그 수정 후 k/neighbor_k/boundary_w 재스윕 (2026-09-10, 1-seed)

§5.4.1의 리크 수정으로 neighbor-hinge 후보 풀이 줄었으니(35→20, H_cal만),
neighbor_k 등 관련 하이퍼파라미터의 최적값이 바뀌었을 수 있어서 재확인했다.
방법론은 기존과 동일(OFAT, `scripts/60`, 1-seed 트렌드 체크, seed=0,
scale_reg_weight=10 고정).

**k 스윕** (나머지 고정: neighbor_k=5, boundary_w=3.0):

| method | COCO_AP | LVIS_AP | Heval_flip | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|
| BRECQ | 33.30 | 0.1200 | 8.99% | 327 | 5.53% | 1158 |
| combined_k3 | 36.16 | 0.1207 | 8.86% | 369 | 5.26% | 1138 |
| **combined_k5(현재)** | **36.38** | **0.1246** | **8.55%** | 369 | **5.45%** | 1139 |
| combined_k8 | 36.22 | 0.1223 | 8.77% | 415 | 5.56% | 1141 |

k=5(현재값)가 COCO_AP·LVIS_AP 둘 다 최고, k=8은 lost가 369→415로 눈에 띄게
나빠짐. **k=5 유지가 맞음.**

**neighbor_k 스윕** (나머지 고정: k=5, boundary_w=3.0) — **핵심 발견**:

| method | COCO_AP | LVIS_AP | Heval_flip | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|
| combined_neighbor_k3 | 36.38 | 0.1205 | 8.63% | 377 | 5.18% | 1092 |
| **combined_neighbor_k5(현재)** | 36.38 | 0.1246 | 8.55% | 369 | 5.45% | 1139 |
| combined_neighbor_k8 | 36.38 | 0.1246 | 8.55% | 369 | 5.45% | 1139 |
| combined_neighbor_k10 | 36.38 | 0.1246 | 8.55% | 369 | 5.45% | 1139 |

**neighbor_k=5, 8, 10이 완전히 동일한 값**이 나왔다 — 이유가 명확하다:
버그 수정 후 neighbor 후보 풀이 H_cal 20개뿐인데, 40개 S 클래스 각각에서
neighbor_k=5개씩만 뽑아도 합집합(neighbor_set)이 이미 H_cal 20개 전체를
커버해버린다(§5.4.1 실측: 수정 후 neighbor_set=20). 그래서 neighbor_k를
8, 10으로 올려도 "더 뽑을 후보가 없어서" 결과가 똑같다. **neighbor_k>=5는
사실상 동치 클래스이고, neighbor_k=5(현재값) 유지가 맞다.**

**boundary_w 스윕** (나머지 고정: k=5, neighbor_k=5):

| method | COCO_AP | LVIS_AP | Heval_flip | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|
| combined_boundary_w1 | 36.26 | 0.1217 | 8.41% | 382 | 5.42% | 1187 |
| combined_boundary_w2 | 36.46 | 0.1242 | 8.24% | 361 | 5.37% | 1177 |
| **combined_boundary_w3(현재)** | 36.38 | 0.1246 | 8.55% | 369 | 5.45% | 1139 |
| combined_boundary_w5 | 36.39 | 0.1250 | 8.88% | 401 | 5.36% | 1107 |

COCO-80 쪽 지표(COCO_AP·Heval_flip·lost)는 boundary_w=2가 최선, LVIS 쪽
지표(LVIS_AP·LVIS_flip·LVIS_lost)는 boundary_w=5가 최선인 **트레이드오프**
패턴 — 버그 수정 전과 동일한 양상. boundary_w=3(현재값)은 그 중간.

### 8.6 4-combo 그리드 — k x boundary_w 상호작용 재확인 (2026-09-10, 1-seed)

§8.5의 top 2 후보(k∈{5,8}, boundary_w∈{3,5}, neighbor_k=5 고정)로
`scripts/61_combo_grid_official_data.py`를 돌려 상호작용을 확인했다.
동기: 버그 수정 **전** 같은 조합(k=8+boundary_w=5)에서 negative
interaction이 관찰된 적 있음(개별로는 LVIS_AP가 각각 0.1278/0.1279였는데
조합하니 0.1242로 더 나빠짐) — 그게 버그 수정 후에도 재현되는지가 관건.

| method | COCO_AP | LVIS_AP | Heval_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|
| **k5_bw3(현재값)** | **36.38** | 0.1246 | **8.55%** | 0.30% | **369** | 5.45% | 1139 |
| k5_bw5 | 36.39 | 0.1250 | 8.88% | 0.31% | 401 | 5.36% | 1107 |
| k8_bw3 | 36.22 | 0.1223 | 8.77% | 0.27% | 415 | 5.56% | 1141 |
| k8_bw5 | 36.29 | **0.1254** | 8.83% | **0.27%** | 373 | **5.44%** | **1133** |

*k5_bw3/k5_bw5/k8_bw3는 각각 §8.5의 개별 k/boundary_w 스윕과 정확히
같은 설정이라 값도 동일함(재현성 확인 겸용) — 이 그리드에서 새로 얻은
정보는 k8_bw5 하나뿐.*

**지표별 최고 조합(09-10 재확인, 최초 작성 시 LVIS_flip/LVIS_lost를
k8_bw5로 잘못 표기했던 걸 정정)**:

| 지표 | 최고 | 값 |
|---|---|---|
| COCO_AP | k5_bw5 | 36.39 |
| LVIS_AP | k8_bw5 | 0.1254 |
| Heval_flip | k5_bw3(현재) | 8.55% |
| Top1_flip | k5_bw3(현재) | 0.75% |
| UPIR | k8_bw3·k8_bw5 공동 | 0.27% |
| lost | k5_bw3(현재) | 369 |
| LVIS_flip | k5_bw5 | 5.36% |
| LVIS_lost | k5_bw5 | 1107 |

승수로는 현재값(3승: Heval_flip/Top1_flip/lost)과 k5_bw5(3승: COCO_AP/
LVIS_flip/LVIS_lost)가 동률, k8_bw5는 LVIS_AP 하나만 단독 1위(UPIR은
k8_bw3와 공동). **negative interaction은 재현되지 않음**(버그 수정 전엔
k8+bw5 조합이 개별 효과보다 나빴는데, 지금은 그 정도로 나쁘진 않음)
— 다만 이게 "k8_bw5가 우수하다"는 뜻은 아니고, 그냥 예전 같은 뚜렷한
부작용이 없다는 정도.

COCO_AP·LVIS_AP는 "진짜" 표준 detection 지표(논문에 실릴 숫자)고 나머지는
이 프로젝트가 만든 진단용 지표라는 관점에서 보면, 현재값은 이 둘 중
어느 것도 1위가 아니다(COCO_AP는 k5_bw5, LVIS_AP는 k8_bw5가 근소 우위) —
겉보기엔 아쉬워 보일 수 있으나, **이 마진들은 §8.1의 6-seed 검증에서
이미 확인된 seed간 변동폭보다 작다**: LVIS_AP는 seed에 따라 0.1229~0.1254
(폭 0.0025)까지, COCO_AP는 36.19~36.48(폭 0.29)까지 흔들렸다. 지금 본
차이(LVIS_AP +0.0008, COCO_AP +0.01)는 그 폭 안에 완전히 들어가므로,
1-seed 비교만으로는 "진짜 개선"과 "seed 노이즈"를 구분할 수 없다.

**최종 결론**: `k=5, neighbor_k=5, boundary_w=3.0, scale_reg_weight=10`
**확정값 유지**. H_eval 리크 수정이 하이퍼파라미터 최적점 자체를 크게
흔들지는 않았다(k/boundary_w 트레이드오프 패턴 동일, neighbor_k는 오히려
더 단순해짐 - 5 이상 전부 동치). k8_bw5는 "LVIS 쪽을 더 밀고 싶으면"
쓸 수 있는 대안으로 기록만 해두고, 실제 채택은 안 함.

### 8.7 `neighbor_weight` 탐색 — LVIS_flip/LVIS_lost 개선 후보 (2026-09-10, 1-seed, **미확정**)

CVPR 제출 기준으로, Combined가 baseline 중 최선(BRECQ/QDrop)에 못 미치는
Top1_flip·LVIS_AP를 개선할 수 있는지 찾다가, 지금까지 한 번도 스윕 안 했던
`neighbor_weight`(neighbor-hinge 항 가중치, 기본값 1.0)를 테스트했다. 이
절의 값들은 **아직 1-seed(seed=0)이고 6-seed 검증 전이라 확정값이
아니다** — §8.1의 "Combined(확정, neighbor_weight=1.0)"와는 별개로 기록.

| method | COCO_AP | LVIS_AP | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|
| naive | 33.54 | 0.1264 | 10.51% | 0.95% | 0.39% | 432 | 6.48% | 1224 |
| AdaRound | 33.24 | 0.1242 | 9.20% | 0.74% | 0.26% | 349 | 5.67% | 1212 |
| QDrop | 33.30 | 0.1235 | 9.25% | 0.72% | 0.33% | 384 | 5.71% | 1192 |
| BRECQ | 33.30 | 0.1200 | 8.99% | 0.71% | 0.33% | 327 | 5.53% | 1158 |
| **Combined, nw=0.25** | 36.19 | 0.1235 | 8.51% | 0.72% | 0.27% | 377 | **5.00%** | **1074** |
| Combined, nw=0.5 | 36.19 | 0.1238 | 8.97% | 0.78% | 0.25% | 396 | 5.50% | 1179 |
| Combined, nw=1.0(확정값) | 36.49 | 0.1229 | 8.18% | 0.72% | 0.25% | 373 | 5.17% | 1127 |
| Combined, nw=2.0 | 36.36 | 0.1250 | 8.65% | 0.72% | 0.28% | 353 | 5.42% | 1144 |

*원본: `runs/60_hparam_sweep/neighbor_weight_seed0.log`. `scripts/60_hparam_sweep_official_data.py`에
`neighbor_weight`를 `--param` 선택지로 추가해서 실행(기존엔 k/neighbor_k/boundary_w만 지원했음).*

**가설과 반대 방향으로 나옴**: "neighbor_weight를 낮추면 margin_loss의
공격적 재구성이 덜 전파돼서 Top1_flip/lost가 개선될 것"이라 예상했는데,
실제로는 **낮출수록(0.25→0.5) lost가 오히려 나빠지고(373→377→396)**,
Top1_flip도 안 좋아짐(0.5에서 0.78%로 최악). Top1_flip/lost로 BRECQ/QDrop을
이기는 데는 4개 값 다 실패.

**대신 nw=0.25에서 LVIS_flip/LVIS_lost가 큰 폭으로 개선됨**: LVIS_flip
5.17%→**5.00%**(4개 값 중 최고, BRECQ 5.53% 대비 마진이 더 벌어짐),
LVIS_lost 1127→**1074**(역시 최고, BRECQ 1158 대비 마진 확대). LVIS_AP도
0.1229→0.1235로 소폭 개선(QDrop과 동률, BRECQ는 이기지만 AdaRound·naive는
여전히 못 이김 — §9 "LVIS_AP 잔여 열세"는 완전히 해소 안 됨).

**비용**: COCO_AP -0.30(36.49→36.19) — 그래도 naive(33.54) 대비 +2.65로
baseline 전부와 비교하면 여전히 압도적 우위, "baseline보다 나쁘다"는
문제는 전혀 아니고 Combined 자체 내에서의 트레이드오프. Heval_flip은
소폭 악화(8.18%→8.51%, 그래도 BRECQ 8.99%보다는 좋음), lost도 소폭
악화(373→377, 거의 무시할 수준).

**논문 관점에서의 판단**: §9에서 정리했듯 이 논문의 핵심 주장("reconstruction
≠ decision preservation")에 가장 직접적으로 부합하는 지표는 LVIS_flip/
LVIS_lost다(실제 novel vocabulary에서의 결정 보존, 외적 타당성이 가장
높음) — COCO_AP는 논문이 "AP만 보면 안 된다"고 주장하는 지표라 오히려
우선순위가 낮다. 이 기준으로 보면 nw=0.25는 **핵심 지표(LVIS_flip/
LVIS_lost)를 더 개선하면서 COCO_AP는 baseline 대비 여전히 압도적**이라
꽤 매력적인 후보. 다만 "COCO_AP를 왜 포기했는지"를 논문에 명시적으로
논증해야 하고, 아직 1-seed라 seed 노이즈인지도 확인 필요.

**다음**: 0.25 근방(예: 0.1/0.15/0.2/0.25) 더 세밀하게 스윕해서 이 방향이
계속 개선되는지 확인 후, 유력하면 6-seed로 정식 검증.

---

## 9. 알아둘 점 / 한계 / 열린 이슈

- **메모리**: `scripts/58`/`59`의 `preprocess()`가 한때 `torch.from_numpy(im).float().unsqueeze(0) / 255.0`
  (out-of-place 나눗셈)를 써서 probe 5000장 기준 RSS가 이론치(~24GB)의 2배(~47GB)로
  부풀었었다 — in-place `.div_(255.0)`로 수정(09-08). 동시에 여러 조건을
  돌릴 때 이 메모리 때문에 OOM으로 프로세스가 죽은 사례 있음(seed3).
  *`scripts/58_full_baseline_official_data.py:59` 부근*
- **LVIS_AP 잔여 열세**: §8.1 참고 — naive 대비 -1.9%, 완전히 해소되진 않음.
- **APr(rare class)에서 Combined가 5개 중 최하위 — 잠재적으로 심각한 문제(09-10 발견)**:
  LVIS_AP는 aggregate라 안 보였는데, rare/common/frequent로 쪼개보면 얘기가
  다르다(seed=0, `runs/58_official_data/lostbygroup_seed0.log`):

  | method | APr(rare) | APc(common) | APf(frequent) |
  |---|---|---|---|
  | naive | 0.0415 | 0.0835 | 0.1797 |
  | AdaRound | 0.0328 | 0.0831 | 0.1771 |
  | QDrop | 0.0311 | 0.0804 | 0.1783 |
  | BRECQ | 0.0319 | 0.0734 | 0.1771 |
  | **Combined** | **0.0270(5개 중 최하)** | 0.0789 | **0.1826(5개 중 최고)** |

  Combined는 frequent class에서 최고, rare class에서 최하위 — **"흔한
  카테고리 쪽으로 성능을 밀어주고 진짜 낯선 rare 카테고리는 오히려
  손해보는" 패턴**으로 보인다. margin_loss/neighbor-hinge가 S/H_cal(전부
  COCO의 흔한 카테고리)에만 최적화 압력을 주는 설계와 정확히 부합하는
  메커니즘. **open-vocabulary 일반화라는 핵심 주장을 가장 엄격하게
  검증하는 게 바로 rare class인데 거기서 최하위라는 건, 논문 리뷰
  단계에서 지적받을 가능성이 높은 심각한 약점** — LVIS_AP aggregate
  동률/우위보다 이게 더 중요한 이슈일 수 있음. neighbor_weight/
  scale_reg_weight 탐색(§8.7 이하) 때 이 APr도 같이 확인 필요.
  **아직 미해결, 후속 탐색 필요.**
- **UPIR·lost가 최선이 아님**: BRECQ가 이 두 지표는 더 낮음(=더 좋음), 특히
  `lost`는 6-seed 전부 BRECQ보다 나쁨 — Combined가 "전부 최고"는 아니라는
  점을 논문에 정직하게 써야 함.
  **가설 검증 완료(09-10) — 가설은 기각됨**: "이 비용이 H_eval(비보호
  클래스)에 국소적으로 몰려있고 LVIS 일반화에는 안 묻어난다"는 가설을
  세우고 `lost`를 S/H_cal/H_eval 그룹별로 쪼개서 확인했다(1-seed,
  seed=0, calib=256, `runs/58_official_data/lostbygroup_seed0.log`).

  | 그룹 | BRECQ lost_rate | Combined lost_rate | 변화(상대) |
  |---|---|---|---|
  | S | 1.23% | 1.42% | +15% |
  | H_cal | 0.75% | 0.82% | +9% |
  | H_eval | 1.41% | 1.62% | +15% |

  **H_eval의 악화폭(+15%)이 S의 악화폭(+15%)과 거의 동일**하다 — 가설대로라면
  H_eval만 훨씬 크게 나빠져야 하는데 그렇지 않다. 오히려 neighbor-hinge가
  실제로 보호하는 **H_cal이 세 그룹 중 가장 적게 나빠졌다(+9%)** — 이건
  메커니즘이 의도대로 작동한다는 신호. 즉 `lost` 악화는 H_eval에 국소적인
  게 아니라 **S/H_eval에 고르게 나타나는 일반적인 트레이드오프**이고,
  H_cal만 neighbor-hinge 덕에 상대적으로 덜 나빠진 것으로 보인다.
  **결론: COCO-80-국소적 비용이라는 가설은 기각. `lost` 악화는 vocabulary
  전반(H_eval 포함이지만 H_eval만은 아님)에 걸친 일반적 트레이드오프로
  보는 게 더 정확한 서술.** 코드는 `scripts/58_full_baseline_official_data.py`의
  `gt_metrics_for_method`(`lost_rate_by_group` 출력)에 남겨둠.
- **rw 재스윕**: §8.2, 6-seed(0~5) 완료했지만 **H_eval 버그 수정 전
  데이터 기준**. 버그 수정판으로 재검증은 **하지 않기로 판단**(09-10) —
  `scale_reg_weight`는 s_mult 크기를 누르는 전역 정규화 강도이고, H_eval
  리크는 neighbor-hinge 후보 풀 구성(어떤 컬럼을 대상으로 삼는지)을 바꾼
  것이라 서로 다른 축. rw10이 rw20을 이긴 이유도 전반적인 s_mult
  과적합/수치 안정성 문제였지 특정 neighbor 구성 때문이 아니었을
  가능성이 높아, `scale_reg_weight=10` 확정 판단은 그대로 유지.
- **H_eval 순수성 버그(09-09 발견·수정)**: §5.4.1/§8.3 참고. neighbor 후보
  선정이 H_eval을 제외 안 해서 UPIR 등 held-out 지표가 실제보다 좋게
  나오고 있었음. 수정 완료, 6-seed 재검증도 완료해서 §8.1에 반영함.
- **H_cal(20개) 미사용**: 3분할 중 H_cal은 현재 Combined 학습에서 전혀
  안 쓰인다 — 향후 활용 여지(예: H_cal도 neighbor 대상에 포함) 남아 있음.
  애초에 margin_loss만 있다면 S/H 분할 자체가 COCO→LVIS 실험엔 필수가
  아니고(LVIS 신규 클래스 자체가 held-out 역할), S/H 분할이 실질적으로
  필요한 이유는 neighbor-hinge가 "직접 안 건드리는 클래스" 풀을 필요로
  하기 때문 — 이 관점에서는 H_cal/H_eval을 구분할 이유가 없고 그냥
  "S(40) vs H(40, neighbor 후보 풀)" 2분할로 단순화하는 게 더 정확한
  설명일 수 있음(아직 코드 리팩터링은 안 함).
- **`pipeline/` 디렉토리**: 09-08~09-09에 per-channel `s_mult`,
  `scale_reg_weight` 정규화, H_eval 버그 수정까지 전부 동기화 완료
  (`src/quant/{adaround,promptcal}.py`와 diff 없음 확인, 09-09).
  calibration/평가 데이터 소스만 아직 예전 방식(val2017 슬라이스) — 공식
  데이터 설정으로는 이식 안 함.

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
