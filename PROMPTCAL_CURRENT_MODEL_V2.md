# Combined 모델 현재 설계 전체 문서 v2 (2026-09-17 작성)

**이 문서가 지금부터의 단일 진실 공급원이다.** [`PROMPTCAL_CURRENT_MODEL.md`](PROMPTCAL_CURRENT_MODEL.md)
(v1, 2026-09-08~09-16 작성)는 per-channel `s_mult` 설계를 중심으로 쓰였고
그 뒤로 수많은 패치(09-09 H_eval 버그, 09-10~09-12 하이퍼파라미터 스윕
10개 절, 09-16 per-tensor+identity-aware 전환)가 쌓여서 지금 실제 설계를
파악하기엔 너무 두껍다 — 이 v2는 **지금(09-17) 확정된 설계만** 처음부터
깔끔하게 다시 쓴 것이다. 과거 스윕의 세부 근거·claim 1~10의 검증 경위가
필요하면 v1과 [`PROMPTCAL_CLAIMS_2026-09-15.md`](PROMPTCAL_CLAIMS_2026-09-15.md)를
감사(audit) 기록으로 참고할 것 — 이 v2는 그 결과만 반영한다.

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
1. FP32 모델 forward + SimilarityHarness로 cv4 유사도 행렬 캡처       (§3)
2. wrap_convs + calibrate                    → naive W8A8            (§4)
3. convert_to_adaround + optimize_adaround(1000 iter)
     → weight 반올림(alpha) 확정, hard-round                         (§5.1)
4. optimize_promptcal_scale_neighbor(1500 iter)
     → alpha는 완전히 고정(requires_grad=False)
     → conv당 per-tensor s_mult(활성화 scale 배율) 하나만
       identity-aware margin_loss(S + H_cal) + asymmetric neighbor
       hinge + scale_reg_weight로 학습                                (§5)
```
*`pipeline/run_comparison.py`의 `build()`가 이 4단계를 그대로 호출한다.
플래그 없이 실행하면 이 설계(§5.5 하이퍼파라미터)가 기본값이다.*

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
전체 실험의 공통 연산이다. 이 파일은 이번 설계 전환(per-tensor·
identity-aware) 기간 내내 변경 없음.

---

## 4. Baseline 4가지 (Combined와의 비교 기준)

| baseline | 방식 | 코드 |
|---|---|---|
| naive | round-to-nearest, 추가 최적화 없음 | `src/quant/fake_quant.py` |
| AdaRound | layer별 출력 재구성 MSE로 반올림(alpha) 학습 | `src/quant/adaround.py`, `optimize_adaround` |
| QDrop | AdaRound + 확률적 activation drop(`qdrop_prob`) | `src/quant/adaround.py` (같은 함수, 인자만 다름) |
| BRECQ | block 단위 joint 재구성(층 간 상관 반영) | `src/quant/brecq.py:35` (`optimize_brecq`) |

4개 전부 activation quantization이 **per-tensor**(scalar `scale`/`zero_point`,
`ActObserver` 참고)다 — Combined의 s_mult가 지금 per-tensor인 이유(§5.2)가
바로 이 4개와의 공정한 비교.

---

## 5. Combined 모델 상세 (현재 확정 설계)

### 5.1 1단계 — AdaRound weight rounding (그대로 재사용)

```python
w_int = floor(w/scale) + h(alpha)     # h(alpha) ∈ [0,1], rectified sigmoid
```
*`src/quant/adaround.py:44` (`class AdaRoundQuantConv2d`)*

`optimize_adaround(m.model, fp.model, calib, device, iters=1000)`로 학습 후
hard-round 확정. Combined는 이 결과를 **그대로 고정**하고 2단계로 넘어간다
(`ac.alpha.requires_grad_(False)`).

### 5.2 2단계 — `s_mult`: **per-tensor** learnable activation scale

```python
self.s_mult = nn.Parameter(torch.tensor(1.0, device=self.conv.weight.device))
                                       # 초기값 1.0, conv당 스칼라 하나(0-dim)
self.use_smult = False                # True일 때만 적용

def _quantize_smult(self, x):
    mult = self.s_mult.clamp(min=0.1, max=10.0).view(1, -1, 1, 1)
    scale = self.a_obs.scale * mult
    x_s = x / scale
    x_r = x_s + (round(x_s) - x_s).detach()   # round만 STE, scale은 grad 유지
    x_c = clamp(x_r + zp, qmin, qmax)
    return (x_c - zp) * scale
```
*`src/quant/adaround.py:46` (constructor, `channelwise_smult` 파라미터로 토글),
`src/quant/adaround.py:110` (`_quantize_smult`)*

`round()`만 straight-through estimator로 처리하고 `scale` 자체는 계산
그래프에 남겨 `s_mult`로 grad가 흐르게 한다 — "이 conv 활성화 양자화 격자의
눈금 크기"를 연속적으로 학습하는 것.

**왜 per-channel이 아니라 per-tensor인가**: 09-07에 원래 스칼라였던
`s_mult`를 per-channel 벡터(conv 입력 채널 수만큼)로 재설계했었다 — 스칼라
하나로는 "S(학습 대상)의 이웃만 조준해서 억제"하려는 neighbor loss(§5.3)의
의도가 구조적으로 불가능해서, calibration에서 존재도 몰랐던 LVIS
클래스까지 무차별적으로 위축시키는 문제가 있었기 때문이다(전체 서사는
v1 §5.2 참고). 그런데 baseline 4개는 전부 per-tensor로 양자화하는데
Combined만 `s_mult`로 per-channel 자유도를 추가로 쓰는 건 **baseline과의
unisolated confound**이고, 표준 INT8 추론 엔진 대부분이 per-channel
activation dequant를 지원하지 않아서 **실제 배포 시 이 이점을 그대로 못
쓴다**는 문제가 있었다(`PROMPTCAL_CLAIMS_2026-09-15.md` claim4). 09-16에
per-tensor로 되돌리고 6-seed로 검증한 결과 baseline 대비 우위는 견고하게
유지됨을 확인해서(§8.1) 채택했다 — **성능이 좋아져서가 아니라 공정성·배포
가능성을 위한 방법론적 필연**이다. per-channel 대비 COCO_AP는 -0.40 손해를
보지만, decision-preservation 지표(Heval_flip/Top1_flip/lost/LVIS_flip/
LVIS_lost)는 §5.3의 identity-aware margin_loss가 그 손실 일부를 상쇄해서
전부 유지되거나 개선된다.

(재현 시 주의: `torch.ones(N)`/`torch.tensor(1.0)` 둘 다 `device=...`를
명시해야 한다 — 0-dim 텐서는 CUDA 텐서와 섞여도 PyTorch가 암묵적으로
허용해주는 경우가 있어 이 버그가 예전에 숨어 있었다.)

### 5.3 학습 목적함수 — `optimize_promptcal_scale_neighbor`

*`src/quant/promptcal.py:372`*

네 개 항의 합으로 `s_mult`만 학습한다(alpha는 고정):

**(a) margin_loss(S)** — S(40개 프롬프트)의 top-(k+1) 인접 margin을 FP와
identity-aware하게 맞춤:

```python
def margin_loss(sim_q, sim_fp, k=5, boundary_w=3.0, identity_aware=False):
    kk = min(k + 1, sim_fp.shape[-1])
    fp_top, fp_idx = sim_fp.topk(kk, dim=-1)
    if identity_aware:
        q_top = sim_q.gather(-1, fp_idx)          # FP가 정한 class 순서 그대로 Q값을 읽음
        mask = torch.zeros_like(sim_q, dtype=torch.bool).scatter_(-1, fp_idx[:, :-1], True)
        q_out = sim_q.masked_fill(mask, float("-inf")).max(-1).values  # FP top-k 밖 최댓값
        q_top = torch.cat([q_top[:, :-1], q_out[:, None]], dim=1)      # 마지막 열만 교체
    else:
        q_top, _ = sim_q.topk(kk, dim=-1)
    fp_m = fp_top[:, :-1] - fp_top[:, 1:]
    q_m  = q_top[:, :-1]  - q_top[:, 1:]
    w = torch.ones(kk - 1, device=sim_fp.device)
    w[-1] = boundary_w
    return ((q_m - fp_m).pow(2) * w).mean()
```
*`src/quant/promptcal.py:36`. 함수 자체의 기본값은 `identity_aware=False`
(라이브러리 레벨 기본값은 의도적으로 안 바꿈, `PROMPTCAL_CLAIMS_2026-09-15.md`
claim5-d 참고) — 확정 설계로 실행되는 건 `pipeline/run_comparison.py`의
CLI 기본값(`--identity-aware-margin` 기본 True)이 이 값을 명시적으로
넘기기 때문이다. `margin_loss`를 직접 호출하는 코드는 이 인자를 스스로
챙겨야 한다.*

**identity_aware가 왜 필요한가**: 원래(정렬 비교) 버전은 `fp_top`/`q_top`을
각자 독립적으로 `topk`해서 정렬된 값끼리만 비교했다 — 어느 class가 그
순위를 차지했는지는 버려진다. top-1/top-2가 서로 값을 맞바꾸는(class
identity가 뒤집히는) flip이 일어나도 margin 패턴 자체는 동일하면 loss가
0이 된다 — 즉 이 loss가 막으려는 대상(flip)이 목적함수에 아예 안 보이는
경로가 있었다(`PROMPTCAL_CLAIMS_2026-09-15.md` claim5). `identity_aware=True`는
`fp_idx`로 `sim_q`를 gather해서 identity를 고정하고, **마지막 열(경계,
`boundary_w`가 걸리는 자리)만 "FP top-k 밖에 있는 class 중 Q에서 가장 높은
값"으로 바꿔서 swap(앞쪽 k개 열)과 intrusion(마지막 열, FP가 전혀
고려하지 않던 class가 치고 올라오는 경우)을 둘 다 탐지한다**(claim5-b —
순수 identity-aware gather만으로는 intrusion을 놓친다는 걸 나중에 발견해서
보강함). 6-seed 풀스케일 검증 결과 AP 손해는 없고(§8.1) 대부분의
decision-preservation 지표가 개선돼서 채택.

**(a') margin_loss(H_cal)** — S와 동일한 margin_loss를 H_cal(20개)에도
직접 적용, `cal_weight`로 가중:

```python
if cidx is not None and cal_weight > 0:
    ml_cal = margin_loss(sim_q[aidx][:, cidx], sim_fp[aidx][:, cidx], k=k,
                         boundary_w=boundary_w, identity_aware=identity_aware_margin)
    ml = ml + cal_weight * ml_cal
```
*`src/quant/promptcal.py:524`*

**(b) confident-anchor 선정** — 어떤 anchor를 위 loss들의 대상으로 삼을지
정하는 기준이 **train_cols(S∪H_cal, 60열) 안에서의 top-1 confidence**로
제한돼 있다:

```python
train_cols = pidx if cidx is None else torch.cat([pidx, cidx])   # S ∪ H_cal
prob = sim_fp[:, train_cols].sigmoid(); mp, _ = prob.max(-1); conf = mp > conf_thres
```
*`src/quant/promptcal.py:499, 511`*

80열(COCO-80) 전체 기준으로 confidence를 재면, FP가 H_eval class를 1등으로
확신한 anchor까지 학습 gradient에 섞여 들어간다 — group_flip(Heval_flip
지표)도 정확히 같은 "80열 전체 top-1 ∈ H_eval" 기준으로 anchor를 고르기
때문에, 학습과 평가가 같은 anchor 풀을 공유하는 결합이 생긴다
(`PROMPTCAL_CLAIMS_2026-09-15.md` claim1). `train_cols`로 제한해서 H_eval이
anchor 선정 자체에도 영향을 못 주게 막았다 — 이 수정은 opt-in이 아니라
항상 적용되는 버그 수정이다.

**(c) asymmetric neighbor hinge** — S 각 class의 text-embedding 최근접
이웃(S∪H_eval 제외) `neighbor_k`개가 FP보다 **강해지는** 방향만 억제
(collateral shift 억제):

```python
diff = sim_q[aidx][:, neighbor_cols] - sim_fp[aidx][:, neighbor_cols]
nl = F.relu(diff).pow(2).mean()
```
*`src/quant/promptcal.py:535` 부근*

지금 확정된 40/20/20(S/H_cal/H_eval) split에서는 S∪H_eval을 제외하면
후보 풀이 **H_cal(20개) 전체와 동치**다 — 40개 S 앵커가 각각 `neighbor_k=5`개씩
뽑아도 합집합이 이미 H_cal 20개를 전부 커버하기 때문에(`neighbor_k`를
5/8/10으로 바꿔도 결과가 완전히 동일함으로 실측 확인), "text-embedding
최근접 이웃 selection"이라는 서술은 이 split에서는 **사실상 "H_cal 전체에
대한 one-sided hinge"와 동치**다(`PROMPTCAL_CLAIMS_2026-09-15.md` claim6).
논문에 이 메커니즘을 쓸 때는 이 사실을 명시할 것 — 후보 풀이 훨씬 큰
세팅에서만 "진짜 selection"이 일어난다.

**(d) scale_reg_weight 정규화** — `s_mult`가 "값싸게 전역적으로 낮추는"
지름길에 비용을 매김:

```python
if scale_reg_weight > 0:
    sr = sum((s - 1.0).pow(2).mean() for s in smults) / len(smults)
    loss = loss + scale_reg_weight * sr
```

최종 loss:

```
loss = margin_loss(S) + cal_weight · margin_loss(H_cal)
                       + neighbor_weight · neighbor_hinge(neighbors)
                       + scale_reg_weight · mean((s_mult-1)²)
```

### 5.4 S / H_cal / H_eval — 프롬프트 3분할

```python
rng = np.random.default_rng(seed)
perm = rng.permutation(80)
S      = perm[:40]     # margin_loss + neighbor 기준점으로 직접 사용
H_cal  = perm[40:60]   # margin_loss 직접 대상(cal_weight) + neighbor 후보 풀(사실상 전체, §5.3(c))
H_eval = perm[60:80]   # 최적화에 전혀 안 씀 -- confident-anchor 선정(§5.3(b))에서도 제외, 순수 평가 전용
```
*`pipeline/run_comparison.py`의 `main()`*

핵심 원칙: 측정할 프롬프트(H_eval)는 학습에 절대 넣지 않는다 — margin_loss
대상도, neighbor 후보도, confident-anchor 선정 기준도 전부 H_eval을
명시적으로 제외한다(§5.3 (a)(b)(c)).

**아키텍처 레벨 confound (수정 안 함, disclosure만)**: YOLO-World의
`C2fAttn`/`ImagePoolingAttn`이 cross-attention으로 **현재 활성화된 전체
vocabulary**에 대해 비전 feature를 조건화하므로, H_eval이 위 손실 함수
어디에도 안 쓰여도 calibration 중 80-class vocabulary에 "존재"하는
것만으로 아키텍처적으로 영향을 받는다(`PROMPTCAL_CLAIMS_2026-09-15.md`
claim2). 5개 방법 전부에 동일하게 적용돼 방법 간 비교는 안 깨지지만,
COCO-80 쪽 H_eval 지표를 "완전히 안 본 vocabulary"로 너무 강하게 해석하면
안 된다 — LVIS-transplant 결과(LVIS는 COCO-80 calibration 중 vocabulary에
전혀 없음)가 이 confound에서 자유로운 진짜 unseen-vocabulary 증거다.

### 5.5 현재 하이퍼파라미터 (확정값)

| 파라미터 | 값 | 비고 |
|---|---|---|
| `calib` | 256 (COCO train2017) | 평가 데이터와 완전 분리 |
| `recon_iters_ada`(AdaRound) | 1000 | |
| `iters`(s_mult 학습) | 1500 | |
| `lr` | 1e-2 | |
| `k`(margin top-k) | 5 | |
| `neighbor_k` | 5 | S 각 class당 이웃 개수(§5.3(c) 참고 — 사실상 H_cal 전체) |
| `neighbor_weight` | 1.0 | |
| `scale_reg_weight` | 10.0 | |
| `cal_weight` | 1.0 | H_cal(20개)에도 S와 동일 margin_loss 적용 |
| `channelwise_smult` | **False = per-tensor** | `--smult-per-tensor`(기본 True, `--no-smult-per-tensor`로 이전 설계) |
| `identity_aware_margin` | **True** | `--identity-aware-margin`(기본 True, `--no-identity-aware-margin`으로 이전 설계) |
| `w_bits` / `a_bits` | 8 / 8 | |

*`pipeline/run_comparison.py`의 `argparse` 기본값과 동일 — 플래그 없이
실행하면 이 표 그대로 재현된다.*

---

## 6. 데이터 설정 ("공식 데이터")

| 구분 | 소스 | 장수 |
|---|---|---|
| calibration | COCO **train2017**(평가셋과 완전 분리) | 256 |
| COCO-80 평가 | COCO val2017 **전체** | 5000 |
| LVIS 평가 | 공식 `lvis_v1_minival.json`(ultralytics 공식 배포) | 4809 |

LVIS minival(4809장)은 "COCO val2017 ∩ LVIS val"과 정확히 일치.

*실행: `pipeline/run_comparison.py`(공식 진입점),
`scripts/58_full_baseline_official_data.py`(원본, 09-16에 같은 설계로
동기화됨), GT 파일: `/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json`*

**주의(claim10)**: `--eval-cap`(스모크 테스트용 probe 수 제한)은
COCO_AP/S_AP/H_eval_AP에는 적용되지 않는다 — `measure_ap`가 `--data` yaml의
고정 val split을 써서 항상 풀스케일로 잰다. LVIS_AP/APr/APc/APf와
flip/GT/UPIR/lost 등 나머지 전부는 `--eval-cap`을 따른다. 실행 시 결과 표
위에 이 안내가 자동 출력된다.

---

## 7. 평가지표 전체 정의

| 지표 | 정의 | 무엇을 보는가 |
|---|---|---|
| COCO_AP | COCO-80 vocabulary, mAP50-95(pycocotools) | 표준 배포 시 총량 성능 |
| LVIS_AP | 같은 모델을 LVIS-1203 vocabulary로 바꿔 평가한 mAP | 낯선/촘촘한 vocabulary로 일반화됐을 때의 총량 성능 |
| S_AP / H_eval_AP | per-class AP를 S/H_eval 그룹으로 평균 | 학습에 쓴 프롬프트 vs 한 번도 안 쓴 프롬프트의 AP 분리 |
| Heval_flip(masked) | H_eval 20개 컬럼을 가린 뒤, FP가 H_eval로 판정했던 anchor에서 나머지 60class 중 1등이 FP/quant 간 같은가 | "H_eval을 아예 모르는 다른 사용자" 반사실적 시나리오(진단용, 실제 배포 조건 아님) |
| Top1_flip(표준) | masking 없이 FP confident anchor의 raw top-1이 quant와 같은가 | AP와 동일한 정상 배포 조건 |
| GT_MRR / GT_R@1 | 실제 COCO GT-anchor에서 quant 자신의 정답 순위 역수 평균 / 1등 비율 | 진짜 정답 기준 순위 보존도 |
| lost / gained | GT-anchor에서 FP는 1등이었는데 quant가 잃은/얻은 개수 | 진짜 정답 기준 손익 |
| UPIR | FP가 GT를 1등으로 맞춘 경우 중, quant가 그 1등을 H_eval class로 바꿔버린 비율 | "모르는 프롬프트의 정답 자리 침입" — 논문 핵심 동기와 가장 직접 대응 |
| LVIS_flip / LVIS_lost | 위 Top1_flip/lost를 LVIS vocabulary·실제 LVIS GT로 재측정 | 촘촘한 vocabulary에서도 결정/정답 보존이 되는가 |

지표 설계 배경은 `PROMPTCAL_HOW_IT_WORKS.md` §6 참고.

---

## 8. 성능 결과 — 확정 설계, 풀스케일 6-seed(0~5) (2026-09-16)

> **09-17 경고: 아래 표의 LVIS_AP/APr/APc/APf 열은 stale하다.** LVIS 평가가
> NMS `multi_label` 불일치 + standard-AP(300-cap) 프로토콜 버그로
> 공개 수치의 절반 수준으로 낮게 측정되고 있었음이 확인됐다(claim11,
> `PROMPTCAL_CLAIMS_2026-09-15.md` 참고 — FP32 기준 LVIS_AP 0.126→0.259로
> 재측정됨, 공개 수치와 6% 이내 일치). 코드는 수정·커밋됐지만
> (`adf9489`), 이 §8 표는 **아직 고친 코드로 재측정 전**이다. COCO_AP/
> S_AP/H_eval_AP 열과 flip/GT/UPIR/lost 계열은 이 버그와 무관해서
> 그대로 유효하다.

`runs/67_final_confirmed_fullscale/seed{0..5}_final.log`. 6개 seed 전부
물리 GPU 0/4/5/6/7(동일 모델, RTX4000 Ada)에서 실행해서 GPU-비결정성
confound 없음(§9 참고). **범위는 최소~최대**(괄호 안이 평균).

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | GT_MRR | GT_R@1 | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **FP32** | **36.80** | **0.1260** | 0.0310 | - | - | - | - | - | - | - | - |
| naive | 33.54 | 0.1264 | 0.0415 | 9.85~14.55%(11.80%) | 0.95% | - | - | 0.18~0.44%(0.31%) | 432 | 6.48% | 1224 |
| AdaRound | 33.24 | 0.1242 | 0.0328 | 8.74~13.39%(10.85%) | 0.74% | - | - | 0.13~0.31%(0.25%) | 349 | 5.67% | 1212 |
| QDrop | 33.30 | 0.1235 | 0.0311 | 8.81~13.29%(10.84%) | 0.72% | - | - | 0.13~0.33%(0.27%) | 384 | 5.71% | 1192 |
| BRECQ | 33.30 | 0.1200 | 0.0319 | 8.55~12.67%(10.48%) | 0.71% | - | - | 0.11~0.33%(0.23%) | 327 | 5.53% | 1158 |
| **Combined** | **35.86~36.07(35.96)** | 0.1201~0.1228(0.1214) | 0.0356~0.0472(0.0387) | **7.29~10.40%(9.23%)** | **0.61~0.73%(0.657%)** | 0.9221~0.9227(0.9224) | 0.8759~0.8774(0.8768) | 0.13~0.29%(0.213%) | 337~381(355.0) | **5.00~5.49%(5.15%)** | **981~1086(1017)** |

**baseline(BRECQ, 가장 강한 baseline) 대비**: COCO_AP(+2.66), Heval_flip,
Top1_flip, LVIS_flip, LVIS_lost 전부 이기고, UPIR은 근소하게 비슷,
`lost`(raw count, 355 vs 327)·CorrRate만 여전히 진다. APr(rare class)은
AdaRound/QDrop/BRECQ를 전부 이기지만 naive(0.0415)보다는 낮다.

**GT_MRR/GT_R@1 대조가 보여주는 것**: 5개 방법의 GT_MRR(0.9220대)·
GT_R@1(0.877대)이 거의 완전히 겹친다 — 절대적 랭킹 정확도만 보면 naive조차
다른 방법들과 별 차이가 없다. 그런데 AP는 33.24~36.07로 크게 갈리고,
Top1_flip/Heval_flip도 뚜렷이 갈린다. **AP 개선이 "GT 랭킹 점수 자체를
절대적으로 더 잘 복원해서" 나온 게 아니라, "FP32가 매기던 순서/결정을
얼마나 유지하는가"(flip)에서 나온다는 뜻** — `margin_loss`가 값 자체가
아니라 순위 간격(margin)을 맞추도록 설계된 것과 정확히 같은 철학이다.

**이전(per-channel, identity-unaware) 설계 대비**: v1 §8.11에 그 표가
보존돼 있다. COCO_AP -0.40(범위가 서로 안 겹칠 만큼 확실한 손실 —
`--eval-cap 1000` 빠른 검증에서 봤던 노이즈 수준(-0.06)보다 훨씬 큼)을
지불하지만, decision-preservation 계열(Heval_flip/Top1_flip/lost/
LVIS_flip/LVIS_lost)은 전부 유지되거나 개선(LVIS_lost -66이 특히 뚜렷)됐다.

과거의 모든 하이퍼파라미터 스윕(scale_reg_weight, k/neighbor_k/boundary_w,
neighbor_weight, cal_weight 도입 등) 근거는 v1 §8.2~§8.10에 그대로 남아있다
— 지금 표의 하이퍼파라미터(§5.5)는 그 스윕들의 결론이다.

---

## 9. 알아둘 점 / 한계 / 열린 이슈 (현재 기준)

- **APr(rare class) 논의는 09-17 claim11로 재측정 전까지 보류**: 아래
  0.0387 vs naive 0.0415는 LVIS eval 버그(§8 상단 경고)가 낀 채로 측정된
  수치라, rare class에서 가장 크게 흔들렸을 가능성이 높다(FP32 기준
  APr이 버그 수정으로 5.7배 뛴 전례). 수정된 코드로 재측정하기 전에는
  "AdaRound/QDrop/BRECQ는 이기지만 naive는 못 이긴다"는 판정 자체를
  신뢰하지 말 것.
- **UPIR·lost는 BRECQ보다 못하다**: `lost`는 6-seed 전부 BRECQ(327)보다
  나쁨(337~381), UPIR도 근소하게 밀림. v1 §9의 그룹별 분해 분석(가설:
  "이 비용이 H_eval에 국소적으로 몰림" → 기각, S/H_eval에 고르게 나타나는
  일반적 트레이드오프)이 여전히 유효한 설명이다.
- **아키텍처 레벨 confound (claim2, §5.4 참고)**: C2fAttn/ImagePoolingAttn이
  H_eval의 vocabulary상 "존재"만으로 비전 feature에 영향을 준다 — COCO-80
  H_eval 지표는 완전한 unseen-vocabulary 증거로 과신하지 말 것, LVIS
  결과가 confound-free다.
- **neighbor selection은 이 split에서 사실상 "H_cal 전체"다 (claim6,
  §5.3(c) 참고)**: "최근접 이웃 selection"이라는 서술을 쓸 때 이 사실을
  명시할 것.
- **GPU 물리 기기가 다르면 동일 코드·동일 seed도 결과가 갈린다**: 과거
  GPU-swap 실험(v1 §9)에서 `lost` 등 일부 지표가 seed보다 물리 GPU를
  따라가는 패턴이 확인됐다. **"논문에 실릴 확정 표"를 만들 때는 6-seed
  전부 같은 GPU 모델에서 돌릴 것** — §8의 표는 이 원칙을 지켰다(전부
  RTX4000 Ada). `CUDA_VISIBLE_DEVICES=N`만으로는 물리 GPU N이 보장 안
  되니 `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 항상 같이 쓸 것.
- **GPU 공유 서버 에티켓**: 실행 전 항상
  `nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv`로
  다른 사용자 프로세스가 없는 GPU인지 확인(메모리 0만 보고 판단하지 말 것).
- **메모리**: 풀스케일 5000+4809장 평가를 여러 seed 동시 실행하면 시스템
  RAM(swap 포함)이 빠듯할 수 있다 — 실행 전후로 `free -h`를 확인할 것.
  단일 job의 CPU RSS가 비정상적으로 커지는 문제(fp16 CPU 버퍼 누적)는
  `free_cpu_mem()`(malloc_trim)으로 이미 완화돼 있다(`src/quant/adaround.py`).

---

## 10. 관련 문서/코드 인덱스

| 내용 | 위치 |
|---|---|
| 이 문서 이전 버전(per-channel 시절 전체 서사 + 하이퍼파라미터 스윕 10개 절 원본) | `PROMPTCAL_CURRENT_MODEL.md`(v1) |
| claim 1~10 검증 경위(무엇을 확인했고 무엇을 왜 고쳤는지 전체 감사 기록) | `PROMPTCAL_CLAIMS_2026-09-15.md` |
| 측정 도구·baseline·평가지표 설계 배경(서술 중심) | `PROMPTCAL_HOW_IT_WORKS.md` |
| `pipeline/` 디렉토리(공식 재현 진입점) | `pipeline/run_comparison.py`, `pipeline/README.md` |
| Combined 학습 루프 | `src/quant/promptcal.py`(`optimize_promptcal_scale_neighbor`, `margin_loss`) |
| s_mult/AdaRound 구현 | `src/quant/adaround.py` |
| baseline 구현 | `src/quant/adaround.py`(AdaRound/QDrop), `src/quant/brecq.py` |
| 측정 하네스 | `src/harness.py` |
| §8 원본 로그 | `runs/67_final_confirmed_fullscale/seed{0..5}_final.log` |
| `_official_data` 계열 스크립트(전부 09-16 기준 확정 설계가 기본값) | `scripts/58_full_baseline_official_data.py`, `scripts/59_rw_sweep_official_data.py`, `scripts/60_hparam_sweep_official_data.py`, `scripts/61_combo_grid_official_data.py` |

---

*이 문서와 실제 코드/결과가 어긋나면 코드/로그를 우선하고 이 문서를 고칠
것 — v1 §9에 기록된 "문서 서술이 실제보다 늦게 갱신되는" 패턴이 여러 번
있었다.*
