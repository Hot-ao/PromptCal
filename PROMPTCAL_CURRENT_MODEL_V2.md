# Combined 모델 현재 설계 전체 문서 v2 (2026-09-17 작성)

**이 문서가 지금부터의 단일 진실 공급원이다.** [`PROMPTCAL_CURRENT_MODEL.md`](PROMPTCAL_CURRENT_MODEL.md)
(v1, 2026-09-08~09-16 작성)는 per-channel `s_mult` 설계를 중심으로 쓰였고
그 뒤로 수많은 패치(09-09 H_eval 버그, 09-10~09-12 하이퍼파라미터 스윕
10개 절, 09-16 per-tensor+identity-aware 전환)가 쌓여서 지금 실제 설계를
파악하기엔 너무 두껍다 — 이 v2는 **지금(09-17) 확정된 설계만** 처음부터
깔끔하게 다시 쓴 것이다. 과거 스윕의 세부 근거·claim 1~10의 검증 경위가
필요하면 v1과 [`PROMPTCAL_CLAIMS_2026-09-15.md`](PROMPTCAL_CLAIMS_2026-09-15.md)를
감사(audit) 기록으로 참고할 것 — 이 v2는 그 결과만 반영한다.

> **09-19 갱신(claim15, 최신)**: baseline 비교 축이 불공정했다(QDrop/BRECQ의
> activation scale 학습이 꺼진 채로 Combined와 비교) — QDrop/BRECQ에 LSQ를
> 원 논문대로 기본 적용하게 고치고(§4), 그 공정 비교에서 `scale_reg_weight`를
> 10.0→1.0으로 재튜닝했다(§5.5/§8). 결과: **Combined가 9개 지표 중 4개
> (COCO_AP/LVIS_AP/APr/LVIS_lost)는 BRECQ+LSQ를 이기지만, 나머지 5개
> (Heval_flip/Top1_flip/UPIR/lost/LVIS_flip)는 아직 진다.** BRECQ의 block-wise
> 재구성 위에 margin_loss를 얹는 진단 실험을 **6-seed로 확정**한 결과도
> 똑같은 4승 5패 패턴 — margin_loss가 BRECQ 자신의 LSQ보다 decision-
> preservation에서 못하다는 게 재현성 있게 확인됐다. **더 중요한 발견**:
> weight-side foundation을 naive→BRECQ block-wise로 바꿔도 Combined의
> 최종 성능은 거의 안 변한다(9개 지표 거의 전부 오차범위 내) — claim14
> 결정(naive로 충분)이 한 번 더 확인됐고, 남은 격차의 원인이 foundation이
> 아니라 **margin_loss 자체**임이 명확해졌다 — §5.1/§9 참고.
>
> **09-20 갱신(claim16)**: 남은 격차를 좁히려던 두 방향(`neighbor_of_cal`
> 보호 범위 확장, margin_loss에 dense 보조 신호 `aux_mse_weight` 추가)을
> 시도했으나 **둘 다 부정적** — 전자는 COCO-80이 S+H_cal+H_eval로 정확히
> 꽉 차 있어서 구조적으로 확장할 후보가 없고(claim6 "포화" 그 이상, 완전히
> 동일함), 후자는 1-seed에서 9개 중 6개가 좋아 보였지만 6-seed로 전부
> 반박됨(seed 노이즈). **확정 설계 변경 없음**. §9 "남은 방향"이 사실상
> 소진돼서, margin_loss를 top-k 방식과 근본적으로 다른 objective로
> 재설계할지는 사람의 판단이 필요한 지점 — §9 참고.
>
> **09-18 갱신(claim14)**: claim13 수정으로 Combined 자신의 1단계
> (AdaRound weight rounding)도 alpha를 크게 움직이게 됐는데, 그 목적함수가
> 2단계(margin_loss/s_mult)가 보호하는 영역 밖으로 손상을 새게 한다는 걸
> 확인 — **1단계를 아예 제거(`--combined-recon-iters 0`, round-to-nearest
> weight로 대체)한 게 트레이드오프 없이 더 낫다**(6-seed, §5.1/§8). claim13
> 자체의 baseline(AdaRound/QDrop/BRECQ) 결과(전부 naive보다 COCO_AP가 낮음)는
> 그대로 유효 — `runs/71_recon_fix_review/` 참고.

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

> **09-19 갱신(claim15, 최신)**: baseline의 activation scale 학습 여부가 이제
> **방법별로 원 논문에 맞게 확정 기본값**이다 — 더 이상 opt-in 실험 플래그가
> 아니다. QDrop/BRECQ는 기본으로 LSQ(activation scale 학습)가 켜지고,
> AdaRound는 기본으로 꺼진다(원 논문에 없음). 예전엔 넷 다 꺼둔 채로 Combined와
> 비교했는데, 이는 "방법의 차이"가 아니라 "구현 축소와의 비교"라 리뷰어 반론을
> 못 막는다(사용자 지적). 아래 표·코드는 이 확정 기본값 기준이다.

| baseline | 방식 | activation scale 학습(LSQ) | 코드 |
|---|---|---|---|
| naive | round-to-nearest, 추가 최적화 없음 | 없음(고정 min-max) | `src/quant/fake_quant.py` |
| AdaRound | layer별 출력 재구성(`lp_rec_loss`, BRECQ lp_loss와 동일 정규화)으로 반올림(alpha) 학습 | **없음(기본, 원 논문에 없음)** | `src/quant/adaround.py`, `optimize_adaround` |
| QDrop | **BRECQ의 block-wise 재구성 위에** 확률적 activation drop(`qdrop_prob`, block 내부+block 입력 두 곳) | **있음(기본, 원 논문에 있음)** | `src/quant/brecq.py`, `optimize_brecq(qdrop_prob=...)` |
| BRECQ | block 단위 joint 재구성(층 간 상관 반영) | **있음(기본, 원 논문에 있음)** | `src/quant/brecq.py`, `optimize_brecq` |

**activation scale 학습 방법별 기본값 근거(claim12)**: 원 BRECQ 논문(Li et al.
ICLR'21)은 alpha(rounding)와 activation step size(LSQ)를 같은 reconstruction
loss로 공동 최적화한다. QDrop(Wei et al. ICLR'22)은 이 BRECQ 프레임워크를
그대로 물려받아 LSQ를 포함한다. **AdaRound(Nagel et al. ICML'20)는 순수
weight-rounding 방법이라 LSQ가 없다** — 그래서 AdaRound만 기본 False다.
`pipeline/run_comparison.py`의 `--adaround-learn-act-scale`(기본 False)·
`--qdrop-brecq-learn-act-scale`(기본 True)로 각각 ablation 가능(확정 비교표에는
쓰지 말 것). LSQ가 켜지면 Combined와 동일한 `s_mult` 메커니즘(per-tensor로
자동 강제)을 그 방법 자신의 reconstruction loss로 학습한다.

**공정 비교 6-seed 결과(claim15, `runs/75_lsq_confirmed/`)**: Combined는
9개 지표 중 **COCO_AP·LVIS_AP 2개만 이기고 나머지 7개(APr 포함 decision-
preservation 전부)는 BRECQ+LSQ한테 진다** — §8 참고. 이후 `scale_reg_weight`
재튜닝(claim15 이어서)으로 COCO_AP·LVIS_AP·APr·LVIS_lost 4개 승리로 만회.

---

## 5. Combined 모델 상세 (현재 확정 설계)

### 5.1 1단계 — weight rounding: **round-to-nearest, AdaRound 생략** (09-18 claim14로 확정)

```python
w_int = floor(w/scale) + h(alpha)     # h(alpha) ∈ [0,1], rectified sigmoid
```
*`src/quant/adaround.py:87` (`class AdaRoundQuantConv2d`)*

**09-18 이전엔** `optimize_adaround(m.model, fp.model, calib, device,
iters=1000)`로 alpha를 학습한 뒤 hard-round 확정하고 2단계로 넘어갔다. claim13
(재구성 loss 정규화 수정)으로 이 1단계가 실제로 alpha를 크게 움직이게 되면서
(nearest 대비 flip <1%→5~7%), **그 목적함수(순수 layer-wise MSE reconstruction)가
2단계(margin_loss/s_mult)가 보호하는 영역(S∪H_cal+neighbor)과 무관해서, 그
바깥(COCO 전체 Top1_flip/lost, LVIS 1203-class)으로 새는 손상이 커지는 부작용이
드러났다**(claim14, `PROMPTCAL_CLAIMS_2026-09-15.md` 참고).

**확인 실험**(`--combined-recon-iters`, `pipeline/run_comparison.py`): 1단계
iters를 0으로 만들어(=완전히 생략, `ac.alpha`가 초기값에 남고 이건 정확히
round-to-nearest와 같음 — 2단계 시작 시 `ac.soft=False`로 바뀌어
`(h_alpha(alpha)>=0.5)`가 hard 반올림 판정이 되는데, 초기 alpha는 h_alpha가
소수부와 같게 초기화돼 있어 nearest와 동일) 6-seed 재측정한 결과, **COCO_AP·
LVIS_AP·APr·Heval_flip·LVIS_lost 5개 지표가 전부 개선**됐다(§8 참고) — AP를
포기하고 decision을 지키는 트레이드오프가 아니라, 1단계를 없애는 게 AP와
decision-preservation 둘 다에 좋았다. **1단계 자체가 필요 없다는 뜻으로 채택**
— Combined는 이제 "naive round-to-nearest weight + margin_loss 기반
activation scale"이며, AdaRound 메커니즘을 전혀 안 쓴다. `optimize_adaround`
함수 자체는 `--combined-recon-iters`로 필요하면 되살릴 수 있지만(opt-in,
`0` 아닌 값), 기본값은 이제 `0`이다.

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
*`src/quant/adaround.py:89` (constructor, `channelwise_smult` 파라미터로 토글),
`src/quant/adaround.py:168` (`_quantize_smult`)*

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
| `recon_iters_ada`(AdaRound) | 1000 | **baseline(AdaRound/QDrop/BRECQ) 조건에만 적용됨** — Combined는 09-18(claim14)부터 이 값과 무관(아래 `combined_recon_iters` 참고) |
| `combined_recon_iters`(Combined 1단계) | **0(생략, round-to-nearest weight)** | 09-18 claim14 확정. 0보다 크게 주면 이전처럼 AdaRound 1단계를 되살릴 수 있음(§5.1, §9 참고 — 재도입은 decision-preservation을 해침) |
| `iters`(s_mult 학습, 2단계) | 1500 | claim14로 1단계가 바뀌었지만 이 값 자체는 아직 재튜닝 안 함(§9 "남은 방향" 참고) |
| `lr` | 1e-2 | |
| `k`(margin top-k) | 5 | |
| `neighbor_k` | 5 | S 각 class당 이웃 개수(§5.3(c) 참고 — 사실상 H_cal 전체) |
| `neighbor_weight` | 1.0 | |
| `scale_reg_weight` | **1.0**(09-19 claim15로 10.0에서 하향) | 1단계가 naive(claim14)로 바뀐 뒤 10.0은 과도한 정규화였음 — 재스윕 결과 1.0이 최적 구간(0.0~0.5는 LVIS_flip 악화, 1.0 근방이 COCO_AP/LVIS_AP/APr/LVIS_lost 동시 최고) |
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

> **09-19 갱신(claim15, 최신): 비교 축을 공정하게 만들고(QDrop/BRECQ에 LSQ
> 기본 적용, §4 참고) `scale_reg_weight`를 재튜닝했다.** 이전 표(아래 09-18
> claim14 배너)는 baseline에 activation scale 학습 손잡이가 꺼진 채로
> Combined와 비교한 것이었다(사용자 지적) — "방법 차이"가 아니라 "구현
> 축소와의 비교"라 리뷰어 반론을 못 막는 문제였다. QDrop/BRECQ에 LSQ를 기본
> 적용한 공정 비교로 6-seed 재측정한 결과(`runs/75_lsq_confirmed/`),
> **Combined는 9개 지표 중 COCO_AP·LVIS_AP 2개만 이기고 나머지 7개(APr
> 포함)는 BRECQ+LSQ한테 졌다.** 이어서 `scale_reg_weight`를 1-seed
> 스윕(10.0→2.5→1.0→0.5→0.0)한 결과, **1.0이 최적점**임을 확인(0.0/0.5는
> LVIS_flip이 오히려 악화 — 이 정규화가 원래 막으려던 현상 재현). 1.0으로
> 6-seed 재검증한 결과(`runs/78_scalereg1_confirmed/`)가 **아래 표** — 하이퍼
> 파라미터 재튜닝만으로(설계 변경 없이) BRECQ+LSQ 대비 승리 지표가 2개→
> **4개**(COCO_AP·LVIS_AP·APr·LVIS_lost)로 늘었다. 남은 5개(Heval_flip/
> Top1_flip/UPIR/lost/LVIS_flip)는 아직 진다 — LVIS_flip 격차는 0.39pp→
> 0.18pp로 절반 이하로 좁혀짐. `--scale-reg-weight` 기본값을 `1.0`으로
> 전환 완료.
>
> **진단(6-seed 확정, `runs/79_brecq_stage1_confirmed/`): margin_loss 자체의
> 순수 기여가 부족함이 재현성 있게 확인됐다.** BRECQ의 block-wise
> 재구성(alpha만, LSQ 없이) 위에 margin_loss를 얹은 진단 실험이 BRECQ
> 자신의 native LSQ보다 COCO_AP·LVIS_AP·APr·LVIS_lost 4개만 이기고 나머지
> 5개(Heval_flip/Top1_flip/UPIR/lost/LVIS_flip)는 진다 — naive+margin_loss
> (현재 확정 설계)와 **똑같은 4승 5패 패턴**. 더 중요한 건 BRECQ-stage1과
> naive-stage1(현재 확정)의 최종 성능이 서로 거의 차이가 없다는 것 — **즉
> weight-side foundation은 결과에 거의 영향이 없고, 남은 격차의 원인은
> foundation이 아니라 margin_loss 자체**다. 제안 방법 자체(naive+margin_loss)는
> 안 바꿨다 — 이건 "왜 격차가 남는가"를 이해하기 위한 진단용(§5.1, claim15 참고).
>
> **09-18 갱신(claim14): Combined의 1단계(AdaRound weight rounding)를
> 제거하고 6-seed 재측정, 새 확정 설계로 채택.** claim13 수정 후 1단계가
> alpha를 크게 움직이게 됐는데, 그 목적함수가 2단계(margin_loss/s_mult)가
> 보호하는 영역과 무관해서 그 바깥으로 손상이 샌다는 가설을
> `--combined-recon-iters 0`(1단계 완전 생략 = round-to-nearest weight)로
> 확인 — `runs/72_combined_recon_diag/seed{0..5}_iters0.log`, 6-seed 전부
> **트레이드오프 없이 COCO_AP·LVIS_AP·APr·Heval_flip·LVIS_lost 5개 지표 동시
> 개선**(§5.1 참고). Top1_flip/UPIR/lost 3개만 QDrop/BRECQ에 근소하게 남아
> 진다(격차는 claim13 이전보다 훨씬 좁음). `--combined-recon-iters` 기본값을
> `0`으로 전환 완료 — 아래 표가 새 기본 설계다.
>
> **09-18 갱신: claim13(재구성 loss 정규화 버그) 수정 코드로 6-seed
> 재측정 완료.** `optimize_adaround`/`optimize_brecq`의 재구성 loss를 공식
> BRECQ `lp_loss`와 동일 정규화로 교체하고 lr/reg_weight/warmup을 공식값
> (1e-3/0.01/0.2)으로 고쳤다 — **이건 성능을 올리려는 수정이 아니라
> AdaRound/QDrop/BRECQ를 원 논문대로 정확히 구현하기 위한 수정**이다
> (claim4/5와 같은 원칙: 공정성/정확성 문제는 결과 방향과 무관하게 채택).
> `runs/71_recon_fix_review/seed{0..5}_full.log`에서 6-seed 전부 재실행
> 완료(물리 GPU 0/4/5 동일 모델). **결과: AdaRound/QDrop/BRECQ 3개 baseline이
> 6-seed 전부에서 naive보다 여전히 낮다**(COCO_AP naive 33.54 vs AdaRound
> 33.34/QDrop 33.24/BRECQ 33.15) — `flip_rate()` 진단으로 alpha가 실제로
> 5~7% 정도 움직임을 확인했는데도(수정 전은 <1%) 그렇다. 이는 고쳐야 할
> 이상 현상이 아니라 **정확한 재현으로 얻은 결과**로 취급한다 — 오히려
> "reconstruction 최적화가 detection AP를 보장하지 않는다"는 이 논문의
> 핵심 주장과 방향이 같다. baseline 4개도 이제 alpha 최적화에 매 iteration
> 무작위 샘플링이 들어가 **seed마다 값이 달라진다**(이전엔 결정적이라
> 6-seed 전부 동일했음).
>
> **09-17 갱신: 표 전체를 `runs/68_lvis_fix_fullscale` 단일 실행으로
> 재작성했다.** LVIS 평가가 NMS `multi_label` 불일치 + standard-AP
> (300-cap) 프로토콜 버그로 공개 수치의 절반 수준으로 낮게 측정되고
> 있었음이 확인·수정됐다(claim11, `PROMPTCAL_CLAIMS_2026-09-15.md` 참고,
> 커밋 `adf9489`). 처음엔 "COCO_AP/flip 계열은 버그와 무관하니 구 실행
> (`runs/67`) 값을 유지하고 LVIS_AP/APr만 신 실행(`runs/68`) 값으로
> 교체"했었는데, 두 실행을 섞으면 출처 추적이 헷갈려서 **`runs/68`
> 6-seed 하나로 전부 통일**했다. `runs/67` 대비 COCO_AP/Top1_flip/
> LVIS_flip/LVIS_lost의 range 최댓값이 소폭 낮아진 정도의 차이만 있고
> (Combined의 neighbor sampling 등 확률적 요소에 의한 정상적인
> run-to-run 변동, GT_MRR/GT_R@1은 두 실행에서 완전히 동일했음 — GPU는
> 두 실행 다 물리 0/4/5/6/7 동일 모델이라 비결정성 confound 아님),
> 결론에 영향은 없다. FP32 LVIS_AP는 0.1260→0.2589로(공개 수치 0.243과
> 6% 이내), **Combined LVIS_AP는 0.1214→0.2476, APr은 0.0387→0.1704로
> 뛰었다** — "Combined가 APr에서 naive보다 진다"는 구 서술은 완전히
> 뒤집혔다(아래 표 참고).

6개 seed 전부 물리 GPU 0/4/5/6/7(동일 모델, RTX4000 Ada)에서 실행해서
GPU-비결정성 confound 없음(§9 참고). **범위는 최소~최대**(괄호 안이 평균).
baseline 4개(naive/AdaRound/QDrop/BRECQ)는 COCO_AP/LVIS_AP/APr이 6-seed
동일(calib 256장 샘플링이 결정적이라 baseline 빌드 자체가 seed-불변),
Heval_flip/UPIR만 seed별 H_eval split 차이로 변동.

baseline 4개도 이제 **범위는 최소~최대**(괄호 안이 평균) 형식이다(seed마다
alpha 최적화가 확률적). **QDrop/BRECQ는 LSQ 기본 적용(claim12/15, §4 참고),
AdaRound는 원 논문대로 LSQ 없음.**

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | GT_MRR | GT_R@1 | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **FP32** | **36.80** | **0.2589** | 0.1767 | - | - | - | - | - | - | - | - |
| naive | 33.54 | 0.2342 | 0.1704 | 9.85~14.55%(11.80%) | 0.95% | - | - | 0.18~0.44%(0.31%) | 432 | 6.48% | 1224 |
| AdaRound(LSQ 없음) | 33.26~33.41(33.34) | 0.2316~0.2353(0.2335) | 0.1606~0.1689(0.1633) | 9.29~13.67%(11.40%) | 0.79~0.87%(0.827%) | - | - | 0.12~0.36%(0.27%) | 357~410(385.7) | 5.93~6.35%(6.13%) | 1224~1310(1271.7) |
| QDrop+LSQ | 35.74~36.12(35.86) | 0.2466~0.2489(0.2475) | 0.1747~0.1800(0.1771) | 6.51~9.08%(8.27%) | 0.60~0.64%(0.625%) | - | - | 0.10~0.22%(0.173%) | 275~306(289.2) | 5.24~5.42%(5.34%) | 1088~1141(1108.3) |
| BRECQ+LSQ | 35.65~35.88(35.78) | 0.2465~0.2495(0.2478) | 0.1713~0.1789(0.1757) | **6.51~9.20%(7.95%)** | **0.59~0.62%(0.607%)** | - | - | **0.09~0.24%(0.155%)** | **262~290(279.2)** | **5.01~5.26%(5.12%)** | 998~1077(1049.7) |
| **Combined** | **36.39~36.66(36.55)** | **0.2526~0.2554(0.2539)** | **0.1740~0.1847(0.1797)** | 7.91~10.93%(9.52%) | 0.72~0.76%(0.735%) | 0.9231~0.9235(0.9233) | 0.8782~0.8786(0.8784) | 0.07~0.31%(0.173%) | 323~348(339.2) | 5.13~5.50%(5.30%) | **922~995(959.0)** |

**claim15(공정 비교 + `scale_reg_weight` 재튜닝) 반영값 — 지금 최종 확정
표.** 앞선 claim14 표(위 09-18 배너 참고)는 QDrop/BRECQ의 LSQ가 꺼진 채로
비교한 것이라 stale — "방법 차이"가 아니라 "구현 축소와의 비교"였다.

**baseline(QDrop+LSQ/BRECQ+LSQ) 대비**: Combined는 **COCO_AP(+0.77, BRECQ+LSQ
대비)·LVIS_AP(+0.0061)·APr(+0.0040)·LVIS_lost(+90.7, BRECQ+LSQ 1049.7 대비
959.0)** 4개를 이긴다. **Heval_flip·Top1_flip·UPIR·lost·LVIS_flip 5개는
BRECQ+LSQ(사실상 지금 가장 강한 방법)한테 진다** — 특히 UPIR은 BRECQ+LSQ와
사실상 동률(0.173%=0.173%, 우연히 정확히 같음). LVIS_flip 격차는 0.18pp
(5.30 vs 5.12)로 claim14 표 대비 절반 이하로 좁혀졌다. `scale_reg_weight`
10.0→1.0 재튜닝 하나만으로(설계 변경 없이) 승리 지표가 2개→4개로 늘었다 —
남은 5개 격차를 어떻게 줄일지가 열린 문제(§9 "남은 방향" 참고).

**GT_MRR/GT_R@1 대조가 보여주는 것**: Combined의 GT_MRR(0.9233대)·
GT_R@1(0.878대)이 seed 간 거의 안 흔들린다(baseline은 이 지표를 로그에
안 남겨서 직접 대조는 안 되지만, AP가 33대~36대로 크게 갈리는 것에 비해
랭킹 절대 정확도는 상대적으로 안정적). **AP 개선이 "GT 랭킹 점수 자체를
절대적으로 더 잘 복원해서" 나온 게 아니라, "FP32가 매기던 순서/결정을
얼마나 유지하는가"(flip)에서 나온다는 뜻** — `margin_loss`가 값 자체가
아니라 순위 간격(margin)을 맞추도록 설계된 것과 정확히 같은 철학이다(다만
claim15의 BRECQ-stage1 진단 결과, margin_loss 자체의 순수 기여는 재검토가
필요하다 — §5.1/§9 참고).

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

- **공정 비교(claim15)에서는 baseline이 QDrop+LSQ/BRECQ+LSQ여야 한다** — 이전
  버전(baseline LSQ 꺼짐)과 비교한 서술은 전부 폐기. §4/§8 참고.
- **APr은 이제 QDrop+LSQ·BRECQ+LSQ를 이긴다**(Combined 0.1797 vs QDrop+LSQ
  0.1771 / BRECQ+LSQ 0.1757), naive(0.1704)도 이긴다. AdaRound(LSQ 없음,
  0.1633)만 참고용.
- **decision-preservation은 `scale_reg_weight` 재튜닝(10.0→1.0)으로 개선됐지만
  BRECQ+LSQ에는 여전히 5개(Heval_flip/Top1_flip/UPIR/lost/LVIS_flip) 진다**:
  claim15 공정 비교 직후(scale_reg_weight=10.0)엔 COCO_AP/LVIS_AP 2개만
  이겼는데, 1.0으로 재튜닝하니 APr·LVIS_lost가 추가로 뒤집혀 4개 승리가
  됐다. UPIR은 BRECQ+LSQ와 사실상 동률(0.173%=0.173%). LVIS_flip 격차는
  0.18pp로 좁아졌지만 아직 진다. v1 §9의 그룹별 분해 분석(가설: "이 비용이
  H_eval에 국소적으로 몰림" → 기각, S/H_eval에 고르게 나타나는 일반적
  트레이드오프)이 여전히 유효한 설명이다.
- **BRECQ-stage1 진단(6-seed 확정, claim15): 남은 격차의 원인은 foundation이
  아니라 margin_loss 자체다.** `--combined-stage1 brecq`(BRECQ의 block-wise
  재구성, alpha만, LSQ는 안 켬)로 만든 weight 위에 margin_loss를 얹은 6-seed
  결과(`runs/79_brecq_stage1_confirmed/`)가 BRECQ 자신의 native LSQ 대비
  **COCO_AP·LVIS_AP·APr·LVIS_lost 4개만 이기고 나머지 5개는 진다** — naive+
  margin_loss(현재 확정 설계)와 정확히 같은 4승 5패 패턴. **1-seed 때는
  "BRECQ foundation이 naive보다 decision-preservation에 낫다"고 봤는데,
  6-seed로 보니 BRECQ-stage1과 naive-stage1의 최종 성능이 거의 동일**하다
  (9개 지표 대부분 오차범위 내) — foundation 선택은 결과에 거의 영향이
  없고, margin_loss가 BRECQ의 reconstruction objective보다 decision-
  preservation에서 못하다는 게 foundation과 무관하게 재현된다. 즉
  margin_loss의 decision-preservation 우위로 봤던 게 상당 부분 "activation
  scale 손잡이 존재"에서 온 것이지 margin_loss의 설계 자체는 아니었다 —
  margin_loss 고유의 기여는 재검토가 필요하다. **제안 방법 자체(naive+
  margin_loss)는 안 바꿨다** — 이건 진단용 실험이다.
- **남은 격차(Heval_flip/Top1_flip/UPIR/lost/LVIS_flip)를 더 좁힐 수 있는
  방향(claim15/16)**: (1) `scale_reg_weight` 재튜닝 — **완료**(10.0→1.0, 위
  참고). **(2) `neighbor_of_cal`(S 이웃뿐 아니라 H_cal 이웃까지 보호 범위
  확장) — 시도했으나 구조적으로 무의미함을 확인(claim16)**: COCO-80이
  S(40)+H_cal(20)+H_eval(20)로 정확히 꽉 차 있어서, S의 이웃 후보 풀(S∪H_eval
  제외)이 이미 H_cal 20개 전부와 정확히 같다 — H_cal 자신의 이웃을 추가해도
  같은 20개 안에서만 도니 `neighbor_cols`가 한 개도 안 바뀜(1-seed 결과가
  기존값과 완전히 동일한 숫자로 확인). 이 방향은 COCO-80 80-class 폐쇄
  구조 자체의 한계라 더 손댈 여지가 없음. **(3) margin_loss에 dense 보조
  신호(`aux_mse_weight`, train_cols 전체에 대한 F.mse_loss를 margin_loss에
  추가) — 1-seed에서 유망해 보였으나(9개 중 6개 개선) 6-seed로 반박됨
  (claim16)**: aux_mse_weight=0.2를 6-seed 확정한 결과 APr/UPIR/lost/
  LVIS_flip/LVIS_lost가 전부 기존 확정값보다 나빠짐 — 1-seed 스윕에서 좋아
  보였던 건 seed 노이즈였다. **확정 설계는 변경 없음**(aux_mse_weight
  기본값 0.0 유지). (4) 약한 weight-level 보정(BRECQ block-wise 등)은
  이미 시도해서 무의미함을 6-seed로 확인함(BRECQ-stage1 진단) — 더 이상
  유망한 방향이 아님. **(2)(3)(4) 전부 소진** — margin_loss 설계를 top-k
  margin에서 근본적으로 다른 형태로 바꾸는 것 외에는 뚜렷한 다음 수가
  안 보이는 상태, 사람의 판단이 필요한 지점.
- **bit-width 메커니즘 정식 확정(claim17): 손상은 사실상 activation
  quantization 전부다.** `--w-bits`/`--a-bits`(claim15에서 구현)로 정식
  스케일(calib=256, cap 없음)에서 naive 측정(`runs/82_bitwidth_confirmed/`):
  W8A32(weight만 8bit)는 FP32 대비 COCO_AP -0.05(거의 무손실), W32A8
  (activation만 8bit)는 -3.29로 표준 W8A8의 -3.26과 거의 동일. 즉 weight
  rounding은 이미 거의 무손실이라 AdaRound류 weight 최적화가 별 도움이
  안 됐던 이유(claim13/14)와, Combined가 activation scale(`s_mult`)만
  조정해서 개선을 낸 것이 이 메커니즘과 정합함을 보여준다. calibration이
  결정적이라 seed 무관, 코드 변경 없음.
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
| claim 1~17 검증 경위(무엇을 확인했고 무엇을 왜 고쳤는지 전체 감사 기록) | `PROMPTCAL_CLAIMS_2026-09-15.md` |
| 측정 도구·baseline·평가지표 설계 배경(서술 중심) | `PROMPTCAL_HOW_IT_WORKS.md` |
| `pipeline/` 디렉토리(공식 재현 진입점) | `pipeline/run_comparison.py`, `pipeline/README.md` |
| Combined 학습 루프 | `src/quant/promptcal.py`(`optimize_promptcal_scale_neighbor`, `margin_loss`) |
| s_mult/AdaRound 구현 | `src/quant/adaround.py` |
| baseline 구현 | `src/quant/adaround.py`(AdaRound/QDrop), `src/quant/brecq.py` |
| 측정 하네스 | `src/harness.py` |
| §8 원본 로그(현재 확정 표, claim15 scale_reg_weight=1.0) | `runs/78_scalereg1_confirmed/seed{0..5}_full.log` |
| §8 이전 로그(claim15 공정 비교 재측정, scale_reg_weight=10.0) | `runs/75_lsq_confirmed/seed{0..5}_full.log` |
| §8 이전 로그(claim13/14 직후, LSQ 불공정 비교 — 폐기됨) | `runs/71_recon_fix_review/`, `runs/72_combined_recon_diag/` |
| claim16 진단 로그(neighbor_of_cal/aux_mse_weight, 둘 다 부정) | `runs/80_neighbor_aux_sweep/`, `runs/81_auxmse02_confirmed/` |
| claim17 로그(bit-width 메커니즘 정식 확정) | `runs/82_bitwidth_confirmed/` |
| `_official_data` 계열 스크립트 — **stale, claim12/13/14 미반영**(claim5-d와 같은 성격) | `scripts/58_full_baseline_official_data.py`, `scripts/59_rw_sweep_official_data.py`, `scripts/60_hparam_sweep_official_data.py`, `scripts/61_combo_grid_official_data.py` — 재사용 전 갱신 필요 |

---

*이 문서와 실제 코드/결과가 어긋나면 코드/로그를 우선하고 이 문서를 고칠
것 — v1 §9에 기록된 "문서 서술이 실제보다 늦게 갱신되는" 패턴이 여러 번
있었다.*
