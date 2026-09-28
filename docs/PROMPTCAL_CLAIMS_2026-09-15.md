# PromptCal-PTQ — Claim 1~10 검증/변경 기록 (2026-09-15~)

이 문서는 `pipeline/` 공식 데이터 설정 포팅(09-15) 이후, 리뷰어 공격 포인트
관점에서 제기된 claim 1~10(+ claim5의 하위 발견 a~f)을 코드로 검증하고
실제로 바꾼 내용을 정리한다.

**09-20 갱신(최신)**: claim17(bit-width 메커니즘 정식 확정 — activation
양자화가 W8A8 손상의 사실상 전부, weight quantization은 거의 무손실)
추가. 코드 변경 없음, 정식 스케일 측정만 수행. claim13/14의
"AdaRound weight-rounding 최적화가 별 도움 안 됨" 및 Combined의
activation-scale-only 설계와 메커니즘적으로 정합.

**09-20 갱신**: claim16(neighbor_of_cal/aux_mse_weight로 남은 격차를
좁히려는 두 시도, 둘 다 부정적 결과 — 하나는 구조적 막다른 길, 하나는
1-seed 노이즈였음) 추가. 확정 설계 변경 없음. margin_loss의 남은 5개
지표 격차를 어떻게 좁힐지는 사람의 판단이 필요한 지점.

**09-19 갱신**: claim15(비교 축 불공정 발견·수정 — QDrop/BRECQ LSQ
기본 적용, `scale_reg_weight` 10.0→1.0 재튜닝, BRECQ-stage1 진단) 추가.
claim14(1단계 AdaRound 제거)가 맞았던 결정이었음을 재확인하면서도, 공정
비교에서는 Combined가 9개 지표 중 4개만 이긴다는 게 드러남 — 지금까지
나온 것 중 가장 중요한 발견. 각 절 참고.

**09-18 갱신**: claim14(Combined의 1단계 AdaRound 제거 — 확정 설계로
채택, `--combined-recon-iters` 기본값 0) 추가. claim12(baseline
activation-scale 학습 비대칭)와 claim13(AdaRound/BRECQ 재구성 loss 정규화
버그, 6-seed 재측정 완료) 위에서 나온 발견. 각 절 참고.

**09-17 갱신**: 이 문서가 다룬 결정(claim4/5 채택 등)은 이제
[`PROMPTCAL_CURRENT_MODEL_V2.md`](PROMPTCAL_CURRENT_MODEL_V2.md)에 확정
설계로 반영 완료됐다 — 이 CLAIMS 문서는 계속 "무엇을 왜 검증했고 왜 그렇게
결정했는지"의 감사(audit) 기록으로 남긴다. 아래 본문에서 v1
(`PROMPTCAL_CURRENT_MODEL.md`)의 특정 절(§8.5, §8.10 등)을 인용하는 부분은
그 시점 문서 상태에 대한 정확한 역사적 인용이라 그대로 둔다.

**09-16 갱신**: claim4(per-tensor)와 claim5(identity-aware margin)를
새 확정 설계로 **채택하기로 결정**했다 — `PROMPTCAL_CURRENT_MODEL.md`
§8.1을 대체할 풀스케일 6-seed 검증이 진행 중이며, 완료되면 §8.1이
갱신된다(부록 D 참고). 그 전까지는 여전히 §8.1이 유효한 확정값이고,
여기 변경들은 opt-in 플래그(`--smult-per-tensor`,
`--identity-aware-margin`)로 기존 동작과 병행 가능하다. 코드는 이미
전부 커밋 완료(`87157e5`, `645f910`, README `c5f2cf1`/`1893592`) —
풀스케일 검증은 기존 커밋된 플래그를 다른 CLI 인자(`--eval-cap` 없이)로
실행한 것뿐이라 새 코드 변경은 없다.

---

## Claim 1 — confident anchor 선정이 80열(COCO-80) 전체 기준이라 H_eval이 샘

**주장**: `optimize_promptcal_scale_neighbor`의 confident-anchor 선정
(`prob = sim_fp.sigmoid(); mp = prob.max(-1)`)이 학습 컬럼(S∪H_cal)이 아니라
80열 전체를 기준으로 top-1을 계산해서, FP가 H_eval class를 1등으로 확신한
anchor까지 `aidx`에 들어가고 그 anchor의 S/H_cal 컬럼 값이 `margin_loss`
gradient에 쓰인다. `group_flip`(masked Heval_flip 지표)도 정확히 같은 기준으로
anchor를 고르므로, 학습과 평가가 같은 anchor 풀을 공유하는 결합이 생긴다.

**검증**: 코드 확인 결과 정확함 — `optimize_promptcal_scale_neighbor`(현재
사용 중인 함수)에서 실제 재현됨.

**수정**: [src/quant/promptcal.py](src/quant/promptcal.py) —
`train_cols = pidx if cidx is None else torch.cat([pidx, cidx])`를 정의하고,
confident-anchor 선정을 `sim_fp[:, train_cols].sigmoid().max(-1)`로 제한.
`optimize_promptcal_scale`(구버전)과 `optimize_promptcal_scale_neighbor_utility`
(미사용 변형)의 동일 패턴은 의도적으로 안 고침(실제 사용 경로만 수정).

**검증 실험**: 1-seed before/after 비교(`runs/62_anchorfix/seed0_{before,after}.log`,
calib=256, eval-cap=1000, seed=0). before는 GPU6, after는 GPU7에서 실행 —
**서로 다른 물리 GPU라 §9의 GPU-비결정성 confound가 섞여 있을 수 있음**에
주의. 결과는 혼재된 효과: Heval_flip/대부분 AP 지표는 개선, UPIR/LVIS_flip/
LVIS_lost는 악화.

**상태**: 코드 수정 완료, `src/`/`pipeline/` 동기화 완료, **6-seed 재검증
안 함, 커밋 안 함**. `PROMPTCAL_CURRENT_MODEL.md` §8.1을 갱신하려면 같은
물리 GPU에서 6-seed(0~5) before/after를 다시 돌려서 GPU 노이즈를 배제해야
한다.

---

## Claim 2 — C2fAttn/ImagePoolingAttn이 전체 vocabulary에 아키텍처적으로 조건화됨

**주장**: YOLO-World의 `C2fAttn`(`MaxSigmoidAttnBlock`)과 `ImagePoolingAttn`이
현재 활성화된 전체 text vocabulary에 대해 cross-attention을 계산하므로,
calibration 중 H_eval이 80-class vocabulary에 "존재"하는 것만으로도 vision
feature가 영향을 받는다 — loss 함수와 무관한 별도의 confound.

**검증**: `ultralytics.nn.modules.block`의 `MaxSigmoidAttnBlock.forward`
(`aw = einsum(...); aw = aw.max(dim=-1)[0]`)와 `ImagePoolingAttn.forward`
(`return x * self.scale + text`)를 직접 읽어 확인 — 정확함.

**결정**: **수정 안 함**. 이유:
1. 5개 방법(naive/AdaRound/QDrop/BRECQ/Combined) 전부에 동일하게 적용되므로
   방법 간 비교의 공정성은 안 깨짐.
2. LVIS-transplant 결과(LVIS는 COCO-80 calibration 중 vocabulary에 전혀
   없었음)는 이 confound에서 완전히 자유로움 — 이미 confound-free 증거가
   있음.
3. COCO-80 쪽 지표(H_eval_AP 등)에 한해서는 "H_eval이 완전히 안 보인
   프롬프트"라는 프레이밍이 살짝 약해진다는 점은 논문에 disclosure 필요.

**남은 선택지(미실행)**: Combined만 calibration 시 vocabulary를 60-class(H_eval
제외)로 줄여서 도는 ablation — 순수 검증용, 시도 안 함.

**09-16 Limitations 문구 초안**(영어, 논문 삽입용):

> *Limitations.* Because YOLO-World's vision-language fusion (C2fAttn's
> `MaxSigmoidAttnBlock` and `ImagePoolingAttn`) conditions visual features
> on the *entire currently active* text vocabulary via cross-attention,
> H_eval prompts — while never used in any loss term during calibration —
> are still architecturally present in the 80-class COCO vocabulary at
> calibration time. This applies identically across all five methods
> compared, so it does not bias the relative comparison, but it does mean
> COCO-80 metrics for H_eval should not be read as measuring performance
> under a *fully unseen* vocabulary in the strictest sense. Our
> LVIS-transplant results are free of this confound, since LVIS categories
> are never present in the vocabulary during COCO-80 calibration.

**상태**: 초안 작성 완료, 논문 Limitations 섹션에 삽입 대기(사용자 검토 필요).

---

## Claim 3 — AdaRound `reg_loss`의 sum/mean 스케일 불일치

**주장**: `reg_loss(self, beta, reduction="sum")`가 기본 sum-reduction인데
reconstruction loss는 `.mean()`, `reg_weight=1e-3`에 warmup도 없어서 정규화
gradient가 reconstruction gradient를 압도해 rounding이 naive round-to-nearest에
가깝게 붕괴할 수 있다는 우려. **"확정은 아니니 먼저 측정"** 하라는 명시적
지시로, 사용자가 제공한 `rounding_diff_rate` 진단 함수를 실제로 돌려서
확인함.

**측정**(calib=64, GPU7, `beta` annealing 20→2는 원 AdaRound 논문과 동일하게
이미 구현돼 있었음 확인):

| method | rounding_diff_rate(전체) | h→0/1 수렴률 | layer별 median | diff=0%인 layer |
|---|---|---|---|---|
| AdaRound | 8.73% | 98.7% | 10.89% | 0/71 |
| QDrop | 11.40% | 98.7% | 14.03% | 0/71 |
| BRECQ | 5.53% | 98.7% | 8.42% | 0/71 |

**결론**: 심각한 형태의 claim(붕괴해서 사실상 round-to-nearest가 됨)은
**기각**. 71개 layer 전부 nonzero diff(중앙값 8~14%), `alpha`가 0/1로
깔끔히 수렴(98.7%) — reg가 최적화를 무너뜨리고 있지 않음. QDrop이 가장
높고(노이즈 보정 필요) BRECQ가 가장 낮은(block-wise 상관 재구성이라 개별
자유도 제약) 순서도 합리적.

**남은 여지**: "완전 붕괴"는 기각됐지만 "약한 억제"까지 배제된 건 아님 —
`reduction="mean"`이나 `reg_weight=0` 대조 실험은 안 함.

**상태**: 코드 수정 없음, 측정만 완료. AdaRound/QDrop/BRECQ 공통 메커니즘이라
셋 다 해당.

---

## Claim 4 — per-channel(Combined) vs per-tensor(4개 baseline) activation quant 비대칭

**주장**: 4개 baseline은 activation을 per-tensor로 양자화하는데 Combined만
`s_mult`로 per-channel 자유도를 추가로 쓰는 게 unisolated confound. 배포
시에도 표준 INT8 커널이 per-channel activation dequant를 지원 안 해서
현실적으로 재현 불가능한 이점일 수 있다는 지적.

**검증**: `ActObserver.scale`/`zero_point`가 0-dim 스칼라(전부 per-tensor)임을
확인. Combined의 `s_mult`만 `torch.ones(in_channels)`(per-channel)이고,
`ac.soft=False`(rounding 확정) 이후 `ac.use_smult=True`가 실행되는 순서도
확인 — "이미 확정된 rounding 위에서 activation scale을 재조정"하는 구조가
맞음.

**코드 변경**: [src/quant/adaround.py](src/quant/adaround.py) —
`AdaRoundQuantConv2d.__init__`/`convert_to_adaround`에 `channelwise_smult`
파라미터 추가(기본 `True`=기존 동작 유지). `False`면 `s_mult`을 conv당
스칼라(`torch.tensor(1.0)`)로 생성. [pipeline/run_comparison.py](pipeline/run_comparison.py)에
`--smult-per-tensor` CLI 플래그 추가(Combined에만 적용).

**실험 1 — Combined: per-channel vs per-tensor**(seed0, 동일 GPU7):
COCO_AP -0.37, S_AP -0.40, H_eval_AP -0.38, Heval_flip/Top1_flip/UPIR/
CorrRate/LVIS_lost 전부 악화. per-tensor로 되돌리는 게 거의 순손실.

**실험 2 — per-tensor 조건에서 Combined vs 4개 baseline**(6-seed 0~5,
`runs/63_smult_pertensor/seed{0..5}_pertensor.log`):

| 지표 | naive | AdaRound | QDrop | BRECQ | Combined(per-tensor) 평균 |
|---|---|---|---|---|---|
| COCO_AP | 33.54 | 33.24 | 33.30 | 33.30 | **35.97** |
| Heval_flip | 11.80% | 11.03% | 11.14% | 10.81% | **9.32%** |
| Top1_flip | 0.86% | 0.68% | 0.68% | 0.67% | **0.64%** |
| UPIR | 0.34% | 0.30% | 0.26% | **0.23%** | 0.26% |
| lost | 90 | 77 | 77 | **64** | 71.8 |
| CorrRate | 22.75% | 23.98% | 29.45% | **29.88%** | 22.15% |
| LVIS_flip | 6.57% | 5.74% | 5.84% | 5.63% | **5.33%** |
| LVIS_lost | 232 | 233 | 236 | 227 | **220.0** |

**결론**: **per-tensor로 granularity를 맞춰도 AP/Heval_flip/LVIS_flip/
Top1_flip/LVIS_lost는 6-seed 내내 안정적으로 baseline을 이긴다** — Combined의
핵심 우위는 granularity 덕이 아니라 학습(margin/neighbor loss) 자체의
효과다. `lost`(raw count)와 CorrRate는 per-channel에서도 이미 있던
약점이라 per-tensor 때문에 새로 생긴 손실이 아님.

**실험 3 — s_mult 수렴 패턴 진단**(iters=300/calib=64로 축소, 정성적 확인용):

| | per-channel(확정) | per-tensor |
|---|---|---|
| conv별 평균의 평균 | 0.9834 | 0.9828(거의 동일) |
| median | 0.9942 | 1.0000 |
| min~max | 0.577~1.008 | 0.432~1.043(범위 훨씬 넓음) |

전체 평균 크기는 비슷한데 분포가 다름 — per-tensor는 "다수 conv는 안 움직이고
소수만 극단으로 튀는" 고분산 패턴. §7.5~7.8의 "일괄적 하향 dampening"과는
다른 실패 모드로 보임(단, iters/calib를 줄인 진단이라 완전 수렴 상태 아님).

**LVIS_AP 관련**: per-tensor Combined의 LVIS_AP(6-seed 평균 0.1951)가
per-channel(seed0 기준 0.2005)보다 낮음. 개선 여지로 논의한 두 방향(미실행):
1. `scale_reg_weight`를 per-tensor 스칼라 전용으로 재튜닝(현재 rw=10은
   per-channel 벡터 기준으로 맞춰진 값).
2. 더 근본적으로, class-selectivity를 activation이 아니라 weight 쪽
   per-output-channel(표준 INT8 엔진이 지원, 배포 문제 없음)로 옮기는
   구조 변경.

**상태**: `channelwise_smult` 플래그 구현 완료, `src`/`pipeline` 동기화
완료. **per-tensor를 새 기본값으로 채택하지는 않음** — per-channel(확정
설계)이 여전히 전반적으로 더 나음. claim 4의 결론은 "논문의 핵심 AP 주장은
공정한 비교에서도 유지된다"는 disclosure 근거로 활용 가능.

---

## Claim 5 — `margin_loss`가 class identity를 무시함(가장 근본적인 발견)

**주장**: `margin_loss`가 `sim_fp`/`sim_q`를 각자 독립적으로 `topk`해서
정렬된 값끼리만 비교 — 어느 class가 그 순위인지는 버려진다. top-1/top-2가
값을 맞바꾸는(class identity가 뒤집히는) flip이 일어나도 margin 패턴
자체가 같으면 loss=0. 즉 막으려는 대상(flip)이 목적함수에 안 보이는 경로가
있음. 제안된 단일 변경: `q_top = sim_q.gather(-1, fp_idx)`(FP가 정한 순서
그대로 Q의 값을 읽음 — identity 고정).

**검증**: [src/quant/promptcal.py:36-53](src/quant/promptcal.py#L36-L53)
확인 결과 정확함. 사용자 지적대로 "v2의 `decision_loss`(완전 discrete
top-1 매칭)가 과거 anti-transfer를 낸 전례"가 있어 개선을 보장하진
않는다는 전제하에 검증용으로 구현.

**코드 변경**: `margin_loss(sim_q, sim_fp, k, boundary_w, identity_aware=False)`
파라미터 추가 — `True`면 `fp_idx`로 `sim_q`를 gather. `optimize_promptcal_scale_neighbor`에
`identity_aware_margin` 파라미터로 전달(S/H_cal 양쪽 margin_loss 호출
모두에 적용). `pipeline/run_comparison.py`에 `--identity-aware-margin`
CLI 플래그 추가.

**실험 — 완전한 6-seed OFF vs ON 비교**(`runs/64_identity_margin/`,
`runs/65_identity_off/`, seed0 OFF는 `runs/62_anchorfix/seed0_after.log`
재사용):

| 지표 | OFF(확정, 6-seed 평균) | ON(identity-aware, 6-seed 평균) | delta | 판정 |
|---|---|---|---|---|
| COCO_AP | 36.36 | 36.30 | -0.06 | 노이즈 수준 |
| S_AP | 35.78 | 35.76 | -0.02 | 노이즈 수준 |
| H_eval_AP | 37.11 | 37.06 | -0.05 | 노이즈 수준 |
| LVIS_AP | 0.1979 | 0.1958 | -0.0021 | 미미 |
| APc | 0.1511 | 0.1534 | +0.0023 | 오히려 근소 개선 |
| APf | 0.2227 | 0.2190 | -0.0038 | 미미 |
| **Top1_flip** | 0.73% | **0.63%** | **-0.10pp** | **견고한 개선(6/6)** |
| **UPIR** | 0.27% | **0.19%** | **-0.08pp** | **견고한 개선** |
| **lost** | 70.5 | **64.3** | **-6.2** | **견고한 개선** |
| **LVIS_flip** | 5.44% | **5.28%** | **-0.16pp** | **견고한 개선(6/6)** |
| **LVIS_lost** | 221.5 | **205.3** | **-16.2** | **견고한 개선(6/6)** |
| Heval_flip | 9.33% | 9.34% | +0.01pp | 무변화(seed0의 개선은 outlier) |
| CorrRate | 25.01% | 23.95% | -1.06pp | 오히려 근소 악화(seed0의 대폭 개선은 outlier) |

**중요한 정정**: seed0 1개만 봤을 때는 "AP -0.18~0.28 손해에 Heval_flip
-0.22pp·CorrRate +7.47pp(BRECQ 첫 역전)"으로 보였으나, 6-seed 평균에서는
AP 손해가 거의 사라지고(-0.02~-0.06) Heval_flip/CorrRate의 극적 개선도
재현 안 됨 — seed0이 두 지표 모두에서 우연히 좋았던 draw였음. 대신
Top1_flip/UPIR/lost/LVIS_flip/LVIS_lost 다섯 개는 6-seed 내내 일관되게
개선됨.

**결론**: **AP는 사실상 그대로 유지하면서 decision-preservation 핵심
지표(특히 LVIS_flip/LVIS_lost — 진짜 held-out vocabulary에서의 성능)를
견고하게 개선**하는, 채택 가치가 있는 변경. v2의 `decision_loss`(discrete
cross-entropy)가 냈던 전면적 anti-transfer와는 다른, 훨씬 국지적이고
유리한 trade-off.

**개선 여지(미실행)**: `boundary_w`가 적용되는 마지막 인접쌍(k-1↔k, 진짜
top-k 경계)만 identity-aware하게 하고 나머지 tail(1↔2, 2↔3, 3↔4)은
기존 정렬-비교로 남기는 절충안 — flip과 무관한 tail 재정렬에 대한 불필요한
페널티를 줄여서 AP 손해를 더 줄일 수 있는지는 아직 확인 안 함.

**상태**: **§8.1 확정 설계 승격 후보** — 사용자 결정 대기 중. 코드는
opt-in 플래그로 구현 완료, `src`/`pipeline` 동기화 완료, 커밋 안 함.

---

## Claim 6 — neighbor 선택이 퇴화돼 있음(H_cal 20개가 사실상 유일한 후보)

**주장**: S(40)와 H_eval(20)을 제외하면 neighbor 후보 풀이 H_cal(20개)뿐인데,
40개 S class 각각이 `neighbor_k=5`개씩 뽑아도 합집합이 이미 H_cal 20개
전체를 커버해버린다. 즉 "text-embedding 최근접 이웃 선택"이 아니라 사실상
"H_cal 전체에 대한 one-sided hinge"인데, 논문 서술이 이 사실과 안 맞으면
리뷰어 공격 포인트가 된다.

**검증**: [src/quant/promptcal.py:446-452](src/quant/promptcal.py#L446-L452)
코드로 확인 — 정확함. 게다가 이미 `PROMPTCAL_CURRENT_MODEL.md` §8.5(줄
491-496)에 "**실측: 수정 후 neighbor_set=20**"으로 기록돼 있었음 — 다만
그때는 "neighbor_k≥5는 동치 클래스, 5로 유지하면 된다"는 **하이퍼파라미터
튜닝 결론**으로만 처리됐고, "그럼 이게 nearest-neighbor **selection**이라는
서술 자체가 부정확하다"는 **논문 서술 리스크**로는 짚어진 적이 없었음.
`cal_weight=1.0` 도입 이후로는 H_cal이 "직접 보호 대상"이자 "neighbor
후보 풀"이라는 이중 역할까지 겹쳐 있어 괴리가 한 번 더 커짐.

**결정**: **코드 변경 없음** — 숫자가 틀린 게 아니라 방법론 서술 문제라서
실험보다는 논문 텍스트 수정이 맞는 대응. 두 가지 선택지:
1. (권장, 비용 낮음) 방법론 섹션을 "40/20/20 split에서는 후보 풀이 H_cal로
   좁혀져 사실상 H_cal 전체에 asymmetric hinge를 적용하는 것과 동치이며,
   이는 `neighbor_k≥5`에서 결과가 불변하는 것으로 확인했다"처럼 있는
   그대로 서술.
2. (비용 높음) 후보 풀이 훨씬 큰 세팅(클래스 수 확대 또는 H_cal 비중
   축소)에서 "진짜 selection"이 일어남을 보여주는 보조 실험 추가.

**09-16 방법론 문구 초안**(영어, 옵션 1 채택, 논문 삽입용):

> We select up to `neighbor_k` semantically nearest non-target prompts per
> anchor class via text-embedding cosine similarity, excluding S and
> H_eval. In our 40/20/20 (S/H_cal/H_eval) split, this candidate pool
> reduces to H_cal in its entirety (20 classes) — with 40 S-anchors each
> drawing `neighbor_k=5` candidates from this pool, the union already
> covers all 20 H_cal classes, which we confirmed empirically: results are
> identical for `neighbor_k` ∈ {5, 8, 10}. The mechanism is thus
> equivalent, under our split, to applying the asymmetric hinge to H_cal in
> its entirety; the neighbor-selection formulation is written generally so
> that it yields genuine locality-restricted selection under splits with a
> larger candidate pool.

**상태**: 초안 작성 완료(옵션 1, 비용 낮은 쪽), 논문 Method 섹션에 삽입
대기(사용자 검토 필요). 옵션 2(보조 실험)는 계획 없음.

---

## Claim 7 — `measure_ap`의 클래스별 AP 매핑이 위치-zip이라 잘못됨

**주장**: `ultralytics`의 `metrics.box.maps`는 이미 클래스 id로 직접
인덱싱된 배열인데, 코드가 `dict(zip(ap_class_index, maps))`로 **위치**
기준으로 짝짓고 있다. `ap_class_index`가 `[0,1,...,nc-1]` 풀레인지일 때만
우연히 맞고, 일부 클래스가 빠진 subset이면 틀린 값이 매핑된다. 구버전
호환용 fallback `all_ap[:, 0]`도 AP50이라 `map`(AP@0.5:0.95)과 다른
지표라는 지적.

**검증**: `ultralytics.utils.metrics.Metric.maps` property 직접 읽어서
확인 — `maps = np.zeros(nc) + self.map; for i, c in enumerate(ap_class_index): maps[c] = self.ap[i]`.
이미 클래스 id로 직접 인덱싱되는 게 맞음. `all_ap[:, 0]`이 `ap50` property와
동일함도 확인.

**중요**: 이번 세션 결과 자체는 이 버그의 영향을 받지 않았다 —
`measure_ap()`가 `model.val(data=args.data, ...)`를 호출하는데,
`configs/coco_local.yaml`의 `val: val2017.txt`가 **`--eval-cap`과 무관하게
항상 고정된 val2017 전체 5000장**(전체 80클래스 포함 확인됨)을 가리킨다.
그래서 `ap_class_index`가 항상 풀레인지였고 position-zip이 우연히 맞았다.
다만 latent bug라 클래스 subset이 실제로 발생하는 상황(예: `--data`를
직접 커스텀 subset yaml로 바꾸는 경우)에서는 조용히 틀린 값을 낸다.

**수정 완료**: [pipeline/run_comparison.py](pipeline/run_comparison.py)의
`measure_ap()` — `per_class = {int(c): float(metrics.box.maps[c]) for c in metrics.box.ap_class_index}`로
직접 인덱싱, fallback도 제거(현재 ultralytics 8.4.121은 `.maps`를 항상 가짐).

**동일 버그가 있던 곳**: `scripts/58_full_baseline_official_data.py`
(포팅 원본) — **09-16 동기화 완료**.

---

## Claim 8 — `switch_vocab`이 `YOLOWorld.set_classes` wrapper를 우회함

**주장**: `switch_vocab`이 `model.model.set_classes(names, cache_clip_model=False)`로
내부 모델을 직접 호출해서, wrapper(`YOLOWorld.set_classes`)가 하는
`self.model.names = classes`/predictor 리셋을 건너뛴다. 지금은
`verbose=False`+예측 후처리가 `nc=0`이라 문제없지만 시각화/verbose를 켜면
이름-인덱스 불일치로 `IndexError`.

**검증**: `YOLOWorld.set_classes` 소스 직접 읽어 확인 —
`self.model.set_classes(classes)` 외에 `self.model.names = classes`와
`if self.predictor: self.predictor.model.names = classes`까지 한다.
`switch_vocab`은 내부 모델의 `set_classes`만 호출(wrapper가 `cache_clip_model`
인자를 안 받아서 직접 호출한 것으로 보임)해서 나머지 두 단계가 스킵됨 — 정확한 지적.

**수정 완료**: `switch_vocab`에 `model.model.names = list(names)`와
`model.predictor = None` 추가.

**동일 문제가 있던 곳**: `scripts/58_full_baseline_official_data.py` —
**09-16 동기화 완료**.

---

## Claim 9 — `requirements.txt`의 `ultralytics>=8.3.0` 버전 미고정

**주장**: 검증된 버전(8.4.121)과 다르게 `requirements.txt`가 열려 있어서,
이후 버전(예: 8.4.152)에서 `fuse()`가 바뀌면(is_qat 분기, `Detect.fuse()`)
재현성이 깨질 수 있다.

**검증**: 설치된 버전 8.4.121 확인, `requirements.txt`가
`ultralytics>=8.3.0`인 것 확인. 8.4.152의 구체적 변경 내용은 오프라인이라
직접 확인 못 했지만, 미고정 자체가 재현성 리스크라는 지적은 타당.

**수정 완료**: `requirements.txt` — `ultralytics==8.4.121`로 고정.

---

## Claim 10 — 기타 (LVIS AP 프로토콜, calib_time 구성, 데이터셋 사본 일치)

**(a) LVIS AP 프로토콜**: `lvis-api`의 `LVISEval`이 계산하는 게 표준 LVIS
AP(federated dataset 특성상 이미지별로 "exhaustively annotated"인
카테고리만 false positive로 벌점)이고, YOLO-World 논문이 쓰는 "fixed AP"와
같은 프로토콜인지는 코드로 판단하기 어려운 도메인 지식 영역 — `LVISEval`
생성자에 별도 "fixed" 토글은 안 보임. **논문에 사용 프로토콜을 명시해야
한다는 지적은 타당**, 추가 검증은 안 함.

**(b) calib_time 구성**: 코드 확인 — `t0 = time.perf_counter()`가
`build()` 호출 전체를 감싸므로, 모델 로드+`set_classes`(CLIP 텍스트
인코딩)+calibrate+모드별 최적화(Combined는 AdaRound+promptcal 둘 다)가
전부 포함되는 게 맞음. 지적대로 "순수 최적화 시간"이 아니라 "빌드 전체
시간"으로 해석해야 함.

**(c) 데이터셋 사본 일치 + 추가 발견**: `coco_local.yaml`(COCO-80 AP용,
`/data/taeho/coco_ultra`)과 `--coco-root`(flip/GT용, `/data/taeho/coco_datasets`)의
val2017이 **파일명 기준으로 완전히 동일한 5000장**임을 `diff`로 확인(차이
없음). 다만 이 과정에서 **claim 10 원래 질문에는 없던 문제를 하나 더
발견**: `measure_ap()`(COCO-80용, 위 claim 7 설명 참고)가 `--eval-cap`을
전혀 안 보고 `coco_local.yaml`의 고정 `val2017.txt`를 써서 항상 COCO
val2017 전체 5000장으로 AP를 재는 반면, Heval_flip/Top1_flip/UPIR/lost
등과 **LVIS_AP/APr/APc/APf는 `lvis_probe_paths`가 `--eval-cap`으로 이미
잘린 `probe_paths`에서 파생되므로 전부 `--eval-cap`을 따른다** — 09-16
재확인 결과 `--eval-cap`이 안 먹는 건 COCO_AP/S_AP/H_eval_AP 세 개뿐이고
LVIS 쪽은 원래도 정상적으로 적용되고 있었다(처음 이 절을 쓸 때 LVIS_AP도
영향받는다고 잘못 적었던 걸 정정). **이번 세션의 모든 `--eval-cap 1000`
실험은 COCO_AP/S_AP/H_eval_AP만 풀스케일(5000장)이고 나머지 전부(LVIS
AP 포함)는 1000장 기준**이었다 — 같은 seed 내 5개 방법 비교는 전부 동일
조건이라 방법 간 비교 자체는 안전하지만, "eval-cap=1000 실험"이라는
표현이 그 세 지표에는 부정확했다. **09-16 결정 및 수정**: 코드 동작은
그대로 유지(AP eval 자체가 병목이 아니라서 굳이 줄일 이유가 없고,
오히려 COCO-80 AP는 풀스케일로 재는 게 더 정확한 결과라 유지가 낫다고
판단) — 대신 `--eval-cap` 사용 시 결과 표 위에 어떤 지표가 영향받는지
알려주는 안내 문구를 추가함(`pipeline/run_comparison.py`).

**상태**: (a)(b)는 문서화만 필요(코드 정상), (c)는 확인 완료(데이터 일치)
+ 새 발견(COCO-80 AP만 `--eval-cap` 미적용)은 **동작 유지 + 안내 문구
추가로 09-16 해결**.

---

## 09-16 추가 발견 (a)(b)(c) — cal_weight confound, identity-aware의 intrusion 미탐지, 실행 편의

**(a) `cal_weight=0` ablation의 confound**: `run_comparison.py`의 `main()`이
`build()`를 호출할 때 `cal_idx=(H_cal if args.cal_weight > 0 else None)`으로
넘기고 있었다. `promptcal.py`의 `train_cols`(claim1 수정으로 도입된
confident-anchor 선정 기준)는 `cidx is None`이면 `pidx`(S, 40개)만 쓰고
아니면 `torch.cat([pidx, cidx])`(S∪H_cal, 60개)를 쓴다. 그래서
`cal_weight=0`을 주면 `ml_cal` 항(H_cal margin_loss)만 꺼지는 게 아니라
**anchor 선정에 쓰이는 열 개수까지 40→60으로 같이 바뀌어서**, `cal_weight`
0 vs 1 비교가 두 변수를 동시에 흔드는 confound가 됐다.
`promptcal.py`(내부 `if cidx is not None and cal_weight > 0:` 게이트)가
docstring에 적어둔 "`cal_weight=0`이면 결과가 같아야 한다"는 불변식도 이
경로로 깨져 있었다(정확히는 claim1의 `train_cols` 도입이 09-15에, `cal_idx`
조건부 전달이 09-12에 각각 따로 생겨서 서로 상호작용을 안 맞춰본 것).

**검증**: 코드로 확인, 정확한 지적. 다만 **이번 세션에서 지금까지 보고한
결과는 전부 영향 없음** — claim1/4/5의 모든 실행이 `--cal-weight`를 아예
지정 안 해서 기본값(1.0, `cal_idx=H_cal`)을 썼기 때문에 이 confound
경로를 안 탔다. `PROMPTCAL_CURRENT_MODEL.md` §8.10의 `cal_weight=0` 원본
검증(09-12)도 claim1의 `train_cols` 자체가 그때는 없었으니(80열 전체
기준) 영향 없음. **앞으로 `--cal-weight 0`으로 재실행할 계획이 있었다면
그 결과만 무효**.

**수정 완료**: `run_comparison.py`에서 `cal_idx`를 조건 없이 항상 `H_cal`로
전달하도록 변경 — `ml_cal` 추가 여부는 `promptcal.py` 내부의
`cal_weight > 0` 게이트에만 맡긴다. 이제 `train_cols`는 `cal_weight` 값과
무관하게 항상 S∪H_cal(60)로 고정되고, `cal_weight`는 순수하게 "H_cal에
margin_loss를 추가로 거냐"만 토글하는 단일 변수가 됐다.

**(b) identity-aware margin이 기존 버전의 상위호환이 아님 — intrusion 미탐지**:
claim5의 `identity_aware=True`는 `fp_idx`로 `sim_q`를 gather하는데,
`fp_idx`는 FP의 top-(k+1) 안에 있는 class만 가리킨다. 그래서 **FP top-(k+1)
밖에 있던 class가 Q에서 값이 치솟아 실제 1등을 빼앗는 경우(intrusion)를
그 열 자체를 아예 안 봐서 완전히 놓친다.** 반면 기존 정렬 버전(`topk`를
Q에도 독립적으로 적용)은 identity는 몰라도 Q 자신의 top-1 값이 커지는 걸
통해 intrusion을 부분적으로 잡아냈었다 — 즉 identity-aware는 "swap은 잡고
intrusion은 놓치는" 다른 trade-off였지 기존 버전의 순수 상위호환이 아니다.

**검증**: 코드 리뷰 + 합성 텐서로 직접 재현 — FP top-6 밖의 class(index 15)를
Q에서 100.0으로 스파이크시켰을 때, 기존 `identity_aware` 구현은 그 값을
전혀 못 보고(gather가 index 15를 안 가져옴) loss가 낮게 유지됐다.

**수정 완료**: 제안하신 대로 마지막 열(boundary_w가 걸리는 FP rank-(k+1)
자리)만 "FP top-k(rank 1..k) 밖에 있는 class 중 Q에서 가장 높은 값"으로
바꿔치기하도록 `margin_loss`를 수정:
```python
if identity_aware:
    q_top = sim_q.gather(-1, fp_idx)
    mask = torch.zeros_like(sim_q, dtype=torch.bool).scatter_(-1, fp_idx[:, :-1], True)
    q_out = sim_q.masked_fill(mask, float("-inf")).max(-1).values
    q_top = torch.cat([q_top[:, :-1], q_out[:, None]], dim=1)
```
앞쪽 k개 열은 그대로 identity-aware gather(swap 탐지), 마지막 열만 "top-k
밖 전체 최댓값"(intrusion 탐지) — 둘 다 하나의 boundary 비교에 담김. 합성
텐서로 재검증: intrusion 시나리오에서 loss가 정상적으로 크게 나옴(전:
낮게 유지 → 후: 5880 수준), swap 시나리오(claim5 원래 케이스)도 여전히
정확히 탐지됨(loss 0 → 1.0). **주의: claim5에서 이미 돌린 6-seed
identity-aware 결과(`runs/64_identity_margin/`)는 이 수정 전 버전으로
나온 것이라, intrusion 미탐지 상태에서의 결과다 — 재현하려면 재실행
필요.**

**(c) 실행 편의**: 두 가지 지적 모두 확인 후 수정.
- 결과 표에 어떤 플래그를 켜고 돌렸는지 안 남아서 `--identity-aware-margin`/
  `--smult-per-tensor`를 켠 로그와 안 켠 로그가 구분이 안 됐음 → `main()`
  시작 시 `print(f"[args] {vars(args)}")` 추가.
- `conditions = ["naive", "adaround", "qdrop", "brecq", "combined"]`가
  하드코딩돼 있어서 Combined 변형 하나만 확인할 때도 QDrop/BRECQ(seed당
  900~1400s대)까지 매번 다시 빌드했음 → `--conditions`(쉼표 구분, 기본값은
  기존과 동일한 5개 전부) 추가. `models["adaround"]`로 모델 크기를 재던
  하드코딩도 `--conditions`로 일부만 돌 때 KeyError 안 나게 존재하는
  AdaRound 계열 아무거나(없으면 첫 조건) 쓰도록 같이 수정.

**상태**: (a)(b)(c) 전부 수정 완료, 스모크 테스트(`--conditions
naive,combined --identity-aware-margin`, calib=8/eval-cap=16, GPU7)로
args 출력·조건 서브셋·에러 없음 확인. `src`/`pipeline` 동기화 완료.
git 커밋 필요.

---

## 09-16 추가 발견 (d)(e) — `scripts/60`이 §8.1 대체 후에도 옛 설계로 돔, cal_weight 게이트 문서/코드 불일치

**(d) `scripts/60_hparam_sweep_official_data.py`가 §8.1 확정 설계 변경(claim4/5
채택) 이후에도 여전히 per-channel + identity-unaware로 돎**: 이 스크립트는
`convert_to_adaround(m.model)`과 `optimize_promptcal_scale_neighbor(...)`를
`channelwise_smult`/`identity_aware_margin` 인자 없이 호출해서, 라이브러리
기본값(`channelwise_smult=True`, `identity_aware_margin=False` — 의도적으로
안 바꾼 값, claim5-c 관련 기록 참고)을 그대로 물려받는다. `run_comparison.py`는
CLI 기본값을 뒤집어서 해결했지만(§부록D 완료 항목 2), `scripts/60`은 별개
스크립트라 그 수정이 안 미쳤다 — **앞으로 per-tensor용으로
`scale_reg_weight`를 재튜닝할 때 이 스크립트를 그대로 쓰면 로그에 아무
표시 없이 엉뚱한(§8.11로 superseded된) 설계를 스윕하게 되는 실사용 리스크**였다.

조사 중 **관련된 두 번째 버그**도 같이 발견: `cal_idx`가 `--param cal_weight`로
그 자체를 스윕할 때만 넘겨지고 있어서, 다른 파라미터(scale_reg_weight 등)를
스윕하는 동안은 `--cal-weight` 기본값(1.0)이 있어도 `cal_idx=None`이라
H_cal 보호가 통째로 꺼지고 confident-anchor 풀(`train_cols`)도 S(40)로
좁아진 채 돌고 있었다 — claim5-a와 정확히 같은 종류의 문제가 이 스크립트에도
있었던 것.

**수정 완료**: `--smult-per-tensor`/`--identity-aware-margin` 플래그 추가
(기본 True, `run_comparison.py`와 동일 패턴), `build_adaround_base`/
`build_combined_from_base`에 전달. `cal_idx`는 이제 스윕 대상과 무관하게
항상 넘기고 `cal_weight` 값만 스윕 시 덮어씀. 스모크 테스트로 검증 —
`neighbor_weight=1.0`(확정값) 행이 `run_comparison.py` 기본값 스모크 테스트와
**수치까지 정확히 일치**함을 확인(Heval_flip=10.34%, Top1_flip=1.08%,
UPIR=1.23%, lost=2). 커밋 `bcf0196`.

**(e) `cal_weight` 게이트 문서/코드 불일치**: 이 claims 문서(claim5-a 절)가
`promptcal.py`에 `if cidx is not None and cal_weight > 0:` 게이트가 있다고
적었는데, 실제 코드는 `if cidx is not None:`만 있고 `cal_weight`는 그 뒤에
곱셈으로만 적용되고 있었다. `cal_weight=0`이면 `0 * ml_cal = 0`이라
**수치 결과는 문서에 적은 것과 동일**(버그 아님) — 다만 `cal_weight=0`에서도
매 iteration `margin_loss`를 불필요하게 계산하고 있었다는 점, 그리고
코드와 문서가 안 맞았다는 점은 사실이었다. `and cal_weight > 0`을 실제
게이트에 추가해서 문서와 일치시키고, 그 낭비 연산도 없앴다(수치 결과는
불변 — 0을 더하던 걸 아예 안 더하게 바뀐 것뿐). `src`/`pipeline`
`promptcal.py` 동기화 완료. 커밋 `bcf0196`(위 (d)와 같은 커밋).

**상태**: (d)(e) 전부 수정·검증·커밋 완료.

**09-16 추가 확인(f)**: "pipeline/ 디렉토리랑 문서가 다 최신화됐나"는
질문에 답하려고 `optimize_promptcal_scale_neighbor`를 호출하는 모든
스크립트를 `grep -rl`로 전수 조사했다. `scripts/58_full_baseline_official_data.py`도
(d)와 정확히 같은 문제(channelwise_smult/identity_aware_margin 미전달 +
cal_idx confound)가 있었고, 같은 "_official_data" 패밀리인
`scripts/59_rw_sweep_official_data.py`·`scripts/61_combo_grid_official_data.py`도
동일 문제(cal_idx는 애초에 이 둘엔 없어서 해당 없음)였다. 전부 수정·
스모크 테스트·커밋 완료(`bc40efa`, `105e16d`) — 세 스크립트의 `combined`
행 수치가 서로, 그리고 `run_comparison.py`/`scripts/60`과도 전부 정확히
일치함을 확인. **09-08~09-15 시절의 개별 진단 스크립트(`scripts/41~57`
등, "_official_data" 접미사 없음)는 의도적으로 안 건드림** — 대부분 특정
과거 설계(예: per-channel 도입 이전/이후 비교)를 그 자체로 검증하는 게
목적이라 "최신 설계로 갱신"하면 오히려 그 스크립트의 원래 용도(역사적
재현)가 깨진다. 이제 `_official_data` 패밀리(`58/59/60/61`) 전부와
`pipeline/run_comparison.py`가 전부 §8.1 확정 설계를 기본으로 사용한다.

---

## Claim 11 — LVIS_AP가 공개 수치의 절반 수준 (09-17, 사용자 발견)

**주장**: LVIS는 long-tail이라 COCO보다 AP가 낮은 게 정상이지만, 우리
FP32의 LVIS_AP(0.126)는 YOLO-World-S(O365+GoldG) 공개 minival 수치(AP
0.243)의 절반 수준이다. COCO_AP(36.80)는 공개 수치(~37.4)와 거의 일치하니
COCO 파이프라인은 정상이고 LVIS 평가만 어긋난다. 원인 후보 셋:
1. NMS `multi_label` 불일치 — 가장 유력.
2. Fixed AP가 아니라 이미지당 300개 cap standard AP.
3. LVIS 프롬프트 문자열이 동의어를 `/`로 합친 한 문자열(확신도 낮음).

**검증**:
1. **정확함.** `DetectionValidator.postprocess`(COCO_AP가 쓰는
   `model.val()` 경로)는 NMS를 `multi_label=True`로 호출하는데,
   `DetectionPredictor.postprocess`(LVIS_AP가 쓰는 `model.predict()`
   경로)는 이 인자를 아예 안 넘겨서 `non_max_suppression`의 기본값
   `multi_label=False`가 조용히 적용되고 있었다. `model.predict(...,
   multi_label=True)`로 넘겨도 `DetectionPredictor`가 `self.args`에서
   그 값을 안 읽어서 무시된다 — `nms` 모듈 함수 자체를 patch해야 실제
   적용됨을 확인.
2. **정확함.** `predict_lvis_results`의 `max_det=300` 기본값,
   `run_lvis_eval`의 `LVISResults(..., max_dets=300)` — 이미지당 300개
   cap의 standard AP가 맞음.
3. `lvis.yaml` 1203개 중 460개(38%)가 실제로 `"aerosol can/spray
   can"`처럼 `/`로 합쳐진 문자열임을 확인(심한 경우 7단어 연쇄). 영향은
   있겠으나 1·2보다 후순위라는 판단에 동의.

**실측(공식 4809장 minival, FP32)**:

| 설정 | AP | APr | APc | APf |
|---|---|---|---|---|
| 기존(버그) | 0.1260 | 0.0310 | 0.0808 | 0.1831 |
| multi_label=True만 | 0.2326 | 0.1579 | 0.2148 | 0.2620 |
| **multi_label=True + FixedAP** | **0.2589** | **0.1767** | **0.2454** | **0.2856** |
| 공개 수치 | 0.243 | 0.166 | 0.221 | 0.277 |

0.126→0.259로 공개 수치와 6% 이내로 일치 — 미스터리 해소. 기여도는
multi_label이 격차의 ~80%(+0.107), Fixed AP가 ~20%(+0.026). **APr이 가장
크게 움직임**(0.031→0.177, 5.7배) — §9의 "Combined가 APr 최하위" 논의가
이 버그 위에서 나온 것이라 재측정 전엔 유효하지 않음.

**수정 완료**: `pipeline/run_comparison.py`, `scripts/58~61` 전부
`predict_lvis_results`에서 `ultralytics.utils.nms.non_max_suppression`을
`predict` 루프 동안 `multi_label=True`로 monkey-patch, `run_lvis_eval`을
Fixed AP(클래스당 상위 10000개 유지, 이미지당 cap 없음)로 전환,
`max_det` 기본값 300→1000. 전체 파이프라인 스모크 테스트로 에러 없음
확인. 커밋 `adf9489`(pipeline+scripts/58), `ac50d82`(scripts/59~61).

**영향 범위**: COCO_AP는 원래도 `model.val()`을 썼으니 무관. **LVIS_AP/
APr/APc/APf는 이번 세션 전체(§8.1 확정 표 포함)가 재측정 대상**이다.
5개 방법이 전부 같은 `predict()` 경로를 타서 방법 간 순위 방향은 유지될
가능성이 높지만, 좁은 마진 우위(BRECQ 대비 등)는 뒤집힐 수 있다.
`scripts/59~61`의 `BASELINE_REF` 하드코딩 값(옛 LVIS_AP)도 이제 stale —
아직 안 고침, 재사용 전 재확인 필요.

**상태**: 코드 수정·커밋 완료. **§8.1 확정 표의 LVIS 재측정(6-seed×5조건)
완료**(`runs/68_lvis_fix_fullscale/seed{0..5}_lvisfix.log`, 물리 GPU
0/4/5/6/7 동일 모델). 결과:

| method | LVIS_AP(6-seed) | APr(6-seed) |
|---|---|---|
| FP32 | 0.2589 | 0.1767 |
| naive | 0.2342 | 0.1704 |
| AdaRound | 0.2338 | 0.1553 |
| QDrop | 0.2323 | 0.1652 |
| BRECQ | 0.2333 | 0.1679 |
| **Combined** | **0.2454~0.2498(0.2476)** | **0.1680~0.1736(0.1704)** |

baseline 4개는 calib 256장 샘플링이 결정적이라 6-seed 값이 완전히
동일(seed는 Combined의 neighbor/margin 학습에만 영향). **결론 완전히
뒤집힘**: Combined가 LVIS_AP에서 4개 baseline 전부를 이기고(BRECQ 대비
+0.0143), APr(rare class)도 naive와 사실상 동률(0.1704=0.1704)이면서
AdaRound/QDrop/BRECQ는 확실히 이긴다 — 구버전의 "Combined가 rare class
에서 유일하게 naive에 진다"는 서술은 LVIS 평가 버그의 산물이었음이
최종 확인됨. `PROMPTCAL_CURRENT_MODEL_V2.md` §8/§9에 반영 완료.
`scripts/59~61`의 `BASELINE_REF` 하드코딩 값(옛 LVIS_AP)은 위 표의
naive/AdaRound/QDrop/BRECQ 값으로 아직 안 고침 — 그 스윕들을 재사용하기
전에 갱신 필요.

---

## Claim 12 — baseline엔 activation scale을 학습하는 손잡이가 아예 없다 (09-17, 사용자 발견)

**주장**: Combined가 학습하는 건 activation scale(s_mult)인데, AdaRound/QDrop/
BRECQ 구현은 weight rounding(alpha)만 최적화하고 activation scale은 min-max로
고정한다. claim4는 granularity(per-channel vs per-tensor)만 맞췄을 뿐, "손잡이가
있느냐 없느냐" 비대칭은 그대로 남아있다. baseline이 naive보다 낮은 것, s_mult
min=0.43(outlier 보정 추정) 등이 정황 증거.

**검증**:
1. 코드: `ActObserver`의 `scale`/`zero_point`는 전부 `register_buffer`(비학습),
   `optimize_adaround`/`optimize_brecq`의 optimizer는 `[alpha]`만 담음 — 확인됨.
2. 문헌: [BRECQ(Li et al., ICLR 2021)](https://openreview.net/pdf?id=POWv6hDd9XH)
   원 논문이 "adopts adaptive rounding for weights and learned step size
   (Esser et al., 2020, LSQ) for activation step size"라고 명시 — 원 논문은
   activation LSQ를 포함함을 확인. [QDrop(Wei et al., ICLR 2022)](https://arxiv.org/pdf/2203.05740)도
   BRECQ의 block reconstruction 프레임워크를 그대로 물려받아 같은 LSQ 메커니즘을 씀.
   [AdaRound(Nagel et al., ICML 2020)](https://arxiv.org/pdf/2004.10568)는 순수
   weight-rounding 방법이라 activation LSQ가 원래 없음 — 이 축소는 AdaRound에는
   해당 안 됨, BRECQ/QDrop 두 개만 해당.
3. Control 실험(`--control-mse`, `promptcal.py` `control_mse=True`): Combined의
   margin_loss/neighbor-hinge를 전부 빼고 s_mult 메커니즘만 유지한 채 순수
   `F.mse_loss(sim_q[train_cols], sim_fp[train_cols])`로 최적화. 3-seed(구
   재구성 코드 기준, `runs/69_control_mse/`) 결과:

   | 지표 | BRECQ | Combined(margin) | Combined(MSE-control) |
   |---|---|---|---|
   | COCO_AP | 33.30 | 35.96 | **36.47** |
   | LVIS_AP | 0.2333 | 0.2476 | **0.2521** |
   | Heval_flip | 9.71% | **9.05%** | 9.93% |
   | Top1_flip | **0.71%** | **0.633%** | 0.713% |
   | LVIS_flip | **5.53%** | **5.06%** | 5.63% |

   AP 계열은 MSE-control이 margin_loss 버전보다도 앞섬(손잡이 존재 자체가 AP
   이득의 주 원인). 그런데 decision-preservation 계열(Heval_flip/Top1_flip/
   LVIS_flip)은 MSE-control이 BRECQ보다도 못하거나 비슷함 — margin_loss의 실제
   기여는 AP가 아니라 decision-preservation에 있다는 뜻.
4. `--learn-act-scale`(`optimize_adaround`/`optimize_brecq`의 `learn_act_scale=True`,
   기존 `s_mult`/`_quantize_smult` 메커니즘 재사용, `channelwise_smult=False`로
   Combined와 granularity 통일): 1-seed(구 재구성 코드 기준, `runs/70_learn_act_scale/`)
   결과:

   | 지표 | BRECQ(원본) | BRECQ+LSQ | AdaRound+LSQ | QDrop+LSQ | Combined(margin) |
   |---|---|---|---|---|---|
   | COCO_AP | 33.30 | 35.46 | **36.10** | **36.18** | 36.03 |
   | Heval_flip | 8.99% | **7.09%** | 7.79% | 9.64% | 8.08% |
   | lost | 327 | **255** | 311 | 406 | 356 |

   AdaRound+LSQ/QDrop+LSQ가 이미 COCO_AP에서 Combined를 앞지르고, BRECQ+LSQ는
   decision-preservation 전 지표에서 Combined(margin_loss)를 이김 — claim4/5로
   완결됐다고 봤던 "Combined의 우위"가 상당 부분 "baseline에 없는 손잡이"에서
   온 것일 수 있음을 시사.

**상태**: 메커니즘 확인·구현 완료. **claim13(재구성 loss 버그) 발견으로 위 3/4번
수치는 전부 재검증 대상** — 구 재구성 코드(정규화 버그 있음) 위에서 측정된 것이라,
claim13 수정 후 baseline 자체의 절대 성능이 달라지면 이 비교도 다시 그려야 한다.
act_lr(`DEFAULT_ACT_LR`)도 처음엔 공식 소스를 잘못 인용(4e-5, 실제는 4e-4)했다가
사용자가 지적해 정정 — 최종적으로 절대값 환산 대신 Combined의 s_mult lr(1e-2)과
맞추고 CosineAnnealing은 유지(수렴 실측 근거, `PROMPTCAL_CLAIMS` 이 문서 서술과
별개로 대화 로그에 diagnostic 스크립트 2개 존재, 저장소엔 미커밋).

---

## Claim 13 — AdaRound/BRECQ 재구성 loss 정규화가 공식 대비 축소돼 있었다 (09-18, 사용자 발견)

**주장**: `(pred-target).pow(2).mean()`은 BRECQ 공식 `lp_loss`(채널축 sum, 나머지
mean) 대비 정확히 C_out배 작다. `reg_loss`(alpha 정규화)는 이 축소 영향을 안
받으므로, 상대적으로 rounding 정규화가 재구성 항을 압도해 alpha가 초기값
(=round-to-nearest)에서 거의 못 움직인다. 여기에 lr(1e-2→1e-3)·reg_weight
(1e-3→0.01, 둘 다 공식값)도 같이 고치고 warmup 20%를 추가하면 반올림 결정이
실제로 바뀐다(nearest 대비 flip 0.069%→1.5%대).

**검증**:
1. 공식 repo(`yhhhli/BRECQ`) `layer_recon.py`/`block_recon.py`/`main_imagenet.py`를
   직접 fetch해서 대조. `main_imagenet.py`의 실제 실험 kwargs가
   `warmup=0.2, weight=0.01, b_range=(20,2)`를 layer/block reconstruction
   **둘 다에** 똑같이 넘김(각 함수 자체의 내부 기본값 0.0/0.001은 실제로 안 쓰이는
   placeholder) — 수정된 `DEFAULT_WARMUP=0.2`/`DEFAULT_REG_WEIGHT=0.01`이 정확히
   일치함을 확인.
2. `lp_rec_loss`(채널축 sum, 나머지 mean) 정규화가 공식 `lp_loss(p=2,
   reduction='none')`와 동일한 정규화임을 GitHub 소스로 확인.
3. alpha lr: 공식이 `Adam(opt_params)`로 lr 명시 안 함 → Adam 기본값 1e-3 —
   `DEFAULT_LR=1e-3`과 일치.
4. **한 군데 인용 오류 발견·정정**: activation LSQ lr 주석이 "공식은 4e-5"라고
   적혀 있었는데, 이건 `block_recon.py` 함수 자체의 내부 기본값이고 실제
   `main_imagenet.py --lr` 기본값은 **4e-4**(10배 차이). 사용자가 지적. 이
   모델에서 `a_obs.scale` 중앙값을 실측하면 0.0576(ImageNet 가정 0.024의
   2.4배)이라 올바른 소스로 환산하면 act_lr≈7e-3이 나오지만, 이 절대값 변환
   자체가 불필요한 불확실성을 더한다는 사용자 지적에 따라 최종적으로는
   Combined의 s_mult lr(1e-2)과 정확히 맞추는 쪽으로 결정(§ claim12 마지막
   문단 참고). 수렴 실측(conv 2개, 1000 iters): act_lr=2e-3+cosine(구현 전
   버전)은 마지막 10%에서 이미 완전히 수렴(변동폭 0.0000~0.0003), lr=1e-2를
   scheduler 없이 쓰면 끝까지 진동(변동폭 0.005~0.05) — CosineAnnealing은
   유지하고 peak lr만 1e-2로 올림.
5. **정황 증거**: 이 세션 내내 확정 §8 표에서 AdaRound(33.24)가 naive(33.54)
   보다 낮았던 게 계속 이상했는데("AdaRound가 naive보다 못한 모델을 만든다"),
   09-08 문서에도 이게 "검증 필요"로만 남아있었다. `flip_rate()` 진단
   (round-to-nearest 대비 반올림이 실제로 바뀐 비율)이 수정 전 0.069%였다는
   재현 실험 결과와 정확히 들어맞음 — AdaRound가 사실상 naive rounding과
   같은 모델을 만들고 있었다는 뜻.

**수정 완료**: `src/quant/adaround.py`(`lp_rec_loss`, `temp_decay`,
`DEFAULT_LR/REG_WEIGHT/WARMUP/ACT_LR`, `flip_rate()` 진단, `optimize_adaround`
전체), `src/quant/brecq.py`(`optimize_brecq` 전체, `_to_cpu`/`_to_dev`/`_mix`
재귀 처리로 다중 입력 list/tuple OOM 버그도 같이 수정, QDrop을 여기로 이전),
`pipeline/run_comparison.py`(QDrop 조건이 `optimize_brecq(qdrop_prob=...)`를
쓰도록 변경 — 이전엔 layer-wise `optimize_adaround`에 qdrop_prob만 얹어서
QDrop 원 논문의 재구성 단위(block-wise)와 달랐음). `pipeline/quant/` 사본 동기화
완료. 스모크 테스트(iters=50, 4조건) crash 없이 통과 확인.

**영향 범위**: `optimize_adaround`가 Combined 자신의 weight-rounding 1단계에도
쓰이므로, **AdaRound/QDrop/BRECQ뿐 아니라 Combined의 COCO_AP/S_AP/H_eval_AP까지
전부 재측정 대상**이다 — claim11(LVIS)보다 범위가 넓다. `scripts/58~61`은 이
수정을 반영 안 함(QDrop이 여전히 layer-wise, `learn_act_scale` 파라미터 자체가
없음) — claim5-d와 같은 성격의 staleness, 아직 안 고침.

**상태**: 코드 수정 완료(커밋 전, uncommitted). **6-seed 재측정 완료**
(`runs/71_recon_fix_review/seed{0..5}_full.log`, 09-18). 결과:

| method | COCO_AP(구코드) | COCO_AP(신코드) |
|---|---|---|
| naive | 33.54 | 33.54(불변) |
| AdaRound | 33.24(고정) | 33.26~33.41(33.34) |
| QDrop | 33.30(고정) | 33.20~33.30(33.24) |
| BRECQ | 33.30(고정) | 33.04~33.19(33.15) |
| Combined | 35.86~36.07(35.98) | 35.53~35.95(35.83) |

**가설 반박됨**: `flip_rate()` 진단으로 alpha가 실제로 5~7% 움직임을 확인했는데도
(구코드 <1%) AdaRound/QDrop/BRECQ 전부 6-seed 내내 naive보다 여전히 낮다.
"재구성 loss 정규화 버그가 AdaRound<naive의 원인"이라는 애초 가설은 데이터로
반박됐다. baseline 4개도 이제 alpha 최적화에 매 iteration 무작위 샘플링이
들어가 seed마다 값이 달라진다(이전엔 결정적 순환이라 6-seed 전부 동일했음).

**사용자 지적(09-18)으로 프레임 수정**: 이 수정은 성능을 올리려는 게 아니라
AdaRound/QDrop/BRECQ를 원 논문대로 정확히 구현하기 위한 것(claim4/5와 같은
원칙 — 공정성/정확성 문제는 결과 방향과 무관하게 채택). "결과가 나빠졌으니
되돌릴지"는 애초에 잘못된 질문이었다 — 신코드가 더 정확한 구현인 이상 그대로
채택하고, "정확히 구현해도 baseline이 naive를 못 이긴다"는 걸 이상 현상이
아니라 실제 발견으로 취급한다("reconstruction 최적화가 detection AP를
보장하지 않는다"는 논문 핵심 주장과 같은 방향). `PROMPTCAL_CURRENT_MODEL_V2.md`
§4/§8 갱신 완료. decision-preservation 4개 지표(Top1_flip/lost/LVIS_flip/
LVIS_lost)에서 이제 QDrop·BRECQ 둘 다 Combined를 이긴다(구코드는 BRECQ만) —
§8 참고.

---

## Claim 14 — Combined의 1단계(AdaRound)가 2단계 보호 영역 밖으로 손상을 샌다 (09-18, 협업으로 도출)

**배경**: claim13 수정 직후 "Combined가 AP는 이기는데 Top1_flip/lost/
LVIS_flip/LVIS_lost는 QDrop·BRECQ에 진다"는 게 확인됐다(claim13 상태 표
참고). 사용자가 "우리가 주장하던 부분(flip/lost)에서 지는 게 아이러니하다"고
지적 — 이 절이 그 원인 규명과 해결.

**가설(사용자+대화로 도출)**: Combined도 s_mult 학습(2단계) 전에 자기 자신의
weight rounding으로 `optimize_adaround`(1단계)를 쓴다. claim13 전엔 이
함수가 거의 안 움직였으니(flip<1%) 1단계가 사실상 no-op이었는데, claim13
수정으로 이제 alpha를 5~7% 실제로 움직인다. 이 1단계의 목적함수(순수
layer-wise MSE reconstruction)는 2단계(margin_loss/s_mult)가 보호하는 영역
(`train_cols`=S∪H_cal 60개 + `neighbor_cols`)과 완전히 무관하다 — 그 바깥
(COCO 80개 전체 기준 Top1_flip/lost, 특히 LVIS 1203개 class 중 대부분)으로
1단계가 만드는 손상이 2단계 보호 없이 그대로 샐 수 있다.

**중간 시도 하나 기각**: 처음엔 "Combined의 1단계를 BRECQ의 block-wise
재구성으로 바꾸자"고 제안했으나, 사용자가 "그러면 reconstruction으로
decision error를 줄이자는 거 아니냐"고 지적 — 논문의 핵심 주장
("reconstruction ≠ decision preservation")과 정면으로 충돌하는 프레이밍이라
철회. 대신 baseline 메커니즘을 빌리지 않는 방향(1단계 자체를 줄이거나
없애기)으로 방향 전환.

**검증(`--combined-recon-iters`, `pipeline/run_comparison.py`)**: Combined의
1단계 iters만 `--recon-iters-ada`와 별개로 조정할 수 있게 파라미터 추가.
`--conditions combined`로 baseline 재빌드 없이 Combined만 테스트(baseline은
claim13에서 이미 6-seed 확보돼 있어 재사용). seed0 1-seed로 iters=0(1단계
완전 생략, alpha가 초기값=round-to-nearest에 남음)과 iters=200을 먼저 비교:

| | 기본(1000) | iters=0 | iters=200 |
|---|---|---|---|
| COCO_AP | 35.53 | **36.07** | 35.84 |
| Heval_flip | 10.04% | **8.88%** | 11.71%(더 나쁨) |
| lost | 420 | **382** | 502(더 나쁨) |

iters=200이 양 극단보다 나쁜 건 `temp_decay`의 beta 스케줄(20→2)이 전달된
iters 자체를 기준으로 돌아서, 200을 주면 "1000짜리의 앞부분"이 아니라
"200으로 압축된 완전한 스케줄"이 되기 때문으로 추정(불안정한 중간 수렴).
그래서 "점진적으로 줄이기"가 아니라 "0이냐 아니냐"가 맞는 축이라고 판단.

**6-seed 확정** (`runs/72_combined_recon_diag/seed{0..5}_iters0.log`, seed0은
1-seed 예비실험 재사용):

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|
| Combined(1단계 유지, claim13 직후) | 35.83 | 0.2482 | 0.1692 | 10.38% | 0.782% | 0.218% | 382.0 | 5.69% | 1160.3 |
| **Combined(1단계 제거, claim14)** | **36.09** | **0.2499** | **0.1745** | **9.58%** | 0.805% | 0.245% | 372.17 | 5.51% | **1077.17** |
| QDrop(최강 baseline, 참고) | 33.24 | 0.2323 | 0.1596 | 10.50% | 0.703% | 0.245% | 345.8 | 5.50% | 1114.8 |

**COCO_AP·LVIS_AP·APr·Heval_flip·LVIS_lost 5개 지표가 트레이드오프 없이
동시에 개선**됐고, LVIS_flip은 QDrop과 사실상 동률(5.51 vs 5.50%)이 됐다.
Top1_flip·UPIR·lost 세 개만 QDrop/BRECQ에 근소하게 남아 진다(claim13
직후엔 네 개+LVIS_flip 다섯 개에서 졌었음).

**해석**: "reconstruction이 decision을 지켜준다"가 아니라 "target 없는
reconstruction이 target 밖 영역에 새 손상을 만든다"는 쪽으로 논문 주장이
오히려 더 날카로워졌다. Combined는 이제 "AdaRound 방식 weight rounding +
margin_loss"가 아니라 **"naive round-to-nearest weight + margin_loss 기반
activation scale"**로 재정의된다 — AdaRound 메커니즘을 전혀 안 쓴다.

**수정 완료**: `pipeline/run_comparison.py`에 `--combined-recon-iters`
플래그 추가(기본값 `0`), `build()`의 `combined_recon_iters` 기본값도 `0`.
`PROMPTCAL_CURRENT_MODEL_V2.md` §5.1/§8/§9 갱신 완료.

**남은 격차(Top1_flip/UPIR/lost)에 대한 다음 방향** (baseline 메커니즘을
빌리지 않는 순서로):
1. `scale_reg_weight`/`iters`(2단계) 재튜닝 — 지금 값(10.0/1500)은 1단계가
   AdaRound-refined weight였던 시절 튜닝된 것이라, 1단계가 naive rounding으로
   바뀐 지금은 재스윕이 필요할 수 있음. 아직 안 함.
2. margin_loss/neighbor 보호 범위 확장(`neighbor_k` 확대 또는 전체 80열
   약한 정규화) — 우리 방법 자체의 확장이라 포지셔닝 문제 없음. 아직 안 함.
3. 약한 weight-level 보정 재고려 — 1·2로도 안 좁혀지면 마지막 옵션. baseline
   메커니즘 차용 프레이밍 위험 있으니 신중히.

**상태**: 확정 설계로 채택 완료. 6-seed 검증 완료. 남은 격차 축소는 미착수.

---

## Claim 15 — 비교 축이 불공정했다: baseline엔 LSQ 없이 Combined와 비교 중이었다 (09-18~09-19, 사용자 발견)

**주장**: claim14까지의 확정 §8 표는 QDrop/BRECQ의 activation scale 학습
(LSQ, claim12에서 이미 원 논문에 있다고 확인한 기능)이 **꺼진 채로**
Combined와 비교한 것이다. Combined는 이제 순수 activation-scale 방법인데
baseline엔 그 손잡이가 없으니, 이건 "방법의 차이"가 아니라 "구현 축소와의
비교"고 리뷰어 반론을 못 막는다. `--control-mse`(같은 s_mult·같은
optimizer·같은 iters, margin_loss 대신 순수 MSE)가 이제 논문에서 유일하게
결정적인 ablation이다.

**부수 지적(같은 리뷰에서, 전부 검증됨)**:
1. **bit-width 하드코딩**: `wrap_convs(m.model, 8, 8)`이 `run_comparison.py`에
   박혀있고 `--w-bits`/`--a-bits` 플래그가 0개 — W8A32/W32A8 같은 메커니즘
   실험(weight vs activation 중 뭐가 손상의 주범인지)을 못 돌림.
2. **죽은 alpha 재계산**: `quant_weight()`가 `soft`/`alpha` 상태와 무관하게
   매 forward마다 weight 크기 그대로 `h_alpha(alpha)`를 재계산 — Combined
   에서는(alpha 학습 안 함) 이게 매번 상수인데 probe 5000장+LVIS 4809장
   내내 낭비되고 있었음.
3. **APr이 FP32와 "동일"하다는 서술은 부정확**: 6-seed 평균 0.1745(당시)
   vs FP32 0.1767 — "근접"이 맞는 표현.
4. `runs/72`의 `combined-recon-iters=200`이 0/1000보다 나쁘다는 결론은
   n=1(seed0 하나)이라 확정 아님.

**검증 및 수정**:
1. `runs/71`(baseline 6-seed)/`runs/72`(Combined 6-seed) 둘 다
   `learn_act_scale: False`였음을 `[args]` 로그로 직접 확인 — 지적이 맞았다.
2. `quant_weight()`에 hard(soft=False) 경로 캐싱 추가(`_hard_weight_cache`,
   soft=True로 돌아가면 무효화) — `src/quant/adaround.py`.
3. `--w-bits`/`--a-bits` CLI 플래그 추가, `wrap_convs`에 전달 —
   `pipeline/run_comparison.py`. W8A32/W32A8 스모크 테스트(calib=32,
   eval-cap=200): naive COCO_AP가 FP32 대비 W8A32는 **-0.06**, W32A8은
   **-1.01** — 손상이 activation quantization 쪽이 압도적이라는 가설을
   지지(정식 스케일 재측정은 아직 안 함).
4. **`--learn-act-scale`(단일 플래그, 기본 False)를 `--adaround-learn-act-scale`
   (기본 False, AdaRound 원 논문엔 LSQ 없음)과
   `--qdrop-brecq-learn-act-scale`(기본 **True**, QDrop/BRECQ 원 논문엔
   있음)로 분리**해서 방법별로 정확한 기본값을 줬다 — claim12를 옵트인
   실험 플래그로 방치했던 게 이번 세션의 실수(claim5-d와 같은 성격).
   `build()`의 `adaround`/`qdrop`/`brecq` 분기도 각각 맞는 플래그를 쓰도록 수정.

**6-seed 공정 비교 결과** (`runs/75_lsq_confirmed/seed{0..5}_full.log`):

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|
| QDrop+LSQ | 35.86 | 0.2475 | 0.1771 | 8.27% | 0.625% | 0.173% | 289.2 | 5.34% | 1108.3 |
| BRECQ+LSQ | 35.78 | 0.2478 | 0.1757 | **7.95%** | **0.607%** | **0.155%** | **279.2** | **5.12%** | 1049.7 |
| Combined(scale_reg_weight=10.0, 구값) | **36.09** | **0.2499** | 0.1745 | 9.58 | 0.805 | 0.245 | 372.2 | 5.51 | 1077.2 |

Combined가 이기는 건 COCO_AP·LVIS_AP **2개뿐**, 나머지 7개(APr 포함)는
BRECQ+LSQ한테 짐. `--control-mse` 재측정(iters=0 foundation 기준,
`runs/74_learn_act_scale_v2/seed0_controlmse.log`, 1-seed)도 같이 확인:
AP는 MSE-control이 이기고(손잡이 존재가 AP의 주 원인, claim12와 일관),
LVIS_flip/LVIS_lost는 margin_loss가 이김(margin_loss의 기여가 "LVIS로의
일반화"에 있어 보임).

**`scale_reg_weight` 재튜닝**: 1-seed 스윕(10.0→20.0/5.0/2.5→0.0~1.5)한
결과, **1.0이 최적점**임을 확인 — 0.0/0.5는 오히려 LVIS_flip 악화(이
정규화가 원래 막으려던 현상, 09-07 도입 근거와 일치). `iters`(1500→3500)·
`neighbor_k`(saturation 재확인, claim6과 일치)도 스윕했으나 GPU마다 결과가
갈리는 걸 발견(완전히 동일한 설정을 다른 물리 GPU에서 돌렸더니 LVIS_flip
5.26%→5.87%로 차이 남 — 기존에 문서화된 GPU 비결정성) — 이 둘의 세부
순위는 신뢰 안 하고, `scale_reg_weight=1.0`만 동일 GPU 6-seed로 확정
검증했다.

**scale_reg_weight=1.0 6-seed 확정** (`runs/78_scalereg1_confirmed/`):

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|
| BRECQ+LSQ | 35.78 | 0.2478 | 0.1757 | **7.95%** | **0.607%** | **0.155%** | **279.2** | **5.12%** | 1049.7 |
| **Combined(1.0)** | **36.55** | **0.2539** | **0.1797** | 9.52% | 0.735% | 0.173% | 339.2 | 5.30% | **959.0** |

승리 지표 2개→**4개**(COCO_AP/LVIS_AP/APr/LVIS_lost)로 증가. UPIR은 BRECQ+LSQ와
사실상 동률(0.173%=0.173%). LVIS_flip 격차 0.39pp→0.18pp.
`--scale-reg-weight` 기본값을 1.0으로 전환.

**BRECQ-stage1 진단(margin_loss 고유 기여 검증, 1-seed)**: "reconstruction
error 최소화 ≠ decision error 최소화"라는 논문 핵심 주장과 충돌할까봐
"Combined의 1단계를 BRECQ block-wise로 바꾸자"는 안을 한 번 철회한 적
있는데(claim14 절 참고), 데이터가 이 질문을 다시 소환했다 — 재제안 시
"reconstruction을 빌려서 숫자를 맞춘다"가 아니라 **"block-wise 재구성과
margin_loss 중 뭐가 decision-preservation의 실제 원인인지 분리하는 통제
실험"으로 재프레이밍**(제안 방법 자체는 불변). `--combined-stage1 brecq`
(BRECQ의 block-wise 재구성, alpha만, LSQ는 안 켬 — activation scale은
여전히 margin_loss/s_mult 전담) 추가, `pipeline/run_comparison.py`.

3-way 비교(seed0, `runs/76_brecq_margin_diag/`):

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|
| BRECQ+LSQ(순수) | 35.85 | **0.2490** | **0.1728** | **7.28%** | **0.62%** | **0.19%** | **287** | **5.01%** | **1068** |
| BRECQ-stage1+margin_loss | 35.96 | 0.2462 | 0.1685 | 7.82% | 0.63% | 0.22% | 311 | 5.24% | 1111 |
| naive+margin_loss(당시 확정) | **36.07** | 0.2495 | 0.1765 | 8.88% | 0.79% | 0.34% | 382 | 5.59% | 1098 |

1-seed에서 두 가지가 동시에 보였음: (1) BRECQ 기반 자체는 확실히 도움
(naive 기반보다 5개 지표 우세). (2) **같은 BRECQ 기반 위에서 margin_loss가
BRECQ 자신의 LSQ보다 COCO_AP 빼고 전부 못함** — margin_loss 고유의 순수
기여가 지금 데이터로는 거의 없거나 마이너스로 보인다. 노벨티 우려(사용자
지적: "BRECQ 위에 올리면 AdaRound 위에 올리는 것과 novelty 상 다를 게
없지 않냐")로 헤드라인 제안 방법은 바꾸지 않고, 이 실험은 순수 discussion용
진단으로 스코프를 한정했다.

**6-seed 확정** (`runs/79_brecq_stage1_confirmed/seed{0..5}_full.log`,
`--combined-stage1 brecq --combined-recon-iters 2000`):

| method | COCO_AP | LVIS_AP | APr | Heval_flip | Top1_flip | UPIR | lost | LVIS_flip | LVIS_lost |
|---|---|---|---|---|---|---|---|---|---|
| BRECQ+LSQ(순수) | 35.78 | 0.2478 | 0.1757 | **7.95%** | **0.607%** | **0.155%** | **279.2** | **5.12%** | 1049.7 |
| BRECQ-stage1+margin_loss | 36.46 | 0.2531 | 0.1781 | 9.67% | 0.703% | 0.185% | 339.5 | 5.28% | **984.5** |
| naive+margin_loss(현재 확정) | 36.55 | 0.2539 | 0.1797 | 9.52% | 0.735% | 0.173% | 339.2 | 5.30% | 959.0 |

**(2)는 6-seed로 재확인됨**: BRECQ-stage1+margin_loss는 BRECQ+LSQ 대비
COCO_AP·LVIS_AP·APr·LVIS_lost 4개만 이기고 나머지 5개는 진다 — naive+
margin_loss와 정확히 같은 4승 5패 패턴. **(1)은 1-seed 때와 결론이
바뀌었다** — 6-seed로 보니 BRECQ-stage1과 naive-stage1(현재 확정)의 최종
성능이 사실상 동일하다(위 표에서 두 행이 거의 모든 지표에서 거의 같음).
즉 "BRECQ 기반이 naive보다 낫다"는 1-seed 결론은 노이즈였고, **weight-side
foundation은 결과에 실질적 영향이 없다** — 남은 격차의 원인은 순전히
margin_loss 자체다. §9 "남은 방향" (4)번(약한 weight-level 보정)은 이걸로
막다른 길임이 확인돼 더 이상 유망하지 않음.

**수정 완료**: `src/quant/adaround.py`(`_hard_weight_cache`),
`pipeline/run_comparison.py`(`--w-bits`/`--a-bits`,
`--adaround-learn-act-scale`/`--qdrop-brecq-learn-act-scale` 분리,
`--combined-stage1`, `--scale-reg-weight` 기본값 1.0). `pipeline/quant/`
사본 동기화 완료.

**09-19 (이어서) 코드 리뷰로 발견된 4개 버그, 전부 수정 완료(사용자 발견)**:
1. `--combined-stage1 adaround`인데 `--combined-recon-iters`가 기본값(0)이면
   1단계가 조용히 생략돼 `none`과 동일한 결과가 나왔다(경고 없음) — 진단
   실험을 돌리고 "adaround stage1이 none과 차이 없다"는 잘못된 결론을 내기
   딱 좋은 형태였다. `combined_stage1 != "none"`인데 `combined_recon_iters
   <= 0`이면 `AssertionError`로 명시적으로 막도록 수정.
2. `combined_stage1="adaround"`는 `combined_recon_iters`, `"brecq"`는
   `recon_iters_strong`을 써서 서로 다른 노브였다 — "1단계를 뭘로 할까"를
   비교하려는 플래그 목적과 달리 예산까지 같이 바뀌어 단일 변수 비교가
   안 됐다. 둘 다 `combined_recon_iters` 하나로 통일(기존
   `runs/76_brecq_margin_diag`는 `recon_iters_strong` 기본값 2000을
   암묵적으로 썼던 것이라, 재현하려면 이제 `--combined-recon-iters 2000`을
   명시해야 함).
3. `quantized_weight_mib`가 `total_elements / (1024**2)`로 원소당 1바이트
   (=8bit)를 암묵 가정 — `--w-bits`를 CLI로 조정 가능하게 만든 뒤로는 실제
   버그(`--w-bits 4`면 실제 크기는 절반인데 8bit 기준으로 찍힘). 각 conv의
   `self.w_bits`로 직접 계산하도록 수정 — `--w-bits 4`로 검증하면
   "이론적 모델 크기: 7.17 MiB"(정확한 기댓값)가 나옴.
4. `_hard_weight_cache`(claim15 앞부분의 alpha 캐싱 수정)가 buffer가
   아닌 일반 속성이라 `.to()`/`.half()`가 자동으로 안 따라감 —
   `switch_vocab()`이 `model.to("cpu")` 후 다시 `.to(device)`를 하는데 그
   사이 forward가 들어가면 device mismatch로 터질 수 있는 잠재 버그(현재
   코드 경로에서는 아직 안 터짐, `set_classes`가 텍스트 인코더만 건드리고
   `model.val()`도 half를 안 써서). `AdaRoundQuantConv2d._apply()`를
   오버라이드해서 캐시도 나머지 파라미터/버퍼와 같은 device/dtype을
   유지하도록 수정.

4개 전부 `--conditions combined --combined-stage1 adaround
--combined-recon-iters 50 --w-bits 4`로 재현 스모크 테스트, 크래시 없음·
모델 크기 기댓값 일치 확인.

**상태**: 코드 수정 완료(위 4개 버그 포함), `scale_reg_weight` 재튜닝과
BRECQ-stage1 진단 둘 다 6-seed 확정 완료. **남은 5개 지표(Heval_flip/
Top1_flip/UPIR/lost/LVIS_flip) 격차는 미해결** — 원인이 weight-side
foundation이 아니라 margin_loss 자체임이 6-seed로 확정됐다. 다음 방향은
`PROMPTCAL_CURRENT_MODEL_V2.md` §9 "남은 방향" 참고(neighbor_k 확장,
margin_loss 설계 재검토 — weight-level 보정은 막다른 길로 확인돼 제외).

---

## Claim 16 — 남은 격차를 좁히려는 두 시도, 둘 다 무산 (09-20, 사용자 지시로 야간 자동 진행)

**배경**: claim15가 남긴 "남은 방향" 목록 중 (2) neighbor_k 확대/보호 범위
확장과 (3) margin_loss 설계 재검토를 사용자가 자면서 자동으로 시도해보라고
지시. "괜찮으면 6-seed 검증→문서화→커밋까지 알아서 하고, 치명적인 문제가
있을 때만 멈추라"는 조건이었음. 결과: 둘 다 부정적(하나는 구조적 막다른
길, 하나는 1-seed 노이즈였음) — 확정 설계 변경 없음, 코드와 발견 자체만
기록.

**시도 1 — `neighbor_of_cal`(H_cal의 이웃도 보호 범위에 추가)**:
`optimize_promptcal_scale_neighbor`에 파라미터 추가 — S의 이웃을 계산하던
루프와 똑같이 H_cal의 이웃도 계산해서 `neighbor_set`에 합침, **같은
`exclude_set`(H_eval 포함)을 재사용**하므로 held-out 불변식은 코드
구조상 절대 안 깨짐. `pipeline/run_comparison.py --neighbor-of-cal` 추가.

**검증(1-seed, seed0)**: 결과가 `neighbor_of_cal` 끈 기존 확정값과
**완전히 동일한 숫자**로 나옴. 원인: COCO-80이 정확히 S(40)+H_cal(20)+
H_eval(20)이라, S의 이웃 후보 풀(S∪H_eval 제외)은 처음부터 H_cal 20개
전부와 일치한다(claim6에서 "포화된다"고만 알고 있었는데, 이번에 "정확히
같다"는 걸 확인). H_cal 자신의 이웃도 같은 후보 풀(exclude_set=S∪H_eval
제외하면 H_cal만 남음) 안에서만 돌기 때문에 `neighbor_cols`에 새로 추가될
원소가 하나도 없다. **구조적으로 무의미함 — 코드 버그가 아니라 COCO-80
80-class 폐쇄 구조 자체의 한계**. 6-seed 검증 불필요(1-seed로 이미
수학적으로 확정).

**시도 2 — `aux_mse_weight`(margin_loss에 dense 보조 신호 추가)**:
BRECQ-stage1 진단(claim15)에서 margin_loss(sparse top-k)가 BRECQ의 dense
reconstruction objective보다 decision-preservation에 못한 게 확인돼서,
margin_loss를 **대체**(`--control-mse`처럼)가 아니라 **보강**하는 방향으로
`F.mse_loss(sim_q[aidx][:,train_cols], sim_fp[aidx][:,train_cols])`를
margin_loss에 가중치를 줘서 더하는 항 추가(`train_cols`=S∪H_cal, H_eval
무관). `pipeline/run_comparison.py --aux-mse-weight` 추가.

**1-seed 스윕(seed0, `runs/80_neighbor_aux_sweep/`)**: 0.05/0.1/0.15/0.2/
0.25/0.3/0.5 스윕. **0.2가 9개 지표 중 6개(LVIS_AP/Top1_flip/UPIR/lost/
LVIS_flip/LVIS_lost)에서 기존 확정값보다 나은 것으로 보였음** — 유망해서
6-seed 확정으로 진행.

**6-seed 확정(`runs/81_auxmse02_confirmed/seed{0..5}_full.log`) — 반박됨**:

| 지표 | 확정값(aux=0, 6-seed) | aux=0.2(6-seed) |
|---|---|---|
| COCO_AP | 36.55 | 36.56(무의미한 차이) |
| LVIS_AP | 0.2539 | 0.2545(무의미한 차이) |
| APr | **0.1797** | 0.1771(악화) |
| UPIR | **0.173%** | 0.182%(악화) |
| lost | **339.2** | 348.3(악화) |
| LVIS_flip | **5.30%** | 5.39%(악화) |
| LVIS_lost | **959.0** | 985.5(악화) |

1-seed 스윕에서 좋아 보였던 6개 지표 중 실제로 6-seed까지 유지된 건 없다
— 전형적인 1-seed 노이즈 함정(이 세션에서 `combined_recon_iters=200`,
scale_reg_weight/iters/neighbor_k 스윕의 GPU 비결정성 등 반복적으로
나왔던 패턴과 같은 성격). **확정 설계는 변경 없음**
(`aux_mse_weight` 기본값 0.0 유지).

**수정 완료(코드는 유지, 진단용)**: `src/quant/promptcal.py`
(`optimize_promptcal_scale_neighbor`에 `neighbor_of_cal`/`aux_mse_weight`
파라미터 추가), `pipeline/run_comparison.py`(`--neighbor-of-cal`/
`--aux-mse-weight` CLI 플래그, 결과 표 태그). `pipeline/quant/promptcal.py`
동기화 완료.

**상태**: 두 방향 다 6-seed(또는 수학적으로) 확정된 부정적 결과.
`PROMPTCAL_CURRENT_MODEL_V2.md` §9 "남은 방향" (2)(3)(4) 전부 소진 —
top-k margin 자체를 근본적으로 다른 objective 형태로 바꾸는 것 외에는
뚜렷한 다음 수가 안 보임. 사람의 판단이 필요한 지점이라 margin_loss
재설계는 여기서 자동 진행을 멈추고(치명적 문제는 아니지만, 사전에 합의된
다음 방향이 없어서), 설계 판단이 필요 없는 안전한 항목(claim17)으로
대신 이어감.

---

## Claim 17 — bit-width 메커니즘 정식 확정: 손상은 activation 양자화가 거의 전부 (09-20, 야간 자동 진행)

**배경**: claim15에서 `--w-bits`/`--a-bits`를 추가하고 스모크 스케일
(calib=32)로만 확인했던 "손상이 weight보다 activation 쪽"이라는 가설을
정식 스케일(calib=256, `--eval-cap` 없음)로 확정.

**실측(naive, seed 무관 — calibration이 결정적이라 1회면 확정)**:

| 설정 | COCO_AP | FP32 대비 | LVIS_AP | FP32 대비 |
|---|---|---|---|---|
| FP32 | 36.80 | - | 0.2589 | - |
| W8A32(weight만 8bit) | 36.75 | **-0.05** | 0.2588 | **-0.0001** |
| W32A8(activation만 8bit) | 33.51 | **-3.29** | 0.2324 | **-0.0265** |
| W8A8(표준) | 33.54 | -3.26 | 0.2342 | -0.0247 |

**weight quantization은 거의 무손실**(-0.05 COCO_AP, LVIS_AP는 사실상
차이 없음)**이고, activation quantization이 전체 W8A8 손상의 사실상
전부를 차지한다**(-3.29 vs 전체 -3.26 -- activation 단독이 전체보다도
살짝 더 나쁘게 나온 것도 노이즈 수준). 스모크 스케일(calib=32) 신호가
정식 스케일에서 그대로, 더 깨끗하게 재현됨.

**의미**: naive weight-rounding(단순 round-to-nearest)은 이미 정보 손실이
거의 없는 수준이라, AdaRound 같은 weight-rounding 최적화가 (claim13/14에서
봤듯) 별 도움이 안 되는 이유를 설명해준다 -- 애초에 고칠 게 별로 없는
곳을 고치려 한 셈. 반대로 Combined가 activation scale(s_mult)만 조정해서
큰 개선을 낸 것도 이 메커니즘과 정확히 들어맞는다 -- 손상의 근원지를
직접 건드리고 있었던 것.

**수정 완료**: 정식 스케일 측정만 수행, 코드 변경 없음(claim15에서 이미
구현된 `--w-bits`/`--a-bits` 사용).

**상태**: 확정 완료(`runs/82_bitwidth_confirmed/`). 다른 PTQ 방법
(AdaRound/QDrop/BRECQ/Combined)에도 W8A32/W32A8을 적용해서 "각 방법이
weight 여유를 얼마나 쓰는지" 볼 수도 있지만, naive 결과만으로 메커니즘
논증에는 충분하다고 판단 — 추가 측정은 필요시 사람 판단으로 진행.

---

## 부록 A — 세션 중 발견한 실행/GPU 이슈

**`CUDA_VISIBLE_DEVICES=N`만으로는 물리 GPU N이 보장 안 됨.**
`CUDA_DEVICE_ORDER=PCI_BUS_ID`를 안 붙이면 CUDA가 nvidia-smi와 다른 순서로
GPU를 enumerate할 수 있다 — 실제로 `CUDA_VISIBLE_DEVICES=0`으로 띄운
프로세스가 물리 GPU1에서 돈 사례가 있었다(claim5 6-seed 실행 중 발견,
다행히 그 시점 GPU1도 비어있어 충돌은 없었음). **앞으로 GPU를 지정할 때는
항상 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N`을 같이 쓸 것.**
`pipeline/run_comparison.py` docstring의 예시 커맨드에는 원래부터 이게
붙어있었다.

**GPU 공유 서버 에티켓**: 모든 실행 전에 `nvidia-smi --query-compute-apps`로
실제 프로세스 소유권을 확인(메모리 수치만으로 판단 안 함). 이번 세션 중
seunghyuk의 vLLM 작업이 GPU1~3을 95~100%로 쓰고 있던 시점이 있었고, 그
GPU들은 비었다고 뜰 때까지 피했다.

**병렬 실행 시 메모리 모니터링**: 6-seed를 한 번에 다 띄우지 않고, 배치
단위(3개 → 상태 확인 → 나머지)로 띄우면서 `nvidia-smi`/`free -h`를 매번
확인하는 방식으로 진행(claim5 재검증부터 적용).

**GPU 모델 이질성(09-16, 풀스케일 최종 검증 때 발견)**: "논문에 실릴 최종
확정 숫자"를 만드는 6-seed 실행에서는 매핑 정확성뿐 아니라 **GPU 모델
자체를 6개 seed 전부 동일하게 맞추는 게 낫다** — `PROMPTCAL_CURRENT_MODEL.md`
§9의 GPU-비결정성 실측(동일 코드/seed도 물리 GPU가 다르면 결과가 달라짐)
때문. 이 서버는 20GB 카드(물리 0/4/5/6/7, RTX4000 Ada) 5개와 46GB 카드
(물리 1/2/3) 3개가 섞여 있어서 6-seed를 돌리려면 최소 하나는 다른 모델을
써야 하는 상황이 생김 — 해결책은 (a) 매칭되는 GPU가 남을 때까지 순차
대기, 또는 (b) 메모리 여유가 큰(job당 ~2.5GB, 카드당 20GB) 걸 이용해
같은 모델 GPU 하나에 2개 job을 동시에 올려서 공유(연산만 나눠 쓰고
메모리는 안 모자람 — 그 GPU의 두 job만 ~2배 느려질 뿐 OOM 위험은
없음). 이번엔 (b)를 씀(GPU7에 seed4/5 공유). 스모크성 빠른 검증(seed 간
비교가 주 목적이 아닌 경우)에서는 모델 이질성이 크게 중요치 않을 수
있지만, "확정 표"를 만들 때는 반드시 신경 쓸 것.

## 부록 B — 코드 변경 파일 목록 (전부 uncommitted)

| 파일 | 변경 내용 |
|---|---|
| `src/quant/adaround.py` | `AdaRoundQuantConv2d.__init__`/`convert_to_adaround`에 `channelwise_smult` 파라미터 추가(claim4) |
| `src/quant/promptcal.py` | confident-anchor 선정 `train_cols` 제한(claim1), `margin_loss`에 `identity_aware` 파라미터 추가(claim5) + intrusion 탐지 보강(claim5-b), `optimize_promptcal_scale_neighbor`에 `identity_aware_margin` 파라미터 전달, `ml_cal` 게이트에 `cal_weight > 0` 추가(claim5-e, 문서/코드 불일치 수정 + 낭비 연산 제거) |
| `pipeline/quant/adaround.py`, `pipeline/quant/promptcal.py` | 위 두 파일과 동기화(diff 없음 확인) |
| `scripts/60_hparam_sweep_official_data.py` | `--smult-per-tensor`/`--identity-aware-margin`(기본 True) 플래그 추가, `build_adaround_base`/`build_combined_from_base`에 전달(claim5-d), `cal_idx`를 스윕 대상과 무관하게 항상 전달하도록 수정(claim5-d) |
| `pipeline/run_comparison.py` | `--smult-per-tensor`, `--identity-aware-margin` CLI 플래그 추가, `build()`에 `channelwise_smult`/`identity_aware_margin` 파라미터 전달, `measure_ap`의 클래스별 AP 매핑 수정(claim7), `switch_vocab`의 names/predictor 갱신 추가(claim8), `--eval-cap` help 문구·결과 표 안내 문구 추가(claim10c), `cal_idx` 항상 전달로 confound 제거(claim5-a), `--conditions` 플래그 + `print(vars(args))` 추가(claim5-c) |
| `requirements.txt` | `ultralytics>=8.3.0` → `ultralytics==8.4.121`로 고정(claim9) |
| `scripts/58_full_baseline_official_data.py` | claim7/8과 동일한 두 수정 동기화 |
| `src/quant/promptcal.py` | `optimize_promptcal_scale_neighbor`에 `control_mse` 파라미터 추가(claim12, margin_loss 대신 순수 MSE로 s_mult만 최적화하는 control 실험) |
| `src/quant/adaround.py` | `learn_act_scale`/`act_lr`/`warmup` 파라미터 추가(claim12), `quant_act()`로 activation 양자화 경로 통합, `lp_rec_loss`/`temp_decay`/`DEFAULT_LR`/`DEFAULT_REG_WEIGHT`/`DEFAULT_WARMUP`/`DEFAULT_ACT_LR` 추가 및 `optimize_adaround` 전체 재작성(claim13, 공식 BRECQ 재구성 loss 정규화·lr·reg_weight·warmup 정합화), `flip_rate()` 진단 추가 |
| `src/quant/brecq.py` | `optimize_brecq`에 `learn_act_scale`/`qdrop_prob`/`grad_clip` 파라미터 추가(claim12/13), `_to_cpu`/`_to_dev`/`_mix` 재귀 헬퍼로 다중 입력 list/tuple 처리 및 QDrop block 입력 drop 구현, 재구성 loss를 `lp_rec_loss`로 교체(claim13), QDrop을 여기로 이전 |
| `pipeline/run_comparison.py` | `--control-mse`, `--learn-act-scale` CLI 플래그 추가, `build()`에 `control_mse`/`learn_act_scale` 파라미터 전달(claim12), QDrop 조건이 `optimize_brecq(qdrop_prob=...)`를 쓰도록 변경(claim13), `--combined-recon-iters`(기본 0) 플래그 추가 — Combined의 1단계(`optimize_adaround`) iters를 `--recon-iters-ada`와 분리, 0이면 1단계 생략(claim14, 새 확정 기본값) |
| `pipeline/quant/adaround.py`, `pipeline/quant/brecq.py`, `pipeline/quant/promptcal.py` | 위 세 파일과 동기화(diff 없음 확인) |
| `src/quant/adaround.py` | `quant_weight()`에 `_hard_weight_cache` 추가 — soft=False(고정)일 때 매 forward 재계산 낭비 제거, soft=True로 돌아가면 캐시 무효화(claim15) |
| `pipeline/run_comparison.py` | `--w-bits`/`--a-bits` 플래그 추가(claim15, 메커니즘 실험용), `--learn-act-scale` 단일 플래그를 `--adaround-learn-act-scale`(기본 False)와 `--qdrop-brecq-learn-act-scale`(기본 True)로 분리해 방법별 원 논문 기본값 확정(claim15, claim12를 옵트인으로 방치했던 실수 수정), `--combined-stage1`(none/adaround/brecq) 추가 — BRECQ block-wise를 Combined 1단계로 쓰는 진단 실험용(claim15), `--scale-reg-weight` 기본값 10.0→1.0(claim15) |
| `pipeline/quant/adaround.py` | 위 `src/quant/adaround.py`와 동기화(diff 없음 확인) |

## 부록 C — 실행 로그 위치

| 실험 | 위치 |
|---|---|
| claim1 전/후 1-seed 비교 | `runs/62_anchorfix/seed0_{before,after}.log` |
| claim4 per-tensor vs per-channel(seed0) | `runs/62_anchorfix/seed0_after.log`(per-channel) + `runs/63_smult_pertensor/seed0_pertensor.log` |
| claim4 per-tensor 6-seed(vs baseline) | `runs/63_smult_pertensor/seed{0..5}_pertensor.log` |
| claim5 identity-aware ON 6-seed | `runs/64_identity_margin/seed{0..5}_identity.log` |
| claim5 identity-aware OFF 6-seed(seed0은 `62_anchorfix` 재사용) | `runs/65_identity_off/seed{1..5}_off.log` |
| claim3 rounding_diff_rate 측정 | 스크립트만 존재(`/tmp` scratchpad, 저장소에 커밋 안 됨) — 재현하려면 이 문서의 claim3 절차대로 재실행 필요 |
| claim11 LVIS-fix 6-seed 재측정 | `runs/68_lvis_fix_fullscale/seed{0..5}_lvisfix.log` |
| claim12 control-mse 3-seed | `runs/69_control_mse/seed{0..2}_controlmse.log` |
| claim12 learn-act-scale 1-seed(구 재구성 코드) | `runs/70_learn_act_scale/seed0_full.log` |
| claim13 재구성 수정 6-seed 재측정(완료) | `runs/71_recon_fix_review/seed{0..5}_full.log` |
| claim14 combined-recon-iters 예비(iters=0/200 비교, seed0) | `runs/72_combined_recon_diag/seed0_iters{0,200}.log` |
| claim14 1단계 제거 6-seed 확정(완료) | `runs/72_combined_recon_diag/seed{0..5}_iters0.log` |
| claim15 learn-act-scale 재측정(현재 코드, seed0) + control-mse 재측정 | `runs/74_learn_act_scale_v2/seed0_{full,controlmse}.log` |
| claim15 공정 비교(QDrop+LSQ/BRECQ+LSQ 기본 적용) 6-seed 확정 | `runs/75_lsq_confirmed/seed{0..5}_full.log` |
| claim15 BRECQ-stage1+margin_loss 진단 예비(seed0) | `runs/76_brecq_margin_diag/seed0_full.log` |
| claim15 BRECQ-stage1+margin_loss 6-seed 확정(완료) | `runs/79_brecq_stage1_confirmed/seed{0..5}_full.log` |
| claim15 scale_reg_weight 스윕(0.0~20.0, seed0) | `runs/77_hparam_sweep/scalereg_*.log` |
| claim15 iters/neighbor_k 스윕(seed0, GPU 비결정성 확인됨 — 세부 순위 신뢰 안 함) | `runs/77_hparam_sweep/{iters,neighbork}_*.log` |
| claim15 scale_reg_weight=1.0 6-seed 확정(완료) | `runs/78_scalereg1_confirmed/seed{0..5}_full.log` |
| claim15 bit-width(W8A32/W32A8) 스모크 테스트(정식 재측정은 claim17 참고) | `runs/73_review_fixes/smoke_w{8a32,32a8}.log` |
| claim16 neighbor_of_cal 확인(1-seed, 구조적으로 무의미함 확정) + aux_mse_weight 1-seed 스윕 | `runs/80_neighbor_aux_sweep/` |
| claim16 aux_mse_weight=0.2 6-seed 확정(반박됨, 확정 설계 변경 없음) | `runs/81_auxmse02_confirmed/seed{0..5}_full.log` |
| claim17 bit-width 정식 확정(W8A32/W32A8, naive, seed 무관) | `runs/82_bitwidth_confirmed/w{8a32,32a8}.log` |

## 부록 D — 결정 현황 (2026-09-16 갱신)

> 이 부록은 09-16 시점 스냅샷으로 고정, 이후 갱신 안 함. claim11~15의
> 최신 결정 현황은 각 claim 절 자체와 이 문서 맨 위 배너, 그리고
> `PROMPTCAL_CURRENT_MODEL_V2.md`를 볼 것.

**09-16에 내려진 결정** (더 이상 "고민 중"이 아니라 확정된 방향):
- **claim4(per-tensor)와 claim5(identity-aware margin, intrusion 수정판)
  둘 다 채택.** 판단 기준을 "성능이 더 좋은가"에서 "성능 손해 없이
  구조적 결함(공정성/배포 가능성/loss의 correctness)을 고치는가"로
  재정립함 — per-tensor는 baseline과의 공정한 비교·실제 배포 가능성을
  위해 필연적, identity-aware margin은 margin_loss가 실제로 flip을
  보게 만드는 correctness fix이고 6-seed 기준 성능 손해가 사실상
  없음(§Claim5 표 참고). 둘 다 성능 저울질 대상이 아니라 "이렇게
  해야 하는" 방향으로 확정.
- claim1(anchor 선정 버그 수정)은 원래도 순수 버그 수정이라 별도 결정
  없이 항상 적용 — 새 확정 설계에 당연히 포함.
- claim2/claim6 논문 문구 초안 작성 완료(해당 절 참고), 실제 논문
  파일에 삽입은 사용자 몫.

**진행 중(GPU)**: 위 결정에 따른 새 확정 설계(`--smult-per-tensor
--identity-aware-margin`, claim1 fix 포함)로 **풀스케일(`--eval-cap` 없음,
val2017 5000장 + 공식 LVIS minival 4809장) 6-seed(0~5) 재검증**을
`runs/67_final_confirmed_fullscale/`에서 돌리는 중. 5개 조건(naive/
AdaRound/QDrop/BRECQ/Combined) 전부 포함 — 최종 확정 표에 baseline도
같은 스케일로 필요하기 때문. 6개 seed 전부 물리 GPU 0/4/5/6/7(동일 모델,
RTX4000 Ada)에서만 실행해서 GPU-모델 confound 없음 — 처음엔 GPU 부족으로
seed5를 물리 GPU2(46GB 카드, 다른 모델)에 띄웠다가, §9의 GPU-비결정성
우려 때문에 초반(2분 경과, 손실 미미)에 죽이고 GPU7(RTX4000 Ada, seed4와
공유)로 재시작함(부록 A 참고). seed4/5는 GPU7을 공유해서 그 둘만 좀 더
오래 걸림.

**완료된 작업** (09-16):
1. ~~`runs/67_final_confirmed_fullscale/`의 6-seed 결과를 §8.1 형식으로
   집계.~~ **완료.**
2. ~~`PROMPTCAL_CURRENT_MODEL.md` §8.1을 이 결과로 교체~~ **완료**(커밋
   `4629adf`) — 이전 per-channel 확정값은 새 §8.11로 보존. 풀스케일로
   재니 COCO_AP -0.40(6-seed 범위가 서로 안 겹침, eval-cap=1000에서 봤던
   노이즈 수준(-0.06)보다 큼)이 확인됐지만, decision-preservation 지표는
   전부 유지/개선(LVIS_lost -66)됨.

**남은 작업**:
1. claim2/claim6 문구를 실제 논문 파일에 삽입(이 저장소 밖 작업).
2. ~~`--smult-per-tensor`/`--identity-aware-margin` 기본값을 뒤집을지
   검토~~ **완료**(커밋 `4332e46`) — `argparse.BooleanOptionalAction`으로
   바꿔서 기본값 True, `--no-*`로 이전 설계 재현 가능. `build()` 기본값도
   맞춰서 변경. 라이브러리 레벨(`adaround.py`/`promptcal.py`) 기본값은
   의도적으로 안 건드림(`run_comparison.py`가 항상 명시적으로 넘기므로
   실제 동작엔 무관, `scripts/58` 등 과거 암묵적 기본값 의존 코드의
   재현성만 흔들 위험 있어서).
3. 이 문서와 코드 변경 전체는 이미 커밋됨(`87157e5`, `645f910`, `d5ce828`,
   `4629adf`, `345e71e`, `4332e46` + README 갱신 3건) — 새 변경이 생기면
   그때그때 커밋.

---

## Claim 18 — "자유도를 늘려 격차를 좁힌다"와 "범위 축이 cross-vocabulary를 가른다", 둘 다 반박됨 (09-24)

**배경**: baseline 정합성 작업(→ `PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md` §7)이
끝나고 확정된 기준선은 다음과 같다(`runs/97`, 6-seed).

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP |
|---|---|---|---|---|---|
| naive | 36.617 | 5.945% | 0.667% | 291 | 0.2554 |
| AdaRound(예산 정렬) | 36.648 | 5.142% | 0.472% | 234 | 0.2577 |
| QDrop | 36.807 | 4.090% | 0.397% | 180 | 0.2556 |
| **BRECQ** | 36.755 | **3.705%** | **0.345%** | **166** | 0.2560 |
| Combined | 36.642 | 5.602% | 0.527% | 248 | 0.2576 |

**Combined는 BRECQ에 Pareto-dominated 상태다.** seed별 짝비교에서 naive에만 전승이고
AdaRound·QDrop·BRECQ에게는 decision 지표 0/6 전패. 유일한 우위인 LVIS_AP(+0.0016)도
예산을 정렬한 AdaRound(0.2577)가 사실상 따라잡았다. **넘어야 할 선은 BRECQ의
`Heval_flip 3.705% / lost 166`.**

### 18-a. alpha를 margin 목적함수 아래에서 여는 시도 — 실패

**가설**: Combined는 s_mult(conv당 스칼라 **52개**)뿐인데 BRECQ/QDrop은 alpha 수백만 개 +
LSQ 52개다. claim14가 실패한 건 1단계 AdaRound가 **MSE 목적함수**로 alpha를 움직여
보호 범위 밖으로 손상이 샜기 때문이니, **같은 alpha를 margin_loss 아래에서 직접 푸는
것은 별개의 시도**일 것이다.

**구현**: `--combined-learn-alpha`(기본 꺼짐). `optimize_promptcal_scale_neighbor`에서
`ac.alpha.requires_grad_(True)` + `soft=True`, alpha 전용 Adam, AdaRound와 같은 rounding
정규화(`temp_decay`, warmup 0.2), 종료 시 hard 확정.

**하이퍼파라미터에서 두 번 틀렸다(기록용)**:
1. 정규화를 `reduction="mean"`으로 뒀다 — margin_loss가 mean 스케일이니 손실 **값**을
   맞춰야 한다고 생각했는데, mean은 **원소당 gradient를 1/N(수백만분의 1)로 나눈다.**
   `alpha_reg_weight=5.0`까지 올려도 h 수렴률이 10%(=초기 균일분포 그대로)에서 꿈쩍
   안 했다. 맞춰야 하는 건 손실 값이 아니라 **alpha 원소당 gradient 크기** →
   공식대로 `sum`으로 수정(10%→23%).
2. 그래도 부족했던 건 **예산**이었다. h를 0/1로 밀려면 alpha가 ~2.0 움직여야 하는데
   `lr=1e-3 × 240 step ≈ 0.24`뿐이다. `alpha_lr`을 1e-2로 올리니 93~99% 수렴.

**스윕(iters=1500, calib=64)**: `lr=1e-2, reg=1e-2`에서 h 99% 수렴, 배포되는 hard 모델의
margin 목적함수가 기준 0.0896 → **0.0589로 34% 개선**. 여기까지는 전부 긍정적이었다.

**실제 지표(`runs/99` vs `runs/100`, seed 0·1, `--deterministic`, `naive,brecq,combined`
고정)**: `naive`/`brecq`가 두 run에서 bit-identical해 격리는 완벽. 그런데
**10개 지표 전부, 2개 seed 전부 악화**.

| 지표 | alpha 동결 | learn_alpha | |
|---|---|---|---|
| COCO_AP | 36.665 | 36.460 | −0.21 |
| S_AP(보호 대상) | 37.345 | 37.135 | −0.21 |
| Heval_flip | 5.345% | **8.040%** | +50% |
| Top1_flip | 0.505% | **0.970%** | +92% |
| lost | 250 | **370** | +48% |
| LVIS_AP | 0.2579 | 0.2550 | −0.0029 |
| LVIS_flip | 2.900% | **4.710%** | +62% |
| LVIS_lost | 672 | **912** | +36% |

alpha는 제대로 움직였다(h 수렴 98%, nearest 대비 flip 5.0% — AdaRound의 7.6%와 같은
수준). **calibration 목적함수는 34% 개선됐는데 배포 지표는 전부 나빠졌다 = 전형적 과적합.**

**교훈(가설의 전제가 틀렸다)**: 문제는 *어떤 목적함수냐*가 아니라 **감독 신호의 밀도
대비 자유도**다.

| | 파라미터 | 감독 신호 |
|---|---|---|
| BRECQ | alpha ~수백만 | **dense** — 모든 블록의 모든 출력 원소 |
| Combined(기존) | s_mult **52개** | **sparse** — confident anchor의 top-(k+1) 경계, 40~60 컬럼 |
| Combined+learn_alpha | alpha ~수백만 | **sparse (그대로)** |

BRECQ가 수백만 파라미터를 감당하는 건 감독이 촘촘하기 때문이다. margin_loss는 설계상
sparse한데(claim15가 이미 지적) 거기 수백만 손잡이를 붙이면 calibration anchor를
외운다. **s_mult가 52개뿐인 게 약점이 아니라 정규화 장치였다.** 손상이 일반화가 필요한
곳일수록 큰 것(LVIS_flip +62% > lost +48% > COCO_AP −0.21)과, **보호 대상인 S_AP조차
떨어진 것**(margin 목적함수가 자기 타깃 지표에도 좋은 프록시가 아님)이 이 해석을 뒷받침한다.

### 18-b. activation 범위(클리핑) 축을 여는 시도 — 실패

**가설**: fidelity 문서 §7.2에서 AdaRound의 activation observer를 min-max(클리핑 없음)
↔ MSE(클리핑)로 바꾸면 **COCO와 LVIS가 6/6 seed 양방향으로 갈린다**는 걸 확인했다.
논문 §Reconstruction–Utility Misalignment("reconstruction-optimal scale ≠ semantic/task-
optimal scale")의 직접적 실례이고, 논문 §Optimization이 "scale/**clipping**/rounding"을
최적화 대상으로 명시하므로, **cross-vocabulary 일반화가 범위 축에 산다**고 보았다.

**구현**: `ActObserver`가 method와 무관하게 클리핑 없는 실제 min/max를 항상 같이 관측하고,
`freeze(range_blend)`에서 MSE 최적 범위와 선형 보간. `--combined-range-blend`(기본 0.0 =
기존 동작, combined 모드에만 적용). 난수를 안 쓰므로 blend=0에서 bit-identical.

**결과(`runs/101`, blend ∈ {0, 0.5, 1.0} × seed {0,1})**: **10개 중 7개가 2/2 seed 악화,
나머지는 무변화. 기대했던 트레이드오프가 아예 없다.**

| 지표 | b0.0 | b0.5 | b1.0 | |
|---|---|---|---|---|
| COCO_AP | 36.665 | 36.605 | 36.590 | 악화 2/2 |
| Heval_flip | 5.345 | 5.500 | **6.015** | 악화 2/2 |
| lost | 250 | 239 | **277** | 악화 2/2 |
| **LVIS_flip** | 2.900 | 3.030 | **3.485** | **악화 2/2** |
| **LVIS_lost** | 672 | 658 | **750** | **악화 2/2** |
| LVIS_AP | 0.2579 | 0.2572 | 0.2577 | 무변화 |
| LVIS_APr | 0.1788 | 0.1760 | 0.1764 | 악화 2/2 |

**교훈: §7.2의 방향이 Combined에서는 뒤집힌다.**

| | LVIS_flip | LVIS_lost | LVIS_AP |
|---|---|---|---|
| AdaRound + **min-max**(넓음) | **2.813** | **639** | **0.2577** |
| AdaRound + MSE(좁음) | 3.603 | 824 | 0.2542 |
| Combined + MSE(좁음) = b0.0 | **2.900** | **672** | **0.2579** |
| Combined + **min-max**(넓음) = b1.0 | 3.485 | 750 | 0.2577 |

`Combined+MSE ≈ AdaRound+min-max`, `Combined+min-max ≈ AdaRound+MSE`. **범위 자체가
원인이 아니라, 범위와 그 위에서 최적화되는 것 사이의 상호작용이었다.** AdaRound는
alpha를 그 격자에 맞춰 다시 푼다 — MSE 격자에서는 alpha와 격자가 같이 COCO에 과적합하고,
min-max 격자에서는 재구성이 넓은 격자를 감당해야 해서 덜 COCO-특화된다. Combined는
weight가 round-to-nearest 고정이라 보상 수단이 없어, 8bit에서 범위만 넓히면 해상도만
굵어진다.

**확정 설계 변경 없음.** `--combined-range-blend` 기본 0.0 유지.

### 18-c. 종합 — 네 번의 시도가 같은 벽을 가리킨다

| 시도 | 무엇을 바꿨나 | 결과 |
|---|---|---|
| claim14 | 1단계 AdaRound(MSE로 alpha) | 보호 범위 밖 손상 |
| claim16 | aux_mse(train_cols에 dense 신호 추가) | 6-seed 반박 |
| **18-a** | **margin_loss로 alpha** | **10/10 지표 악화** |
| **18-b** | **activation 범위 확장** | **7/10 악화, 이득 0** |

원인이 "목적함수 선택"이 아니라 **감독 신호의 밀도**로 특정됐다. 자유도를 늘리려면
감독을 먼저 촘촘하게 만들어야 하는데, 그 시도(claim16 aux_mse)는 이미 반박됐다.

뒤집어 보면 이게 논문의 이야기일 수 있다 — **Combined의 LVIS 우위는 자유도가 낮아서
나오는 것**이고, 그렇다면 목표는 "BRECQ를 decision 지표에서 이긴다"가 아니라
**"capacity/generalization 곡선에서 다른 지점을 차지한다"** 가 된다.

### 18-d. 부수 발견: `--deterministic`에서도 BRECQ는 완전히 결정적이지 않다

`runs/101`의 5개 run과 `runs/99` 사이에서 `naive`는 7/7 bit-identical인데 **`brecq`만
두 값 중 하나에 무작위로 떨어진다**(AP 36.74↔36.75, Heval_flip 3.72↔3.68, lost 190↔188).
`--deterministic`에서도 결정적 구현이 없어 경고만 뜨는
**`adaptive_max_pool2d_backward_cuda`**(`ImagePoolingAttn`의 `AdaptiveMaxPool2d`)가
유력하다 — BRECQ는 `ImagePoolingAttn`을 블록 타깃으로 재구성하므로 그 backward를
정면으로 통과한다.

변동폭이 측정 대상의 1/25 수준이라 위 결론에는 영향이 없지만, **"baseline은 실행 분산
0"이라는 서술은 `--deterministic`을 켠 BRECQ에 한해 정정이 필요하다**(fidelity 문서
§7.0, `pipeline/BASELINE_STATUS.md` §5.1).

---

## Claim 19 — 논문 §4.3 Utility-Constrained Refinement: 절반은 죽은 코드였고, 고쳐도 순수 이득은 없다 (09-25)

**배경**: claim18-c에서 "자유도를 늘리는 방향"이 닫혔고, 남은 건 **감독의 종류를
바꾸는 것**이었다. 논문 §4.3(Utility Constraints: prompt ranking/margin, threshold
crossing, box consistency)이 명시한 두 항 `l_thresh`/`l_box`가
`semantic_calib.utility_refinement_terms`에 **구현만 돼 있고 확정 경로에서 호출된 적이
없었다**. 파라미터를 하나도 안 늘리면서(s_mult 52개 그대로) 감독만 추가하는 방향이라
claim18-a의 과적합 패턴에 안 걸린다고 보았다.

**구현**: 구 `optimize_promptcal_scale_neighbor_utility`를 그대로 쓰지 않았다 — 확정
설계 이전 버전이라 `cal_idx`/`identity_aware_margin`/`scale_reg_weight`가 없어서 변수가
여러 개 동시에 바뀐다. 확정 함수 `optimize_promptcal_scale_neighbor`에 옵트인으로
얹어 단일 변수 비교가 되게 했다(`--combined-utility-frac`, 기본 0.0 = bit-identical).

### 19-a. 1차 측정: 개선은 있으나 트레이드오프 (`runs/102`)

`frac=0.3`(마지막 30% 구간), `thresh_w=1.0, box_w=0.5`(구 함수에서 상속한 미튜닝 값):

| | 개선 2/2 seed | 악화 2/2 seed |
|---|---|---|
| | `Heval_flip` 5.345→5.180, `lost` 250→**229**, `LVIS_lost` 672→646 | `LVIS_AP` 0.2579→0.2566, `LVIS_APr` 0.1788→**0.1758**, `H_eval_AP` 32.94→32.79 |

네 번의 시도 중 **처음으로 일관된 개선**이 나왔다. 그러나 대가가 하필 Combined의
유일한 차별점이었다 — LVIS_AP 마진이 BRECQ 대비 +0.0028→+0.0015로 반토막, LVIS_APr는
동률(0.1788 vs 0.1789)에서 열세(0.1758)로 전환. `frac=0.5`는 0.3보다 나쁨.

### 19-b. 두 항 분해에서 버그 발견 (`runs/103`)

`l_thresh`는 sparse/AP정렬, `l_box`는 dense MSE(=claim16 `aux_mse`와 같은 패턴)라
성격이 정반대여서 분해했다. 결과가 결정적이었다:

| 지표 | base | both | **thresh만**(box_w=0) | **box만**(thresh_w=0) |
|---|---|---|---|---|
| Heval_flip | 5.345 | 5.180 | **5.345** | **5.180** |
| lost | 250 | 229 | **250** | **229** |
| LVIS_AP | 0.2579 | 0.2566 | **0.2579** | **0.2566** |

**`thresh만` = `base`와 10개 지표 전부 bit-identical**(`thresh_w`를 1.0→2.0으로 올려도
동일), **`box만` = `both`와 완전히 동일**. 즉 `l_thresh`의 기여가 정확히 0이었다.

**원인 — `semantic_calib.py:169` 인덱싱 버그**:
```python
fp_positive = prob_fp[torch.arange(len(aidx), device=device), local_argmax] > det_thres
#                     ^^^ prob_fp는 전체 anchor 배열인데 앞에서부터 len(aidx)개 행을 본다
```
같은 함수의 다른 줄(`sub_fp[aidx]`, `sim_q[aidx, targets]`, `cv2_q[aidx]`)은 전부
`aidx`를 제대로 쓴다. 올바른 인덱싱은 `prob_fp[aidx, local_argmax]`(= `maxp[aidx]`).

**실측 (수정 전 → 후)**: reliable anchor 538개 중 `fp_positive` **0개 → 538개**,
그중 hinge 실제 발동 **0개 → 27개**(quant prob 최솟값 0.0000). 보호 대상이 실재했는데
항이 죽어 방치돼 있었다.

**영향 범위**: 이 helper를 쓰는 `optimize_semantic_pcal`(scripts/30), 구
`optimize_promptcal_scale_neighbor_utility`, 그리고 확정 경로의 `--combined-utility-frac`.
**과거에 utility/threshold 항에 성과를 귀속시킨 결과가 있다면 무효다.** 19-a에서 본
효과는 전부 `l_box` 단독이었다. 수정 커밋 `5bc00d6`.

### 19-c. 버그 수정 후 `l_thresh` 단독 측정 (`runs/104`) — 순수 이득 없음

`frac=0.3`, `box_w=0`으로 고정하고 `thresh_w` ∈ {1.0, 5.0, 20.0}:

| 지표 | base | box만 | **thr w=1** | **thr w=5** | brecq |
|---|---|---|---|---|---|
| Heval_flip | 5.345 | 5.180 | **5.175** 개선2/2 | 5.280 개선2/2 | **3.765** |
| lost | 250 | **229** | 240 개선2/2 | 240 (1/2) | **190** |
| LVIS_lost | 672 | **646** | 649 (1/2) | 702 악화2/2 | 630 |
| LVIS_AP | 0.2579 | 0.2566 | 0.2587 (1/2) | 0.2576 악화2/2 | 0.2551 |
| **LVIS_APr** | **0.1788** | 0.1758 | **0.1745** 악화2/2 | 0.1745 악화2/2 | **0.1789** |

`l_thresh`는 이제 작동한다(`Heval_flip` 2/2 개선). 그러나 **`LVIS_APr`이 2/2 악화로
`l_box`보다도 나쁘다**. 가중치를 올려도 개선되지 않는다 — `w=5`는 `w=1`보다 나쁘고
`LVIS_lost`는 base보다도 악화(702 vs 672).

seed0 가중치 사다리에서 `LVIS_APr`이 `0.1779 → 0.1711(w1) → 0.1775(w5) → 0.1813(w20)`로
**단조롭지 않게 요동**한다. Combined는 `--deterministic`에서 실행 분산이 0이므로 이건
노이즈가 아니라 실재하는 값이고, **방향성 있게 이끄는 게 아니라 흔들어놓는 데 가깝다**는
뜻이다.

**결론: 어떤 설정도 "decision 개선 + LVIS 유지"를 만들지 못한다.** 셋 다 COCO decision을
조금 얻고 `LVIS_APr`을 내주는 같은 모양이고, 최선(`box`, lost 229)조차 BRECQ의 190에
한참 못 미친다. **`base`(utility 꺼짐)가 Combined의 차별점을 가장 잘 지키는 설정이며
확정 설계는 불변이다.** 세 플래그 전부 기본 0.0 = 꺼짐 = bit-identical.

### 19-d. 다섯 번의 시도 종합

| | 무엇을 바꿨나 | 결과 |
|---|---|---|
| claim14 | 1단계 AdaRound(MSE로 alpha) | 보호 범위 밖 손상 |
| claim16 | `aux_mse`(dense 신호 추가) | 6-seed 반박 |
| claim18-a | margin_loss로 alpha | 10/10 지표 악화 |
| claim18-b | activation 범위 확장 | 7/10 악화, 이득 0 |
| **claim19** | **§4.3 utility 두 항** | **LVIS_APr 트레이드, BRECQ 근처 못 감** |

**논문 §4.3이 명시한 마지막 축까지 닫혔다.** "BRECQ를 decision 지표에서 이긴다"는
목표는 이 메커니즘 계열로는 도달 불가로 보인다 — 원인은 claim18-c의 진단(sparse
calibration-vocabulary 감독 + 저자유도 조합의 한계)과 일관된다.

데이터가 확실히 받쳐주는 대안 축은 **calibration 비용**이다(`runs/97` seed0 빌드 시간):
AdaRound 4304s / QDrop 2988s / BRECQ 1742s / **Combined 109s**. **BRECQ의 1/16 비용으로
cross-vocabulary AP 최상위권**(LVIS_AP 0.2576 vs BRECQ 0.2560, LVIS_APr 0.1788 vs 0.1789).
논문 §Experiments의 Efficiency(Calibration cost) 절에 직접 대응한다.

---

## Claim 20 — 논문의 핵심 주장이 실증됐다: 기반이 잘못돼 있었다 (09-27~28)

**배경**: claim19까지 다섯 번(claim14/16/18-a/18-b/19), 이후 `region_dir`·단측 margin까지
**여덟 번**의 시도가 모두 실패했다. 그런데 사용자가 *"논문 방향은 decision error를 직접
복원하는 게 아니라 **reconstruction만으로는 부족하니 ranking 보존을 추가**하자는 것인데
그게 잘 살려졌나"* 고 물었고, 확인해보니 **아니었다.**

### 20-a. 여덟 번의 실패는 전부 잘못된 기반 위에 있었다

확정 설계는 `--combined-recon-iters 0`, `--combined-stage1 none`이다. 즉
**reconstruction이 아예 없는 "naive + ranking"** 이었다. 논문이 주장하려던
"reconstruction + ranking"이 아니다.

```
논문의 주장   :  reconstruction          <  reconstruction + ranking
실제 비교     :  reconstruction (BRECQ)  vs  naive + ranking      ← 다른 비교
```

BRECQ는 수백만 파라미터로 오차를 줄이는데 우리는 그걸 빼고 s_mult 52개만 썼다.
**당연히 진다.** 논문 §4 Quantizer Initialization이 원래 *"strong reconstruction-based
PTQ 또는 Naive"* 로 둘 다 열어뒀는데, claim14에서 1단계를 제거한 뒤 모든 시도가
naive 기반에 묶여 있었다.

그리고 논문의 주장을 실제로 테스트한 실험은 claim15의 BRECQ-stage1 진단 하나였는데,
거기엔 두 결함이 있었다: **(1) 구 코드**(채널별 max-abs 대칭 weight, BRECQ 자체가
훨씬 나빴음 -- `Heval_flip` 7.28% vs 현재 3.80%) **(2) 1-seed**.

또한 노벨티 우려(*"BRECQ 위에 올리면 novelty가 약하지 않나"*)로 naive 기반을 고수한
것도 근거가 약했다 -- **PTQ 문헌에서는 표준 패턴**이다: QDrop = BRECQ + activation drop
(ICLR'22), PD-Quant = BRECQ + prediction difference (CVPR'23).

### 20-b. 확정 결과 (`runs/112`, W8A8, full probe, 6-seed, `--deterministic`)

`--conditions naive,brecq,combined --combined-stage1 brecq --combined-recon-iters 2000`.
`brecq` 열은 공식 BRECQ(LSQ 포함), `combined` 열은 BRECQ 재구성 + 우리 ranking
목적함수로 activation scale 학습. **같은 run 안 비교.**

| 지표 | naive | brecq | **ours** | Δ | seed별 |
|---|---|---|---|---|---|
| **LVIS_flip** | 4.467 | 2.723 | **2.227** | **−18.2%** | **6/6** |
| **LVIS_lost** | 985 | 622 | **461** | **−25.9%** | **6/6** |
| **LVIS_AP** | 0.2554 | 0.2560 | **0.2579** | +0.0020 | **6/6** |
| COCO_AP | 36.617 | 36.753 | 36.753 | ±0.000 | 3/5 (중립) |
| lost | 291 | 182 | 180 | −2 | 3/5 (중립) |
| LVIS_APr | 0.1777 | 0.1780 | 0.1774 | −0.0007 | 3/6 (노이즈) |
| **Heval_flip** | 5.945 | **3.802** | 4.278 | **+12.5%** | **1/6 (악화)** |
| Top1_flip | 0.667 | 0.367 | 0.378 | +0.012 | 2/6 (악화) |

**LVIS 세 지표가 6/6 일관 개선.** 무결성: `naive`가 `runs/106`과 6 seed 전부
bit-identical(비교 축 검증).

**손상 복구율**(naive 손상 = 100%, FP32 바닥 기준):

| 지표 | BRECQ | ours | |
|---|---|---|---|
| LVIS_AP | 17.5% | **72.6%** | +55.2pp |
| LVIS_lost | 38.5% | **53.2%** | +14.7pp |
| LVIS_flip | 41.7% | **50.1%** | +8.5pp |
| COCO_AP | 75.5% | 74.5% | −0.9pp |
| lost | 42.8% | 38.0% | −4.8pp |
| Heval_flip | **37.7%** | 28.0% | −9.6pp |

**전체 순위**(baseline은 `runs/106` 6-seed): LVIS 3개에서 **1/6**, `Top1_flip` 2/6,
나머지 3/6. 구 Combined는 5~6위였다. `LVIS_lost` 461은 baseline 4개가 몰려 있던
구간(606~985)을 확실히 벗어난다.

**확정된 서사**: *재구성은 calibration vocabulary의 결정을 잘 지키지만(Heval_flip 37.7%
복구) held-out vocabulary로 전이되지 않는다(LVIS_AP 17.5%). ranking 보존 항을 더하면
held-out 복구가 크게 늘고(LVIS_AP 72.6%, LVIS_lost 53.2%), 대가로 calibration
vocabulary의 flip을 일부 내준다(37.7% → 28.0%).*

**서술 주의**: `Heval_flip`은 이름에 held-out이 붙었지만 **calibration 이미지 × COCO-80
vocabulary 안에서** 측정된다. 진짜 vocabulary shift는 LVIS(1203 class)다. 논문에서 이
구분을 명확히 해야 "held-out이 좋아진다/나빠진다"가 모순으로 읽히지 않는다.

**비용**: `combined` 빌드가 stage1 BRECQ를 포함해 4929s(brecq 단독 2546s). 구 설계의
"BRECQ의 1/26 비용" 포지셔닝은 못 쓴다. 대신 **"BRECQ를 개선한다"는 원래 주장**을
되찾았고, QDrop·PD-Quant와 같은 표준 구성이라 방어가 쉽다.

### 20-c. 같이 기각된 세 수정안 (전부 기본 꺼짐 유지)

| 수정안 | 근거 | 결과 |
|---|---|---|
| `--margin-one-sided` | 대칭 `(q_m−fp_m)²`는 margin이 **넓어진** 것도 벌한다. 실측: 항의 48.1%가 넓어진 쪽, 전체 벌점의 30.4%가 그걸 억제 | **더 나쁨**(LVIS 3개 악화 2/2). 대칭 형태가 *"FP margin으로 되돌려라"* 라서 **암묵적 앵커** 역할을 하고, 단측은 그 앵커를 없애 s_mult가 표류한다(COCO lost는 개선, LVIS는 악화 = claim16/19의 calibration 과적합 패턴) |
| `--combined-random-sample` + `--combined-per-group-anchors` | 전자: `it % n` 결정적 순환으로 이미지 0~219는 6번, 220~255는 5번 쓰임(AdaRound/BRECQ는 09-18에 이미 고쳤음). 후자: anchor를 train_cols(60)로 선정하고 margin은 S(40)/H_cal(20)에서 따로 계산 | **순이득 없음**(`lost` 개선2/2이나 `COCO_AP`·`Heval_flip`·`LVIS_flip` 악화2/2). `per-group`이 anchor 수를 줄여 표본이 감소한 영향으로 보임 |
| `--combined-local-recon-weight` | 논문 §4 Semantic Objective의 *"Local reconstruction"* 절반이 구현에 없음. 대칭 margin이 암묵적 앵커라면 명시적 앵커는 더 나을 수 있다는 근거 | **무효**(8개 중 7개가 1/2 = 동전 던지기) |

### 20-d. bit-width 스캔과 다음 방향 (`runs/107`, `runs/108`, `runs/109`)

`--eval-cap 500`, 1-seed 스캔:

| 설정 | naive | brecq | 해석 |
|---|---|---|---|
| W8A6 | COCO_AP 31.87 | 34.59 | 손상 큼, 복구됨 |
| W8A5 | 23.41 | **14.36** | **BRECQ가 naive보다 나쁨** = 재구성 발산 |
| W8A4 / W8A3 | 0.02 / 0.00 | 0.01 / 0.00 | 전부 붕괴 |
| **W4A8** | **1.21** | **33.53** | naive 완전 붕괴, BRECQ 완전 복구 |
| **W4A4** | 0.00 | **0.73** | **BRECQ조차 붕괴** |

**activation 5bit 이하는 이 스킴(per-tensor asymmetric)으로 전부 죽는다.** W4A4로
가려면 우리 방법이 아니라 **activation 양자화 스킴 자체**를 고쳐야 한다(per-channel 등,
다만 claim4의 배포성 제약과 충돌).

Combined(naive 기반)의 동작 한계: **W8A8 7/7 개선 → W8A7 4/7 → W8A6 0/7**. BRECQ
기반을 깔아도 W8A6은 7/7 악화. **2비트 내려가는 사이에 부호가 뒤집힌다** -- s_mult는
"거의 맞는 출발점에 작은 배율 보정"을 전제하기 때문. **W8A8에서의 성공은 저비트로
전이되지 않는다.**

**W4A8이 다음 목표**다. 결정적 관찰: COCO `lost`에서 우리 항의 부호가 손상 수준에
따라 뒤집힌다 --

| | BRECQ | ours | |
|---|---|---|---|
| W8A8 (손상 작음) | 182 | 180 | 중립 |
| **W4A8** (손상 큼) | 203 | **185** | **ours 우세** (1-seed) |

**손상이 커질수록 우리 항이 유리해진다**는 가설을 뒷받침한다. 논문 서사로도 강하다.
W8A7/W8A6은 **Limitations**로 정직하게 기록할 것.

### 20-e. 실행 환경 발견

- **`OMP_NUM_THREADS`는 수치를 바꾼다** -- 30으로 줄이니 `naive`조차 달라졌다
  (LVIS lost 904 vs 902). 학습 없는 추론 경로까지 영향받으므로 **스레드 수는 건드릴 수
  없다.** 속도 레버는 `taskset` 범위뿐(이건 bit-identical 확인됨).
- **처리량은 GPU 수가 아니라 코어 수로 정해진다.** 120코어에서 4-way든 5-way든 빌드
  1개당 640~660s로 동일(192코어 2-way는 410s). **GPU 3장 + `taskset 0-159`가 GPU 5장
  + 120코어와 총 시간이 같고 load average는 206 → 87로 떨어진다.**

---

## claim21 — 공동 최적화(joint) 기각: W4A8에서 붕괴 (2026-09-28)

**설정** (commit `e4324f2`, `runs/114_joint`, seed 0·1, full probe, `--deterministic`)

```
--w-bits 4 --a-bits 8 --calib 256 --conditions naive,brecq,combined
--combined-stage1 brecq --combined-recon-iters 2000
--combined-block-recon-weight 1.0 --combined-learn-alpha
```

`naive`/`brecq` 열이 `runs/113_w4a8`과 **비트 동일** → 비교 가능.

**결과 (seed 0·1 평균, 같은 run 안에서 brecq와 비교)**

| 지표 | naive | brecq | 순차 ours | 공동 ours | 순차 | 공동 |
|---|---|---|---|---|---|---|
| COCO_AP | 1.140 | 33.510 | 32.930 | **18.485** | 악화 2/2 | 악화 2/2 |
| Heval_flip | 84.860 | 27.980 | 28.970 | **53.760** | 악화 2/2 | 악화 2/2 |
| Top1_flip | 59.640 | 5.150 | 5.445 | **22.925** | 악화 2/2 | 악화 2/2 |
| lost | 19644 | 2087 | 2066 | **8100** | 1/2 | 악화 2/2 |
| LVIS_flip | 88.440 | 20.745 | 24.610 | **50.345** | 악화 2/2 | 악화 2/2 |
| LVIS_lost | 18812 | 4704 | 5097 | **10588** | 악화 2/2 | 악화 2/2 |
| LVIS_AP | 0.0097 | 0.2296 | 0.2209 | **0.1285** | 악화 2/2 | 악화 2/2 |
| LVIS_APr | 0.0016 | 0.1614 | 0.1619 | **0.0942** | 1/2 | 악화 2/2 |

손상 복구율(naive=0%, FP=100%): LVIS_AP brecq 88.2% / 순차 84.8% / **공동 47.6%**.
COCO_AP brecq 90.8% / 순차 89.1% / **공동 48.6%**. 즉 공동 최적화는 BRECQ 재구성을
**절반 되돌려버렸다**.

**원인 (진단)**

로그: `nearest 대비 평균 flip 6.227% (max 26.697%), h->0/1 수렴 95%`.
claim18-a(W8A8)에서 alpha가 **거의 안 움직여** 무효였던 것과 정반대로, W4에서는
alpha가 크게 움직였다. 그런데 그 움직임을 이끈 목적함수는 **256장 calib에서 뽑은
reliable anchor 538개짜리 decision loss**다. BRECQ가 alpha를 맞출 때 쓰는
제약(전 calib 셋에 대한 블록 MSE)보다 자유도 대비 제약이 압도적으로 부족해서,
수백만 개 alpha가 calib 538 앵커에 과적합하며 일반화를 파괴한다.
`block_recon_weight=1.0`(정규화 MSE 17블록 평균)은 이 과적합을 막지 못했다.
`alpha_warmup=0.2` → 2000 iters 중 1600 iters를 `alpha_lr=1e-2`로 학습한 것도 과도.

**교란 요인 (미해소)**: `block_recon`과 `learn_alpha`를 동시에 켰다. 다만
`learn_alpha` 없이 블록 재구성만 켜면 학습 파라미터가 **s_mult 52개뿐**이므로
블록 MSE 항은 s_mult를 "재구성 최적점(=아무것도 안 하기)"으로 당기는 정규화로만
작동한다 — 순차와 거의 같아질 것이 예측되며, 붕괴의 원인은 `learn_alpha`로
사실상 특정된다.

**결론**: 사전에 정한 판정 기준("W4A8에서 brecq를 넘으면 저비트가 열린다 /
둘 다 안 되면 순차·공동 구분이 원인이 아니다")에 따라 **공동 최적화 기각**.
논문은 **W8A8 순차 결과(runs/112, 6-seed, LVIS 3지표 6/6)** 로 쓴다.
저비트(W4A8/W4A4) 확장은 decision loss의 **제약 부족**이 본질 문제이므로,
앵커 수를 늘리거나(calib 확대) weight를 건드리지 않는 설계를 유지하는 쪽이
남은 경로다.

---

## claim22 — alpha_bias(저자유도 rounding 보정) 기각, 그리고 **claim21의 진단이 틀렸다** (2026-09-28)

**동기**: claim21은 W4A8 공동최적화 붕괴의 원인을 **자유도 폭발**로 진단했다 —
"alpha는 수백만 개인데 decision_loss의 reliable anchor는 538개뿐이라 calib에
과적합한다". 이 진단이 맞다면 **자유도만 줄이면 된다**. 그래서 `alpha_bias`를
만들었다: conv당 스칼라 **1개**(전체 52개, `s_mult`와 정확히 같은 자릿수)로
`h_alpha(alpha + alpha_bias)`처럼 반올림을 conv 전체에 균일하게 민다.
BRECQ가 원소별로 맞춰둔 alpha는 건드리지 않는다.

**구현 (commit 대기, `runs/120_alphabias_ste`)**
- `round_ste`: forward는 hard 반올림(=bias 0이면 BRECQ 모델과 **bit-identical**),
  backward만 `h_alpha`로 흘리는 STE. 첫 시도(`runs/119`, 폐기)는 `soft=True`를 썼는데
  그러면 "최적화 대상 모델"이 바뀌어 s_mult-only 실험(runs/113)과의 비교에 교란이
  섞인다(실측: soft vs hard weight 3.77% 차이). STE로 이 교란을 제거했다.
- `eff_alpha_bias()`: forward clamp(±limit, 기본 0.5). Adam은 gradient 크기와
  무관하게 스텝당 ~lr씩 움직여서 1500 iter면 |bias|가 이론상 15까지 간다.
- `use_alpha_bias` 플래그: 꺼져 있으면 덧셈 노드조차 안 만든다(기존 경로 완전 동일).

**검증**: `naive`가 3/3 seed 모두 `runs/113`과 **bit-identical**(1.21/1.07/1.06) —
기능 추가가 기존 조건을 안 건드림이 실증됨.

**결과 (3-seed, W4A8, head 제외, full probe)**

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_flip | LVIS_lost | LVIS_AP | LVIS_APr |
|---|---|---|---|---|---|---|---|---|
| brecq | 33.48 | 27.77 | 5.09 | 2042 | 20.50 | 4654 | 0.2303 | 0.1649 |
| s_mult만 ours (runs/113) | 32.96 | 28.68 | 5.34 | 2065 | 24.20 | 5045 | 0.2227 | 0.1634 |
| **alpha_bias ours (runs/120)** | **27.77** | **40.30** | **9.37** | **3599** | **34.47** | **6798** | **0.1713** | **0.1161** |

brecq 대비 8개 지표 **전부 0/3**(전 seed 악화): COCO_AP -5.71, LVIS_AP -0.0590,
LVIS_APr -0.0488, Heval_flip +12.53, LVIS_flip +13.97, lost +1557, LVIS_lost +2144.

brecq 대비 COCO_AP 델타: s_mult만 `-0.37/-0.79/-0.52`(평균 **-0.56**) vs
alpha_bias `-7.33/-5.36/-4.44`(평균 **-5.71**). **10배 더 나쁘다. 3/3 붕괴.**

**진단 (claim21 정정)**

| | 학습 파라미터 | nearest 대비 flip | W4A8 결과 |
|---|---|---|---|
| BRECQ 자신 | alpha 수백만 | 0.50% | 33.48 |
| claim21 `learn_alpha` | **수백만** | 6.23% (max 26.7%) | 붕괴 18.49 |
| claim22 `alpha_bias` | **52개** | 6.60% (max 25.4%) | 붕괴 27.77 |

자유도를 **수만 배** 줄였는데 붕괴가 그대로 재현됐다. 공통 변수는 파라미터 수가
아니라 **"BRECQ 반올림에서 얼마나 멀어졌는가"(flip 6%대)**다. 즉 claim21의
"자유도 폭발" 진단은 **틀렸다** — 진짜 원인은 **W4에서 decision_loss가 weight
rounding을 건드리는 행위 자체**이고, 손잡이 개수와 무관하다.

부수 관측: `|bias|` 평균이 0.26~0.30이고 52개 중 **15~23개가 상한 0.5에 걸렸다**
(3/3 seed). L2 reg 1e-3은 전혀 저항하지 못했다 — 최적화기가 일관되게 상한까지
밀어붙인다. 가설: `ContrastiveHead`가 region embedding을 L2 정규화하므로 weight
전역 스케일은 similarity에 거의 영향이 없고, `alpha_bias`(반올림 일괄 이동)와
`s_mult`(activation scale)가 서로 상쇄 가능한 **평평한 방향**을 만들어 그 축을
따라 표류하다 clamp에 부딪히는 것으로 보인다(미검증).

**결론**: W4A8에서 rounding에 접근하는 경로는 자유도를 어떻게 조절하든 닫혀 있다.
`s_mult`(activation scale만, rounding 불가침)가 W4A8에서 brecq에 0.56 지는 것이
이 설계의 바닥이며, 그보다 잘하려면 **rounding이 아닌 다른 축**을 찾아야 한다.
저비트 확장은 두 번 연속 실패했고(claim21, claim22), 논문은 **W8A8**로 쓴다.
상한 sweep(0.05~0.1)이 형식상 남아 있으나, bias 0.26에서 -5.71, bias 0에서
-0.56이라 단조 악화 곡선으로 보여 중간에 brecq를 넘는 지점이 있을 근거가 없다.
