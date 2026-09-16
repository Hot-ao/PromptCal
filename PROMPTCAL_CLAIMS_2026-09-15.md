# PromptCal-PTQ — Claim 1~10 검증/변경 기록 (2026-09-15~)

이 문서는 `pipeline/` 공식 데이터 설정 포팅(09-15) 이후, 리뷰어 공격 포인트
관점에서 제기된 claim 1~10을 코드로 검증하고 실제로 바꾼 내용을 정리한다.
**`PROMPTCAL_CURRENT_MODEL.md`의 §8.1 확정 하이퍼파라미터는 이 문서의 어떤
내용으로도 아직 바뀌지 않았다** — 여기 나오는 변경들은 전부 opt-in 플래그로
구현돼서 기존 확정 동작과 병행 가능하고, "채택" 여부는 별도 결정 사항으로
남겨둔다. git 커밋도 아직 안 했다(`git status`로 확인 가능, 아래 부록 참고).

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

**상태**: 미결정, 사용자 판단 대기.

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

## 부록 B — 코드 변경 파일 목록 (전부 uncommitted)

| 파일 | 변경 내용 |
|---|---|
| `src/quant/adaround.py` | `AdaRoundQuantConv2d.__init__`/`convert_to_adaround`에 `channelwise_smult` 파라미터 추가(claim4) |
| `src/quant/promptcal.py` | confident-anchor 선정 `train_cols` 제한(claim1), `margin_loss`에 `identity_aware` 파라미터 추가(claim5), `optimize_promptcal_scale_neighbor`에 `identity_aware_margin` 파라미터 전달 |
| `pipeline/quant/adaround.py`, `pipeline/quant/promptcal.py` | 위 두 파일과 동기화(diff 없음 확인) |
| `pipeline/run_comparison.py` | `--smult-per-tensor`, `--identity-aware-margin` CLI 플래그 추가, `build()`에 `channelwise_smult`/`identity_aware_margin` 파라미터 전달, `measure_ap`의 클래스별 AP 매핑 수정(claim7), `switch_vocab`의 names/predictor 갱신 추가(claim8), `--eval-cap` help 문구·결과 표 안내 문구 추가(claim10c) |
| `requirements.txt` | `ultralytics>=8.3.0` → `ultralytics==8.4.121`로 고정(claim9) |
| `scripts/58_full_baseline_official_data.py` | claim7/8과 동일한 두 수정 동기화 |

## 부록 C — 실행 로그 위치

| 실험 | 위치 |
|---|---|
| claim1 전/후 1-seed 비교 | `runs/62_anchorfix/seed0_{before,after}.log` |
| claim4 per-tensor vs per-channel(seed0) | `runs/62_anchorfix/seed0_after.log`(per-channel) + `runs/63_smult_pertensor/seed0_pertensor.log` |
| claim4 per-tensor 6-seed(vs baseline) | `runs/63_smult_pertensor/seed{0..5}_pertensor.log` |
| claim5 identity-aware ON 6-seed | `runs/64_identity_margin/seed{0..5}_identity.log` |
| claim5 identity-aware OFF 6-seed(seed0은 `62_anchorfix` 재사용) | `runs/65_identity_off/seed{1..5}_off.log` |
| claim3 rounding_diff_rate 측정 | 스크립트만 존재(`/tmp` scratchpad, 저장소에 커밋 안 됨) — 재현하려면 이 문서의 claim3 절차대로 재실행 필요 |

## 부록 D — 다음에 결정할 것들 (우선순위 순)

1. **claim5(identity-aware margin)를 §8.1 확정 설계로 승격할지** — 가장
   근거가 탄탄한 후보.
2. claim1 수정을 6-seed(같은 물리 GPU)로 재검증 후 §8.1 갱신할지.
3. claim6의 논문 서술을 어떻게 고칠지(방법론 문장 수정 vs 보조 실험 추가).
4. claim4의 per-tensor 결과를 논문에 disclosure로 쓸지, 아니면 per-tensor
   자체를 새 기본값으로 밀고 갈지(현재는 비권장).
5. 위 결정들이 나면 이 문서 내용을 `PROMPTCAL_CURRENT_MODEL.md`로 승격/병합하고
   git commit.
