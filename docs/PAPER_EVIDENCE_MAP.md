# 논문 절별 증거 매핑 (2026-09-26)

**대상 독자**: 이 논문을 쓰는 사람.

`논문.txt`는 아직 `%` 주석 뼈대다. 그리고 **09-25에 포지셔닝이 바뀌었는데
(decision 우위 → 비용 + cross-vocabulary, `PROMPTCAL_CURRENT_MODEL_V3.md` §5)
그 뒤로 증거를 다시 매핑한 적이 없다.** 이 문서는 절마다
**① 쓸 수 있는 주장 ② 뒷받침하는 측정 ③ 빠진 것**을 정리한다.

- 방법 현재 상태 → `PROMPTCAL_CURRENT_MODEL_V3.md`
- baseline 확정 수치·재현 → `pipeline/BASELINE_STATUS.md`
- 원 논문 대조 → `PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md`
- 시도 경위 → `PROMPTCAL_CLAIMS_2026-09-15.md` claim1~19

> 표기: ✅ 측정 완료 / ⚠️ 부분적 / ❌ 없음

---

## Abstract — **문장 단위로 손봐야 한다**

| 초록의 주장 | 상태 |
|---|---|
| "배포 시 달라지는 프롬프트 집합을 기존 PTQ가 고려하지 않음" | ✅ 유지 |
| "작은 수치 오차에도 region-prompt 순위 재배열 / 미포함 프롬프트 침입 / semantic substitution" | ✅ 유지 |
| "일반적인 reconstruction error는 semantic decision damage를 **설명하지 못함**" | ⚠️ **스케일·상관 수준으로 한정해서 써야 한다.** §7.2가 뒷받침하지만, "reconstruction **방법**이 decision을 못 지킨다"로 읽히면 자기 데이터와 충돌한다(아래) |
| "prompt ranking 보존만으로 최종 탐지 성능은 보장되지 않음" | ✅ 유지 |
| "제안 방법이 기존 reconstruction 기반보다 region-prompt decision과 탐지 정확도를 **일관되게 보존**" | ❌ **반박됨. 반드시 교체.** BRECQ가 decision 지표에서 우세(아래 §5.2 표) |

**교체할 주장**: *"제안 방법은 reconstruction 기반 PTQ의 **1/24 calibration 비용**으로
cross-vocabulary 의미 보존에서 동등 이상을 달성하며, 이는 (비용, 일반화) 축에서
기존 방법과 다른 동작점을 차지한다."*

**주의(자기 모순 방지)**: naive 5.945% → BRECQ 3.705%로 **reconstruction이 held-out
decision을 38% 개선한다.** v1 시절 문서의 *"강한 PTQ 넷 다 손상의 ~91%를 동일하게
남긴다"* 는 이 설정에서 성립하지 않으므로 Motivation에 남기면 안 된다.

---

## §2 Related Work — ✅ 텍스트만 쓰면 됨

측정 불필요. baseline 3종의 구현 충실도는 `PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md`가
항목별로 대조해두었으므로, "공식 설정을 재현했다"는 서술의 근거로 인용 가능.

---

## §3 Motivation — **가장 단단한 부분. 주력으로 승격 권장**

### §3.1 Preliminary ✅
- region-prompt similarity 정의: `ContrastiveHead`의 `sim_j = x̂ · ŵ_j` (코드 그대로)
- 지표 정의: `run_comparison.py`의 `group_flip`(L130) / `standard_flip`(L147) /
  `gt_metrics_for_method`(L242) / `compute_lvis_flip_gt_streaming`(L364)
- Reconstruction fidelity / semantic fidelity / detection utility 3분류도 지표가 이미 그 구조

### §3.2 Precision-Induced Semantic Damage ⚠️
claim17이 **bit-width 분해**를 확정했다 — 이게 이 절의 핵심 수치다:

| 설정 | COCO_AP | FP32 대비 | LVIS_AP | FP32 대비 |
|---|---|---|---|---|
| FP32 | 36.80 | — | 0.2589 | — |
| W8A32 (weight만) | 36.75 | **−0.05** | 0.2588 | **−0.0001** |
| W32A8 (activation만) | 33.51 | **−3.29** | 0.2324 | **−0.0265** |
| W8A8 | 33.54 | −3.26 | 0.2342 | −0.0247 |

**손상은 거의 전부 activation 양자화에서 온다.** 제안 방법이 activation scale만
건드리는 설계 근거가 여기서 직접 나온다 — 논문 서술에 꼭 넣을 것.

❌ **빠진 것: FP16 / FP8.** 논문은 "FP32/FP16/FP8/INT8 precision ladder"를 명시하는데
FP16·FP8 측정이 없다. `--w-bits`/`--a-bits`가 정수 비트만 받는다. 넣으려면
구현이 필요하고, 안 넣으려면 ladder 서술을 INT8 중심으로 줄여야 한다.

### §3.3 Prompt-Axis Shift ⚠️
- seen(S) vs held-out(H_eval) 비교: ✅ 모든 run이 `S_AP`/`H_eval_AP`/`Heval_flip` 산출
- COCO→LVIS vocabulary shift: ✅ `LVIS_flip`/`LVIS_lost`/`LVIS_AP`/`APr`
- ❌ **semantic hard negatives vs random negatives 대조 실험이 없다.** `text_neighbor_order`로
  이웃을 뽑는 기계는 있으니(`semantic_calib.py`) 구현 부담은 작다. 논문이 명시한 비교라 필요.
- ⚠️ claim6/16이 발견한 **구조적 제약을 반드시 각주로 밝힐 것**: COCO-80에서
  S(40)+H_cal(20)+H_eval(20)이 정확히 80개 전부라 **이웃 후보 풀이 H_cal과 수학적으로 동일**하다.
  이 때문에 `neighbor_loss`가 `margin_loss(H_cal)`와 중복이고, "prompt 경쟁 증가"를
  독립 변수로 조작할 수 없다. LVIS 쪽으로 넘어가야 풀린다.

### §3.4 Reconstruction–Utility Misalignment ✅ **가장 강한 실증**
`PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md` §7.2 — AdaRound의 activation observer만
min-max(클리핑 없음) ↔ MSE(재구성 최적)로 바꾼 **단일 변수 6-seed** 실험:

| | COCO_AP | COCO Heval_flip | COCO lost | LVIS_AP | LVIS flip | LVIS lost |
|---|---|---|---|---|---|---|
| min-max | 36.648 | 5.142% | 234 | **0.2577** | **2.813%** | **639** |
| MSE(재구성 최적) | **36.757** | **4.567%** | **210** | 0.2542 | 3.603% | 824 (**+29%**) |
| 방향 일치 | 6/6 | 6/6 | 5/6 | **6/6 반대** | **6/6 반대** | **6/6 반대** |

**재구성 오차를 최소화한 스케일이 held-out vocabulary에는 최적이 아니다** — 6 seed
전부에서 COCO와 LVIS가 **반대 방향**으로 갈린다. 논문 §3.4 주장의 교과서적 실증이고,
단일 변수 통제라 반론하기 어렵다.

"Rank preservation alone이 AP를 보장하지 않음"도 ✅ — claim19에서 `l_thresh`/`l_box`
어느 쪽도 rank 개선이 AP로 이어지지 않음을 보였다.

---

## §4 Methodology — **구현과 서술을 일치시켜야 한다**

| 논문 소절 | 실제 구현 | 상태 |
|---|---|---|
| Quantizer Initialization | MSE observer(채널별 비대칭) + round-to-nearest. **AdaRound 미사용**(claim14) | ⚠️ 논문은 "strong reconstruction-based PTQ 또는 naive"를 열어두는데, 실제로는 naive만 쓴다. 이유(claim14: 1단계가 보호 범위 밖으로 손상을 샌다)를 본문에 밝힐 것 |
| Region Filtering | `reliable = maxp > 0.25 & margin > 0.5` | ✅ |
| Prompt Selection | `text_neighbor_order`(FP text encoder만, GT 불필요) | ✅ 논문 서술과 일치 |
| Semantic Objective: "local reconstruction + region-prompt semantic consistency" | `margin_loss + neighbor_loss` (+ 09-25 `region_dir`) | ⚠️ **"local reconstruction" 항이 원래 없었다.** `region_dir`(cv4 입력 단위 방향 보존)이 그 자리를 채우는 형태 — 확정되면 이 서술이 맞아떨어진다 |
| Utility Constraints (threshold crossing, box consistency) | 구현돼 있으나 **기본 꺼짐** | ❌ claim19에서 순수 이득 없음 확인. **본문에서 빼거나 "시도했고 이랬다"로 기록**. 특히 `l_thresh`는 인덱싱 버그로 **한 번도 작동한 적이 없었다**(커밋 `5bc00d6`) |
| Optimization: "scale/clipping/rounding parameter만" | **scale(s_mult 52개)만** | ⚠️ rounding(claim18-a)·clipping(claim18-b) 둘 다 시도했고 악화. 서술을 "scale만"으로 정정하고, 나머지 둘은 ablation으로 |
| Complexity: inference-time zero overhead | ✅ s_mult는 activation scale에 흡수 가능 | ✅ 단 **실측 latency 없음**(§5.4) |

---

## §5 Experiments

### §5.1 Setup ⚠️
- ✅ COCO / LVIS, W8A8, calib 256, operator scope(head 제외·CLIP 제외)
- ❌ **OWLv2 코드가 저장소에 0줄.** `grep`에 `owlv2|owl_vit|owlvit` 전무

### §5.2 Main Results ✅ (YOLO-World 한정)
`runs/97` 6-seed 확정값:

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP | LVIS_APr | **빌드** |
|---|---|---|---|---|---|---|---|
| naive | 36.617 | 5.945% | 0.667% | 291 | 0.2554 | — | 7s |
| AdaRound | 36.648 | 5.142% | 0.472% | 234 | 0.2577 | — | 4304s |
| QDrop | **36.807** | 4.090% | 0.397% | 180 | 0.2556 | — | 2988s |
| **BRECQ** | 36.755 | **3.705%** | **0.345%** | **166** | 0.2560 | **0.1790** | 1742s |
| Combined | 36.642 | 5.602% | 0.527% | 248 | 0.2576 | 0.1788 | **109s** |

**이 표를 그대로 쓰되 헤드라인 주장을 비용 축으로 잡을 것.** decision 지표를
숨기면 안 된다 — 숨기면 재현 시 바로 드러난다.

⚠️ `runs/106`(진행 중)이 이 표를 `--deterministic` + `region_dir` on/off로 갱신한다.

### §5.3 Generalization
- Cross-Vocabulary (seen/held-out, COCO→LVIS): ✅
- ❌ **Cross-Architecture (OWLv2): 전무.** 비용 축 주장은 아키텍처 하나로는
  *"이 모델 특성 아니냐"* 는 반론에 약하다. **두 번째 아키텍처가 이 포지셔닝에서
  오히려 더 중요해졌다.**

### §5.4 Analysis
| 항목 | 상태 |
|---|---|
| Ablation: Semantic Calibration on/off | ✅ `--control-mse`(margin→MSE 대체), claim15 |
| Ablation: Utility Refinement on/off | ✅ claim19 (결과는 음성) |
| Ablation: Initialization | ✅ claim14 (`--combined-stage1 none/adaround/brecq`) |
| Efficiency: **calibration cost** | ✅ **핵심 수치.** 132.5s vs BRECQ 3173.0s = **1/24** (동일 5-way 병렬) |
| Efficiency: model size | ✅ 9.88 MiB (CLIP 오염 수정 후, claim18 참고) |
| Efficiency: **latency / memory / 실제 INT8 배포** | ❌ **전무.** `tensorrt`/`onnx`/`torch.quantization` 코드 0줄. 지금은 fake-quant 시뮬레이션이라 **실제 속도·메모리 이득을 측정한 적이 없다.** 비용을 헤드라인으로 내세우면 리뷰어가 반드시 짚는다 |

---

## 우선순위 (빠진 것 기준)

| | 항목 | 왜 | 규모 |
|---|---|---|---|
| **1** | **실제 INT8 배포 측정** (latency/memory) | 비용이 헤드라인인데 배포 측 수치가 없다 | 중 |
| **2** | **OWLv2 cross-architecture** | 단일 아키텍처 반론 차단, 논문이 명시 | 대 |
| 3 | semantic hard negatives vs random (§3.3) | 논문이 명시한 비교, 기계는 이미 있음 | 소 |
| 4 | FP16/FP8 ladder (§3.2) | 없으면 ladder 서술을 INT8 중심으로 축소 | 소~중 |

**1번이 2번보다 급하다** — 포지셔닝이 비용으로 바뀐 순간 §5.4가 논문의 근거이지
부록이 아니게 됐다.
