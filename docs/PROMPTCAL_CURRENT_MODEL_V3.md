# PromptCal 제안 방법 — 현재 상태 정리 (V3, 2026-09-25)

**대상 독자**: 이 방법을 이어서 설계하거나 논문을 쓰는 사람.

V2(`PROMPTCAL_CURRENT_MODEL_V2.md`, 09-17)는 설계·하이퍼파라미터·결과의 단일 진실
공급원이었다. 09-23~09-25 세션에서 **baseline이 정합성 수정으로 바뀌었고**, 그 위에서
**격차를 좁히려는 다섯 번의 시도가 전부 실패**했다. 이 문서는 그 이후의 상태를 정리하고
**남은 선택지를 정직하게 제시**한다.

- 설계 세부·목적함수 유도·프롬프트 3분할 정의 → **V2 §5** (여전히 유효)
- baseline 확정 수치·재현 절차 → **`pipeline/BASELINE_STATUS.md`**
- baseline이 원 논문과 어떻게 대응되는지 → **`PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md`**
- 각 시도의 상세 경위 → **`PROMPTCAL_CLAIMS_2026-09-15.md` claim14~19**
- **논문 절별 증거 매핑·빠진 실험** → **[`PAPER_EVIDENCE_MAP.md`](PAPER_EVIDENCE_MAP.md)** (09-26)


> ## 🔴 09-28 — 이 문서의 §2~§6은 **뒤집혔다**
>
> **원인이 특정됐다: 여덟 번의 실패는 전부 잘못된 기반 위에 있었다.** 확정 설계가
> `--combined-stage1 none`(재구성 없음)이라 "naive + ranking"을 테스트하고 있었고,
> 논문이 주장하려던 **"reconstruction + ranking"** 이 아니었다. 논문 §4가 원래
> *"strong reconstruction-based PTQ 또는 Naive"* 로 둘 다 열어뒀는데 claim14에서
> 1단계를 제거한 뒤 모든 시도가 naive 기반에 묶여 있었다.
>
> `--combined-stage1 brecq`로 되돌리니 **6-seed에서 LVIS 세 지표가 6/6 개선**됐다
> (`runs/112`): `LVIS_flip` −18.2%, `LVIS_lost` −25.9%, `LVIS_AP` +0.0020.
> **포지셔닝 전환(§5)은 불필요하다 — 원래 주장이 성립한다.**
>
> 아래 §2(Pareto-dominated), §3(닫힌 방향), §5(대안 포지셔닝)는 **naive 기반 시절의
> 기록**으로 읽을 것. 현재 상태는 **claim20**과 `pipeline/BASELINE_STATUS.md`를 보라.
>
> ### 확정 설계 (09-28)
> ```
> --conditions naive,brecq,combined
> --combined-stage1 brecq --combined-recon-iters 2000   ← 1단계 BRECQ 재구성
> --deterministic --calib 256 --iters 1500              ← 2단계 s_mult(52개) ranking 학습
> (margin-one-sided / random-sample / per-group-anchors / local-recon / region-dir
>  / range-blend / learn-alpha 전부 기본 꺼짐 = 측정으로 기각됨, claim19·20-c)
> ```
>
> ### 확정 수치 (`runs/112`, W8A8, full probe, 6-seed)
> | 지표 | naive | BRECQ | **ours** | seed별 |
> |---|---|---|---|---|
> | **LVIS_flip** | 4.467 | 2.723 | **2.227** | **6/6** |
> | **LVIS_lost** | 985 | 622 | **461** | **6/6** |
> | **LVIS_AP** | 0.2554 | 0.2560 | **0.2579** | **6/6** |
> | COCO_AP | 36.617 | 36.753 | 36.753 | 중립 |
> | lost | 291 | 182 | 180 | 중립 |
> | **Heval_flip** | 5.945 | **3.802** | 4.278 | **악화 5/6** |
>
> 손상 복구율: `LVIS_AP` 17.5% → **72.6%**, `LVIS_lost` 38.5% → **53.2%**,
> 대가로 `Heval_flip` 37.7% → 28.0%.
>
> **서술 주의**: `Heval_flip`은 이름에 held-out이 붙었지만 calibration 이미지 ×
> COCO-80 안에서 측정된다. 진짜 vocabulary shift는 LVIS(1203 class)다.
>
> ### 다음
> **W4A8**(손상이 큰 영역). COCO `lost`에서 우리 항의 부호가 손상 수준에 따라 뒤집힌다
> — W8A8에서는 BRECQ 182 vs ours 180(중립), **W4A8에서는 203 vs 185(ours 우세, 1-seed)**.
> **W4A4는 BRECQ조차 붕괴**(COCO_AP 0.73)하므로 activation 양자화 스킴 선행 개선 필요.
> W8A7/W8A6(방법이 무너지는 지점)은 Limitations로 기록.

---

## 1. 확정 설계 (변경 없음)

**2단계 구조이고, 실제로는 2단계만 작동한다.**

| 단계 | 내용 | 상태 |
|---|---|---|
| 1단계 weight rounding | **round-to-nearest 고정** (`--combined-recon-iters 0`) | claim14로 확정. AdaRound 메커니즘 미사용 |
| 2단계 activation scale | `s_mult` — **conv당 스칼라 1개, 총 52개** | `margin_loss + neighbor_loss` + `scale_reg`로 학습 |

```
--calib 256  --iters 1500  --lr 1e-2  --k 5  --neighbor-k 5
--neighbor-weight 1.0  --scale-reg-weight 1.0  --cal-weight 1.0
--smult-per-tensor  --identity-aware-margin  --act-observer mse
--combined-recon-iters 0  --combined-stage1 none
```

**09-24 이후 추가된 플래그는 전부 기본 꺼짐이고, 켜지 않으면 bit-identical이다**
(`--combined-learn-alpha`, `--combined-range-blend 0.0`, `--combined-utility-frac 0.0`).
전부 실패한 시도의 잔재이며 재현/ablation용으로만 남긴다.

**한 가지 실행 설정이 추가됐다**: `--deterministic`. Combined는 이게 없으면 같은 seed에서도
결과가 달라진다(52/52 conv, `max|Δ| 0.248`). **Combined를 측정하는 모든 실행에 켤 것.**
baseline 수치는 이 플래그로 바뀌지 않는다(claim18-d 참고, BRECQ만 미세 예외).

---

## 2. 현재 위치 — Combined는 BRECQ에 Pareto-dominated

`runs/97` 6-seed 확정값(AdaRound 예산 정렬 후):

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP | LVIS_APr | **빌드** |
|---|---|---|---|---|---|---|---|
| naive | 36.617 | 5.945% | 0.667% | 291 | 0.2554 | — | **7s** |
| AdaRound | 36.648 | 5.142% | 0.472% | 234 | **0.2577** | — | 4304s |
| QDrop | **36.807** | 4.090% | 0.397% | 180 | 0.2556 | — | 2988s |
| **BRECQ** | 36.755 | **3.705%** | **0.345%** | **166** | 0.2560 | **0.1790** | 1742s |
| **Combined** | 36.642 | 5.602% | 0.527% | 248 | 0.2576 | 0.1788 | **109s** |

seed별 짝비교에서 Combined는 **naive에만 전승**이고 AdaRound·QDrop·BRECQ에게는
decision 지표 **0/6 전패**다.

**중요**: 이 표는 09-24에 baseline을 고친 뒤의 것이다. 이전에는 AdaRound가 절반
예산(iters 1000 vs 2000)으로 돌아 과소평가돼 있었고, 그걸 고치자 **AdaRound의 LVIS_AP가
0.2577로 올라와 Combined(0.2576)와 동률**이 됐다. 즉 **Combined의 유일한 우위조차
baseline 오류에 의존하고 있었다.**

---

## 3. 닫힌 방향 다섯 개

전부 같은 벽을 가리킨다. 상세는 claim14/16/18/19.

| | 무엇을 바꿨나 | 결과 |
|---|---|---|
| claim14 | 1단계 AdaRound (MSE 목적함수로 alpha) | 보호 범위 밖 손상 |
| claim16 | `aux_mse` (train_cols에 dense 신호) | 6-seed 반박 |
| claim18-a | **margin_loss로 alpha** 공동 최적화 | **10/10 지표 악화** |
| claim18-b | **activation 범위(클리핑) 확장** | **7/10 악화, 이득 0** |
| claim19 | **§4.3 utility 두 항** (`l_thresh`/`l_box`) | **LVIS_APr 트레이드, BRECQ 근처 못 감** |

### 진단 — 감독 신호의 밀도 대비 자유도

| | 파라미터 | 감독 신호 |
|---|---|---|
| BRECQ | alpha ~수백만 + LSQ 52 | **dense** — 모든 블록의 모든 출력 원소 |
| **Combined** | **s_mult 52개** | **sparse** — confident anchor의 top-(k+1) 경계, 40~60 컬럼 |

BRECQ가 수백만 파라미터를 감당하는 건 감독이 촘촘해서다. margin_loss는 설계상 sparse라
자유도를 늘리면 calibration anchor를 외운다(18-a). 그렇다고 dense 신호를 얹으면 그게
곧 **calibration vocabulary 과적합**이라 held-out(LVIS)이 무너진다(claim16, 19).

**s_mult가 52개뿐인 게 약점이 아니라 정규화 장치였다.** 이게 Combined가 LVIS에서
버티는 이유이자, decision 지표에서 BRECQ를 못 따라가는 이유다 — 같은 원인이다.

### 부수 발견 — §4.3 threshold 항은 한 번도 작동한 적이 없었다

`semantic_calib.utility_refinement_terms`의 `fp_positive`가 `prob_fp`(전체 anchor 배열)를
`torch.arange(len(aidx))`로 인덱싱하고 있었다. 실측 결과 reliable anchor 538개 중
`fp_positive`가 **0개** → `l_thresh`가 항상 정확히 0. 커밋 `5bc00d6`에서 수정했고,
수정 후 538/538 중 27개에서 실제로 hinge가 발동한다. **이 helper를 쓰는 모든 코드
(scripts/30 포함)의 과거 결과에서 threshold 항에 성과를 귀속시킨 게 있다면 무효다.**

---

## 4. 논문 초록과 구현의 간극 (정리해야 할 것)

초록은 *"calibration vocabulary에 대한 과적합을 억제하여 새로운 vocabulary에서도 의미적
의사결정을 보존"* 이라고 한다. 그런데 **현재 목적함수의 모든 항이 calibration
vocabulary(S ∪ H_cal) 위에서 계산된다**:

- `margin_loss` — S, H_cal 컬럼
- `neighbor_loss` — COCO-80 내 이웃. **claim16이 증명했듯 구조적으로 H_cal과 정확히 동일**
  (S∪H_cal∪H_eval = COCO-80 80개 전부라 이웃 후보 풀이 H_cal로 포화)
- `scale_reg` — s_mult를 1.0(MSE 재구성 최적 범위)으로 당김

**unseen vocabulary를 대변하는 항이 하나도 없다.** 과적합 억제 담당이던
`neighbor_loss`가 COCO-80 폐쇄 구조에서 무력화된 상태다. 초록의 주장을 유지하려면
이 간극을 메우거나 서술을 바꿔야 한다.

---

## 5. 데이터가 받쳐주는 대안 포지셔닝

**"BRECQ를 decision 지표에서 이긴다"는 이 메커니즘 계열로 도달 불가로 보인다.**
대신 측정값이 확실히 지지하는 축이 있다:

| | 빌드 시간 | LVIS_AP | LVIS_APr |
|---|---|---|---|
| AdaRound | 4304s | 0.2577 | — |
| QDrop | 2988s | 0.2556 | — |
| BRECQ | 1742s | 0.2560 | 0.1790 |
| **Combined** | **109s (BRECQ의 1/16)** | **0.2576** | 0.1788 |

**BRECQ의 1/16 calibration 비용으로 cross-vocabulary AP 최상위권.** 논문 §Experiments의
Efficiency(Calibration cost) 절에 직접 대응하고, 이건 해석이 아니라 측정값이다.

또한 **논문 motivation 자체는 그대로 살아 있다.** fidelity 문서 §7.2에서 activation
observer를 min-max↔MSE로 바꾸면 **COCO와 LVIS가 6/6 seed 양방향으로 갈리는 것**을
확인했다 — 논문 §Reconstruction–Utility Misalignment("reconstruction-optimal scale ≠
semantic/task-optimal scale")에 대한 깨끗한 실증이다. **분석 기여는 견고하고, 흔들리는
것은 "제안 방법이 reconstruction 계열을 decision 지표에서 이긴다"는 주장뿐이다.**

---

## 6. 열린 선택지 (아직 결정 안 함)

1. **포지셔닝 전환** — 헤드라인을 "decision 지표 우위"에서 "(calibration 비용,
   cross-vocabulary 보존) Pareto 지점"으로. 데이터가 바로 받쳐주지만 기여의 성격이 바뀐다.
2. **§4 간극 메우기** — calibration vocabulary 밖을 대변하는 항을 실제로 설계.
   현재 `neighbor_loss`가 COCO-80에서 무력화된 게 핵심 문제이므로, 외부 텍스트
   뱅크 등으로 이웃 풀을 COCO-80 밖까지 확장하는 방향(H_eval/LVIS 누설 없이).
   **아직 시도된 적 없는 유일한 방향이다.**
3. **문제 재정의** — 다섯 번의 실패가 모두 "sparse 감독 + 저자유도"의 한계를
   가리키므로, 그 조합 자체를 바꾸는 설계(예: 감독을 vocabulary-agnostic dense로,
   자유도는 낮게 유지).

---

## 7. 재현

```bash
export YOLO_AUTOINSTALL=false     # 없으면 ultralytics가 numpy를 덮어씀
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python pipeline/run_comparison.py \
  --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
  --data configs/coco_local.yaml \
  --lvis-ann /data/taeho/lvis_datasets/.../lvis_v1_minival.json \
  --calib 256 --recon-iters-ada 2000 --deterministic \
  --conditions naive,adaround,qdrop,brecq,combined --seed <0..5> --device 0
```

환경 구축·소요 시간·주의사항은 `pipeline/BASELINE_STATUS.md` §2, §5.

**설계 반복 시 주의**: `--conditions` 목록이 RNG 스트림 위치를 정한다. 두 run을
비교하려면 반드시 동일하게 유지할 것(BASELINE_STATUS §5.2).
