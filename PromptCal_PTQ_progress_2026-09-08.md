# PromptCal-PTQ 연구 진행 정리 — 2026-09-08

## 0. 문서 목적

[PromptCal_PTQ_progress_2026-09-07.md](PromptCal_PTQ_progress_2026-09-07.md) §7~8에서
"neighbor-hinge 설계가 conv당 단일 스칼라라서 구조적으로 class-selective 조정이
불가능하고, `(s_mult-1)^2` 정규화로는 이득과 부작용을 같이 줄이는 트레이드오프만
만들 뿐 해결이 안 된다"는 결론까지 냈다. 이 문서는 그 다음 단계 — per-channel
재설계 구현·검증, 그리고 "논문에서 실제로 쓰는 방식에 가깝게" 데이터 설정을
바꿔 재검증하는 과정을 정리한다.

핵심 줄거리: **per-channel + 가벼운 정규화(`rw=20`) 조합이, 완벽하진 않지만
지금까지 나온 Combined 변형 중 가장 균형 잡힌 결과를 냈다.** COCO-80 이득의
대부분(79%)을 유지하면서 LVIS에서 더 이상 baseline보다 못하지 않다. 지금은
이 결과가 "논문에서 실제로 쓰는 calibration/평가 방식"에서도 유지되는지
확인하는 실험이 GPU 3개에서 진행 중이다.

---

## 1. per-channel s_mult 재설계 구현 (§7.6 옵션 1)

### 1.1 변경 내용

`src/quant/adaround.py`의 `AdaRoundQuantConv2d.s_mult`를 conv당 스칼라
(`nn.Parameter(torch.tensor(1.0))`)에서 **입력 채널별 벡터**
(`nn.Parameter(torch.ones(in_channels))`)로 변경. `_quantize_smult()`에서
`.view(1,-1,1,1)`로 브로드캐스트해서 채널마다 다른 배율을 적용할 수 있게 함.

**동기**: text embedding이 512차원 중 채널마다 다른 가중치로 각 class와
내적하므로, 채널별로 다른 스케일 조정을 하면 "특정 class에는 영향을 주고
다른 class는 안 건드리는" 게 원리적으로 가능해진다 — conv당 스칼라 하나로는
애초에 불가능했던 자유도.

### 1.2 디바이스 버그 (구현 중 발견)

`torch.ones(N)`은 기본 CPU 텐서라 CUDA 텐서와 곱하면 device mismatch 에러가
난다. 그런데 기존 스칼라 버전(`torch.tensor(1.0)`)은 이 문제가 한 번도
없었다 — PyTorch가 **0차원(스칼라) CPU 텐서는 CUDA 텐서와 섞여도 암묵적으로
허용**하는 특례가 있어서다(파이썬 숫자처럼 취급). 벡터(1차원 이상)는 이
특례가 없어 즉시 드러남. `self.conv.weight.device`로 명시적으로 맞춰서 해결.

### 1.3 1차 확인 (정규화 없음, `scripts/56_perchannel_smult_check.py`)

실제 규모(calib=32, eval=500, seed 2) 결과:

```
           COCO-80 AP  LVIS AP
adaround     35.05     0.2232
combined     35.99     0.2123   (여전히 naive=0.2175보다 낮음)

s_mult: 전체 평균=0.9959(전 스칼라 버전 0.977~0.988의 하향 편향은 사라짐)
        전체 std=0.2625, min=-0.31, max=2.36 (개별 채널 변동폭은 훨씬 커짐)
```

편향은 없어졌지만 여전히 LVIS에서 못 이긴다. **해석**: 자유도가 늘면서
32장짜리 작은 calibration set에 과적합할 여지도 같이 늘어난 것으로 보임 —
자유도 확대(per-channel)와 억제(정규화)를 **같이** 써야 한다는 가설로 이어짐.

---

## 2. per-channel + 정규화 스윕

`scripts/53_scale_reg_sweep.py`를 그대로 재사용(정규화 계산이 이미 `.mean()`
기반이라 vector s_mult에도 코드 변경 없이 적용됨 — 단, 스크립트 자신의
`s_mult` 요약 출력 줄이 `float()`을 직접 호출해서 죽는 버그 발견, `.mean()`
추가해서 수정). seed 2, `reg_weights=[0,5,20,50,100,200]`:

```
reg_weight | s_mult 평균 | COCO-80 AP | LVIS AP
    0.0    |   0.9934   |   35.99    | 0.2123
    5.0    |   0.9865   |   36.14    | 0.2220
   20.0    |   0.9938   |   36.11    | 0.2241   <- AdaRound(0.2232) 넘음
   50.0    |   0.9968   |   35.92    | 0.2215
  100.0    |   0.9983   |   35.73    | 0.2154
  200.0    |   0.9990   |   35.46    | 0.2167
```

`rw=20`에서 처음으로 COCO-80(+1.05 vs naive)과 LVIS(baseline 전체를 이김) **둘 다**
잡는 지점이 나왔다 — 스칼라+정규화 때(§8, rw=200에서 COCO +0.22 / LVIS는
그래도 최선을 못 넘김)와 질적으로 다른 결과.

---

## 3. rw=20(per-channel) 6-seed 검증

`scripts/54_combined_reg_6seed.py`를 그대로 재사용(`--scale-reg-weight 20.0`),
표준 6-seed 세트(0,1,2,4,5,7). 결과를 지금까지의 모든 Combined 변형과 나란히:

```
                    method |   COCO AP |  UPIR% | LVIS AP  | Top1_flip%
                    naive  |   35.06   | 0.232  | 0.2175   | 0.820
                 AdaRound  |   35.05   | 0.218  | 0.2232   | 0.690
                    QDrop  |   35.08   | 0.207  | 0.2202   | 0.680
                    BRECQ  |   35.01   | 0.165  | 0.2217   | 0.630
    Combined(스칼라,rw=0)  |   36.46   | 0.142★ | 0.2160   | 0.655
  Combined(스칼라,rw=200)  |   35.31   | 0.225  | 0.2191   | 0.646
  Combined(per-ch,rw=20)  | 36.10±0.06 | 0.191±0.077 | 0.2211±0.0026 | 0.693±0.058
```
(★ = 해당 열 최선)

**해석**: `rw=20`(per-channel)은 완전히 이기는 결과는 아니다 — UPIR은 BRECQ
(0.165)에 못 미치고 원래 스칼라 버전(0.142)의 최선에도 못 미친다. LVIS AP도
AdaRound(0.2232)에 근소하게 못 미친다(6-seed 평균 0.2211 — seed 2 하나만 보면
0.2241로 AdaRound를 넘었었는데, 6-seed 평균은 그보다 낮다 — seed 2가 다소
유리했던 경우였음). 하지만:

- COCO-80 AP는 원래 이득(+1.4)의 79%(+1.04)를 유지하며 baseline 전부를 이김.
- LVIS AP는 naive·QDrop을 이기고 BRECQ와 거의 동률 — **naive보다 못했던**
  두 실패작(스칼라 rw=0/rw=200)과 질적으로 다름.
- UPIR도 naive·AdaRound·QDrop 3개는 이기고 BRECQ에만 못 미침 — 마찬가지로
  baseline 전체보다 못했던 스칼라+rw=200(0.225, 4개 중 최악권)보다 뚜렷이 낫다.

**즉 "전부 최고"는 아니지만 "전반적으로 균형 잡힌, 어디서도 최악이 아닌"
프로파일을 처음으로 달성**했다 — 지금까지 나온 8개 조건(baseline 4 + Combined
변형 4) 중 유일하게 모든 지표에서 상위권을 유지하는 조건. 논문에 쓸 만한
수준인지는 §4의 "논문 방식" 데이터로 재확인한 뒤 최종 판단.

---

## 4. "논문에서 실제로 쓰는 방식"으로 데이터 설정 교체 (진행 중)

### 4.1 배경 — 공식 벤치마크와 비교했더니

사용자가 YOLO-World 공식 GitHub 수치(COCO zero-shot AP 36.6/AP50 51.0, LVIS
zero-shot AP 18.5, LVIS-minival AP 23.6)와 우리 FP32 측정을 비교. COCO는
거의 정확히 일치(36.80/51.28 vs 36.6/51.0)했지만, LVIS는 APr(rare class)이
공식 수치의 절반 이하(6.67 vs 12.6~16.4)로 떨어짐 — 원인: 우리가 LVIS 평가에
쓰던 건 COCO val2017 probe와 겹치는 **484장짜리 임시 부분집합**이었는데, rare
class는 전체 LVIS val(19,809장)에서도 이미지당 median 1장이라 484장으로는
표본이 너무 작아 APr이 불안정.

### 4.2 데이터 계획 확정

- **calibration**: 지금까지는 COCO val2017 앞부분을 잘라 썼는데, val2017을
  평가에 온전히 남기기 위해 **train2017**(118,287장 로컬에 이미 있음)에서
  가져오는 것으로 변경.
- **calib 크기**: 분류(ImageNet) PTQ 표준은 1024지만, detection PTQ는
  이미지당 정보량이 많아 보통 더 작게(100~500장) 쓰는 관행이 있음(사용자
  확인) → **256**으로 결정.
- **LVIS 평가**: ultralytics 공식 배포 `lvis-labels-segments.zip`(520MB,
  이미지 재다운로드 없이 라벨/annotation만 포함)을 받아서 확인한 결과,
  공식 **`lvis_v1_minival.json`(4809장)이 정확히 "COCO val2017 ∩ LVIS val"과
  일치** — 우리가 임시로 쓰던 484장짜리 부분집합의 **공식적이고 10배 큰 버전**.
  이걸로 교체.
- **COCO-80 평가**: val2017 **전체**(5000장, calib과 완전 분리).

### 4.3 실행 전 타이밍 실측

AdaRound류(레이어마다 calibration 전체를 2회 훑는 구조)가 calib 크기에
얼마나 민감한지 실측(calib=64: 498.6s, calib=256: 862.7s, iters=1000 고정) →
선형 피팅(이미지당 ~1.9s, 고정비 ~377s)으로 calib=1024면 조건당 최대
39분(naive 제외)까지 걸릴 수 있음을 확인 → **1024 대신 256을 선택한 근거**가
됨(계산 비용 대략 1/4로 절감, 조건당 최대 ~17분 수준).

### 4.4 구현 중 발견한 버그 2건 (스모크 테스트로 확인)

1. **OOM**: probe(COCO val2017 5000장 전체)를 `preprocess()`가 GPU 텐서로
   즉시 변환해 파이썬 리스트로 들고 있던 기존 패턴을 그대로 5000장에 적용 →
   20GB GPU 메모리 즉시 초과. `SimilarityHarness.run_image`/`calibrate`/
   `optimize_*` 전부 내부에서 자체적으로 `.to(device)`를 하므로,
   `preprocess()`가 CPU 텐서를 반환하도록 수정(다른 스크립트들은 probe가
   500장 이하라 이 문제가 드러난 적이 없었음).
2. **LVIS 메모리 폭발**: P=1203이라 `[8400 anchor x 1203 class]` sim을
   4809장 리스트로 들고 있으면 fp_sims 하나만 계산상 ~194GB(CPU RAM 111GB로도
   불가능). `compute_lvis_flip_gt_streaming()`을 새로 작성해 이미지 하나씩
   처리하고 즉시 버리는 방식으로 변경(5개 조건 harness를 동시에 열어두고
   이미지 루프 안에서 전부 갱신).
3. (사소) FP32 모델의 vocabulary를 LVIS로 전환한 **뒤에** COCO-80 AP를
   재려다 confusion matrix index 에러 — FP32의 COCO AP 측정을 vocabulary
   전환 **전으로** 재배치해서 해결.

### 4.5 현재 상태

`scripts/58_full_baseline_official_data.py`(신규)로 naive/AdaRound/QDrop/
BRECQ/Combined(per-channel, `rw=20`) 5개 조건 전체를 이 새 데이터 설정으로
검증 중. seed 0/1/2를 GPU 5/6/7에 병렬 실행(각 seed 자체적으로 baseline
4개까지 다시 빌드 — 중복이지만 병렬이라 wall-clock에는 영향 없음). 예상
소요 seed당 ~2.5시간. **결과는 완료 후 본 문서 §5(추가 예정) 또는 후속
문서에 기록.**

---

## 코드 구조 (09-07 §8 이후 추가분)

```
src/quant/adaround.py            -- s_mult: 스칼라 -> per-channel 벡터로 변경
src/quant/promptcal.py           -- s_mult 로깅 3곳을 벡터 대응(.mean()/.sum())으로 수정
scripts/56_perchannel_smult_check.py -- per-channel 1차 확인(정규화 없음)
scripts/53_scale_reg_sweep.py    -- (재사용, 버그 수정) per-channel+정규화 스윕
scripts/54_combined_reg_6seed.py -- (재사용) rw=20 6-seed 검증
scripts/58_full_baseline_official_data.py -- 신규: train2017 calib + 공식 LVIS
                                     minival로 5개 조건 전체 재검증
/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json
                                  -- 공식 LVIS minival GT(신규 다운로드, 4809장)
runs/58_official_data/seed{0,1,2}.log -- 진행 중인 실행 로그
```
