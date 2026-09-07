# PromptCal-PTQ 연구 진행 정리 — 2026-09-07

## 0. 문서 목적

[PromptCal_PTQ_progress_2026-09-06.md](PromptCal_PTQ_progress_2026-09-06.md) §18에서
Phase 1(RankSafe) 표준 평가지표(Top1_flip, GT MRR/R@1/lost/gained, UPIR)를
구현하고 6-seed 실험을 실행하기로 했다 — 실제 실행은 09-06 밤~09-07 새벽으로
넘어가 완료됐다. 이 문서는 그 실행 과정에서 있었던 운영상 실수와 수정,
결과의 시각화, 그리고 지금까지의 전체 상태를 정리한 두 문서
(`MASTER_SUMMARY.md` 갱신, `PROMPTCAL_HOW_IT_WORKS.md` 신규)를 기록한다.

핵심 줄거리: **6-seed 실험 결과, Combined는 AP뿐 아니라 표준 Top1_flip과
real-GT 기반 UPIR에서도 QDrop/BRECQ를 포함한 모든 baseline을 이기거나
동등했다. 우리가 만든 masked H_eval flip에서만 여전히 BRECQ가 낫다 — 이건
버그가 아니라 다른 배포 시나리오를 재는 지표이기 때문이라는 게 09-06 §15의
가설대로 6-seed 규모로 확증됐다.**

---

## 1. 백그라운드 실행 실수: `nohup & disown`은 harness가 못 쫓아간다

`scripts/45_baseline_compare.py`를 6-seed(0,1,2,4,5,7) 전체에 대해 순차
실행하는 셸 스크립트를 작성해서 `nohup bash run_all_seeds.sh & disown` 형태로
띄우고, Bash 툴을 `run_in_background: true`로 호출했다.

**실수.** `disown`으로 진짜 백그라운드 프로세스를 만들면, Bash 툴이 실제로
추적하는 건 "그 프로세스를 띄우고 바로 리턴하는 런처 명령" 자체다. 그래서
런처가 끝나자마자(수 초 만에) `status: completed` 알림이 왔는데, 이건 6-seed
실험이 끝났다는 뜻이 전혀 아니었다 — 실제 실험은 seed 0의 naive 빌드조차
안 끝난 시점이었다. `pgrep -af 45_baseline_compare.py`로 실제 프로세스가 아직
살아있는 걸 확인하고 나서야 착오를 알아챔.

**수정.** `Monitor` 툴로 driver 로그 파일(`runs/45_full/driver.log`)을
polling하는 새 워처를 걸었다 — seed 시작/완료 문자열이 찍힐 때마다 이벤트로
받고, `Traceback`/`CUDA out of memory`/`Killed` 같은 실패 시그니처와
`ALL_SEEDS_DONE` 완료 시그니처를 모두 감시하도록 구성(`persistent: true`,
60초 간격). 이후 seed 1~7 완료가 정확한 타이밍에 알림으로 들어옴.

**교훈(재사용 시 참고).** 여러 시간짜리 job을 진짜 백그라운드(`nohup`+`disown`)로
던질 땐, harness의 `run_in_background`가 추적하는 건 "job을 던지는 명령"이지
"job 자체"가 아니다. job 자체의 완료를 알림받으려면 별도로 로그 파일을
polling하는 `Monitor`(또는 `until` 루프의 `Bash run_in_background`)를 걸어야
한다.

---

## 2. 6-seed 실험 결과 (요약 — 전체 표는 09-06 §18 참고)

```
              AP(overall)      Top1_flip(표준)   GT_R@1            UPIR%
naive         35.06 (const)    0.820 (const)     0.8843 (const)    0.232 ± 0.130
adaround      35.05 (const)    0.690 (const)     0.8851 (const)    0.218 ± 0.104
qdrop         35.08 (const)    0.680 (const)     0.8866 (const)    0.207 ± 0.156
brecq         35.01 (const)    0.630 (const)     0.8866 (const)    0.165 ± 0.090
combined      36.46 ± 0.06     0.655 ± 0.061     0.8874 ± 0.0014   0.142 ± 0.059
```

- AP: Combined 6/6 완승(모든 baseline 대비).
- Top1_flip(표준): Combined ≈ BRECQ(최선 baseline), 6개 중 3개 seed는 BRECQ보다도 낮음.
- GT_R@1/MRR: Combined ≈ BRECQ(사실상 동률, 평균은 근소 우위).
- UPIR: Combined가 5/6 seed에서 BRECQ보다 낮고 평균 최저 — 논문 핵심 동기와
  가장 직결되는 지표에서 명확히 최선.
- (표에 없음) masked H_eval_flip: Combined가 6/6 최악권 — BRECQ가 매 seed 더 낮음.
  이건 실제 배포 조건(Top1_flip/AP)과는 다른, "H_eval을 아예 안 쓰는 사용자"
  반사실적 시나리오를 재는 지표라서 나오는 차이로 해석됨(09-06 §15 메커니즘).

naive/AdaRound/QDrop/BRECQ 값이 전부 `(const)`인 이유: 이 4개 방법은 `pidx(S)`나
`H_eval`을 전혀 안 쓰고 calib/probe 이미지도 seed와 무관하게 고정이라
6-seed 전부에서 완전히 동일한 값을 낸다 — 진짜 seed 분산은 Combined와
H_eval-의존 지표에만 있다(09-06 §18.3에 명시).

---

## 3. 결과 시각화 (Artifact)

사용자가 "결과 표 나한테도 보여줘"라고 요청해서, 위 표 두 개(AP/flip, GT/비용)와
§2의 판정을 정리한 리포트 페이지를 Artifact로 발행함 — "Combined PTQ 6-Seed
검증"(요약 카드 4개 + 데이터 테이블 2개 + 판정 6항목 + 방법론 주의사항 콜아웃
구조). URL은 대화 내역 참고(세션마다 바뀌지 않는 한 재발행 시 같은 링크 유지).

---

## 4. 문서 정리: 상태 요약 갱신 + 원리 문서 신규 작성

사용자가 "지금까지의 상황을 정리하는 문서"(이 문서류)와 "모델 작동방식/원리를
자세히 설명하는 문서"를 요청함. 두 가지를 만듦:

### 4.1 `MASTER_SUMMARY.md` 갱신

기존 파일은 09-03 시점(국면 D 착수 전)에 멈춰 있어서, D(baseline)/E(Combined)/
F(오늘까지의 확장 평가) 국면이 전부 "미착수"로 잘못 표시돼 있었다. 로드맵
표를 A~H(H=실배포 성능, 별도 보류)까지 갱신하고, 국면 D·E·F 섹션을 실제
6-seed 결과표로 다시 씀. 국면 D 절에는 AdaRound/BRECQ의 누적 오차 미반영
버그와 수정도 함께 기록.

### 4.2 `PROMPTCAL_HOW_IT_WORKS.md` 신규

기존 `PROMPTCAL_METHOD_SPEC.md`는 세 번의 **실패한** 시도(alpha rounding만으로
margin을 맞추려던 접근)만 기록한 문서라 지금 방법을 전혀 설명하지 못한다는
걸 확인함. 그래서 처음부터 새로 썼다 — 다루는 내용:

1. 문제 정의(reconstruction ≠ decision preservation)
2. `SimilarityHarness`(cv4 훅으로 region-prompt 유사도 행렬 캡처) 원리
3. fake-quant 스킴(`QuantConv2d`/`ActObserver`, per-channel weight / per-tensor activation)
4. AdaRound/QDrop/BRECQ 각각의 알고리즘과 누적 오차 반영 수정
5. Combined의 실제 메커니즘: 왜 alpha-rounding 접근이 세 번 실패했는지부터,
   연속값 `s_mult`로 옮긴 이유, `margin_loss`, asymmetric neighbor hinge
   (collateral shift 억제) 수식까지 코드 포함 설명
6. 평가지표 6종의 정의와 "왜 이걸 재는가"(특히 Top1_flip vs H_eval_flip이
   서로 다른 배포 시나리오라는 점)
7. 재현/확장 시 주의할 함정 4가지(누적오차 버그, cuDNN 비결정성, fake-quant
   caveat, GPU 인덱스 함정)

---

## 5. 남은 일

- `논문.txt`(git 미추적) 초안에 §2의 "3단 구조" 서술(메인 클레임 승리 / GT
  기준 BRECQ와 동률 / masked flip 한계 인정)을 실제로 반영.
- 실제 하드웨어(TFLite/TensorRT) latency/메모리 측정은 여전히 별도 후속
  과제로 보류.
- Boundary inversion, NDCG@10 등 09_phase1 §1.3의 나머지 secondary 지표는
  아직 미구현(우선순위 낮음 — 현재 6개 지표로 메인 클레임과 한계를 이미
  충분히 뒷받침).

---

## 코드/문서 구조 (09-06 문서 이후 추가분)

```text
MASTER_SUMMARY.md                 -- 갱신: 국면 A~H 로드맵, D/E/F 실제 결과로 교체
PROMPTCAL_HOW_IT_WORKS.md         -- 신규: 현재 파이프라인 전체 기술 설명(원리 문서)
PromptCal_PTQ_progress_2026-09-07.md -- 본 문서

runs/45_full/run_all_seeds.sh     -- 6-seed 순차 실행 드라이버(nohup, PCI_BUS_ID 고정)
runs/45_full/seed{0,1,2,4,5,7}.log -- 6-seed 원본 stdout(결과 로그와 동일 내용)
results_v1/diag/45_metrics_seed{0,1,2,4,5,7}_full.txt -- 위 로그를 관례 경로로 복사
```

관련 커밋: `511f438`(Top1_flip/GT 지표 구현 + 6-seed 결과), `379fdca`
(MASTER_SUMMARY 갱신 + PROMPTCAL_HOW_IT_WORKS 신규).

---

## 6. v1(최초 PromptCal: margin+decision+reg, alpha rounding)을 새 지표로 재검증

사용자 질문: "v1도 flip이 안 좋아지는 걸 확인했었는데, 평가지표가 바뀌었으니
v1도 새 지표로 다시 확인해보고 싶다." 여기서 v1은 `scripts/20_promptcal_minimal.py`
/ `src/quant/promptcal.py`의 `optimize_promptcal` — AdaRound의 alpha(rounding)를
`margin_loss + decision_weight·decision(CE) + reg_weight·reg`로 최적화하는,
Combined(연속 s_mult + asymmetric neighbor hinge)로 피벗하기 **이전**의 최초
시도다(`PROMPTCAL_METHOD_SPEC.md` 참고). 당시엔 masked held-out flip만 쟀고
AdaRound보다 못했다(때에 따라 9.24%→14.80%까지 악화)는 이유로 폐기됐었다.

그런데 09-06/09-07에서 Combined도 masked H_eval_flip에서는 나쁘지만 표준
Top1_flip/AP/GT 지표에서는 최선이라는 게 밝혀졌으니(§18, 본 문서 §2), "v1도
당시 masked flip만으로 성급하게 버려진 건 아닐까?"라는 합리적 의문이 생김.
`scripts/46_v1_metrics_check.py`를 새로 작성해서(naive/v1만 새로 빌드, AdaRound/
Combined는 이미 확보된 6-seed 값 재사용) 같은 6-seed·같은 probe로 확인.

### 6.1 결과 (6-seed 집계)

```
condition |    AP      | H_evalAP   | Heval_flip% | Top1_flip% | GT_MRR        | GT_R@1        | lost      | UPIR%
naive     | 35.06(c)   | 35.63±3.97 |  9.79±1.06  |  0.820(c)  | 0.9267(c)     | 0.8843(c)     | 41.0(c)   | 0.232±0.130
v1        | 35.56±0.08 | 36.06±3.85 | 10.00±1.07  |  0.920±0.036| 0.9269±0.0005 | 0.8850±0.0009 | 42.0±2.4  | 0.220±0.113
(calib 시간: naive 3.7s, v1 129s)
```

v1 vs naive 승패(6-seed):
- AP: v1이 **6/6 승**(+0.5, 이전엔 한 번도 측정된 적 없던 지표 — v1도 naive보다 AP는 낫다).
- Top1_flip(표준): v1이 **0/6 승** — 6개 seed 전부 naive보다 나쁨(naive 0.82% vs v1 평균 0.92%).
- H_eval_flip(masked)/lost/UPIR: 승패가 반반으로 갈려 사실상 **naive와 구분 안 됨**(noise 수준).
- GT_MRR/R@1: 근소하게 나음(+0.0002/+0.0007) — 사실상 무의미한 차이.

### 6.2 판정 — Combined와 다르게, v1은 새 지표로도 구제되지 않는다

Combined는 "masked H_eval_flip에서만 나쁘고 나머지(AP·표준 flip·GT·UPIR)는
전부 최선"이라는 패턴이었다. v1은 그 반대에 가깝다: **AP는 naive보다 낫지만,
정작 이 연구의 핵심 관심사인 "결정 일치도"를 표준 지표(masking 없는 실제
배포 조건)로 재면 v1은 naive(아무것도 안 함)보다도 못하다.** GT 기반 지표·
masked flip도 개선은커녕 naive와 다를 바 없다. 즉:

> v1의 문제는 "잘못된 지표로 잰 것"이 아니라 **실제로 결정 보존에 실패했다는 것**
> 자체였다 — masked H_eval_flip이 우연히 그 실패를 잡아낸 것뿐, 어떤 지표로
> 봐도 v1은 개선이라 부를 수 없다. Combined로의 피벗(alpha rounding → 연속
> s_mult, decision CE → margin+neighbor hinge)이 정당했다는 게 새 지표
> 세트로도 재확인됨.

### 6.3 코드/결과

```
scripts/46_v1_metrics_check.py      -- v1(naive/promptcal alpha) 전용 새 지표 재검증
runs/46_v1/seed{0,1,2,4,5,7}.log    -- 원본 stdout
results_v1/diag/46_v1_metrics_seed{0,1,2,4,5,7}_full.txt -- 관례 경로 복사본
```

---

## 7. 국면 I: LVIS 일반성 검증과 메커니즘 규명

### 7.1 배경 — MASTER_SUMMARY §8 국면 I 착수

§6까지의 검증(6-seed, COCO-80)이 끝난 뒤, 사용자가 "우리 model이 v1(YOLO-World
v1 체크포인트)/LVIS 확장에 대해서도 검증했었나?"를 물어서 확인한 결과:
`MASTER_SUMMARY.md`가 09-03 시점 이후 갱신되며 원래 있던 "LVIS AP 측정" 항목이
누락돼 있었고, D 이후 스크립트가 전부 `yolov8s-world.pt`(v1) 기본값을 쓰는데
환경 설명은 여전히 v2로 남아있었던 것도 발견 — 둘 다 `MASTER_SUMMARY.md`에서
수정(국면 I 신설).

### 7.2 실험 1 — zero-shot 이식 (47/49번, COCO-80 학습 -> LVIS 배포)

COCO-80으로 학습한 Combined를 재학습 없이 LVIS-1203 vocabulary로 바꿔 배포.
- 47번(pseudo-label flip/margin, 진짜 GT 없음): LVIS에서 Combined(5.33%)가
  naive(5.65%)보다는 낫지만 AdaRound(5.17%)/QDrop(5.02%)/BRECQ(4.88%) 전부보다 못함.
- 사용자 지적: "COCO의 dog가 LVIS의 Yorkshire Terrier로 갈리는 것도 flip으로
  잡히는데, 이게 무해한 세부 재배치인지 진짜 손상인지 flip만으로는 못 가른다.
  AP가 있어야 안다" -> 진짜 LVIS GT 필요.
- **LVIS 진짜 annotation 확보(09-07)**: 다른 사용자(`seunghyuk`) 소유 파일을
  쓰지 않고, 공식 https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip 을
  새로 받아 `/data/taeho/lvis_datasets/annotations/`에 배치. `lvis-api`
  설치(`pip install lvis`, numpy 2.x 호환을 위해 `np.float = float` 패치 필요
  -- 2020년 배포된 패키지). ultralytics `lvis.yaml`의 `names[i]`가 실제 LVIS
  `category_id = i+1`과 정확히 대응함을 확인(별도 remap 불필요). 우리 probe
  이미지(COCO val2017)와 LVIS val split(19809장, COCO val2017과 다른 split)의
  교집합은 ~97%.
- **49번(실제 GT AP, seed 2, 484장)**: Combined(AP 0.2109)가 **6개 조건 중 최악**
  — naive(0.2175)보다도 낮음. AdaRound(0.2232)가 최선. **결론: flip 증가가
  무해한 재배치가 아니라 진짜 검출 품질 저하와 같이 간다 — 한계가 진짜다.**

### 7.3 메커니즘 규명 — s_mult는 왜 LVIS에서 해가 되는가 (사용자 주도 추론)

사용자가 제기한 가설과 검증 과정(중요도 순 아님, 대화 순서):

1. "AdaRound 혼자(5.17%)가 Combined(5.33%)보다 나은 걸 보면, Combined의 이득은
   AdaRound 성분이 떠받치고 neighbor-scale 성분은 LVIS에서 오히려 해가 되는
   것 아닌가?" — 데이터가 정확히 이 패턴을 보임.
2. "neighbor_loss가 원래 COCO 이웃의 절대 유사도만 억제하려던 건데, s_mult가
   전역 스칼라라서 억제가 새어나가는 것 아닌가?" — 검증: 45번(COCO-80 native,
   verbose=True) 로그에서 Combined의 `s_mult 평균`이 학습 내내 **1.0 미만
   (0.97~0.99, 최종 0.977)** 으로 수렴함을 확인. asymmetric hinge가 "경쟁자
   점수가 FP32보다 오르면만" 벌점을 주므로, 전역 스칼라를 살짝 줄이는 게
   가장 값싼 해법이 되어 이런 하향 편향이 생김.
3. "근데 그건 LVIS를 안 쓴 실험이니까 LVIS 세부 클래스 전환을 억제하는지는
   모르는 것 아닌가?" — 정확한 지적. s_mult<1은 순수 COCO-80 학습 결과이지
   LVIS에 대한 직접 증거가 아님. 인과관계 확정에는 별도 확인(예: 이식 시
   s_mult를 강제로 1로 되돌리는 ablation)이 필요하다고 정리.
4. "neighbor 유사도는 원래 전체 클래스 기준 아니었나? COCO->LVIS 확장 땐 전체
   클래스를 모르는데 neighbor_loss가 어떻게 쓰이나?" — **결정적 확인**:
   `text_neighbor_order(txt_feats)`의 `txt_feats`는 학습 시점(항상 COCO-80)의
   text embedding이라 `[80,512]`뿐이고, 실제 로그도 "neighbor 35개(k=5)"로
   80-class 공간 안에서만 계산됐음을 보여줌. **neighbor_loss는 LVIS 배포
   시점에는 아예 계산되지 않는다** — 학습 후 남는 건 얼어붙은 s_mult 스칼라
   뿐이고, 그 스칼라는 태생적으로 "35개만 조준"이 불가능해서(conv당 숫자
   하나뿐) 전역적 억제로 새어나갔고, 그 전역 편향이 존재도 몰랐던 LVIS
   1203개 클래스에도 무차별 적용된다.

**결론(재설계 방향의 근거)**: 문제는 "학습 vocabulary가 좁아서"가 아니라
**"s_mult가 conv당 스칼라 하나뿐이라 애초에 class-selective한 조정이 구조적으로
불가능하다"**는 파라미터화 자체의 한계로 보인다. margin_loss·neighbor_loss
둘 다 이 하나의 자유도를 놓고 경쟁하고, 어느 쪽이 이기든 "전체를 같이 늘리거나
줄이는" 것 말고는 할 수 있는 게 없다 -- pre-sigmoid 공간에서 단일 곱셈
스칼라는 모든 class 쌍의 margin을 동일 비율로만 조정하기 때문(개별 조정 불가).

### 7.4 대조 실험 — native calibration이 이 문제를 피할 수 있는가 (진행 중)

같은 메커니즘이 "재학습 없는 이식"이라는 시나리오 특유의 문제인지, 애초에
LVIS처럼 촘촘한 vocabulary에 처음부터(native) 맞춰 학습해도 반복되는
구조적 한계인지 가르기 위해 두 실험을 GPU 2개에서 동시 실행(둘 다
`verbose=True`로 s_mult 로그 확보):

- `scripts/50_lvis_native.py`(GPU 7): LVIS-1203을 처음부터 native vocabulary로
  놓고 S/H_cal/H_eval을 LVIS frequent bucket(405개, rare/common은 val 이미지가
  너무 적어 calibration 신호가 없음)에서 다시 분할, calib=128, 전체 지표
  (AP+S/H_eval subset+Top1_flip+H_eval_flip+GT_MRR/R@1/lost/gained/UPIR+비용).
- `scripts/51_lvis_transplant_full.py`(GPU 6): 47/49와 같은 이식 시나리오를
  하나로 합치고 진짜 GT 기반 GT_MRR/R@1/lost/gained까지 추가(단, H_eval_flip/
  UPIR/subset AP는 구조적으로 이식 시나리오에 적용 불가 -- 스크립트 docstring에
  근거 명시).

**해석 기준**: 두 실험 모두에서 Combined가 최악이면 -> neighbor-hinge
설계(§7.3의 단일 스칼라 파라미터화) 자체의 구조적 한계 -> 재설계 필요.
이식에서만 나쁘고 native에서는 괜찮으면 -> "재학습 없는 이식" 시나리오의
한계일 뿐, 설계 원리는 유효 -> 스코프 명시로 충분.

(50/51 결과는 완료 후 본 문서 또는 후속 문서에 추가 예정.)

### 7.5 인과관계 확정 — s_mult ablation (`scripts/52_smult_ablation.py`)

§7.3의 가설("s_mult 드리프트가 범인")을 직접 확인하는 가장 싼 방법: 이미 학습된
Combined 모델에서 **s_mult만 추론 직전에 1.0으로 강제로 되돌리고**(rounding은
그대로) LVIS AP를 다시 잰다. `_quantize_smult()`의 `scale = a_obs.scale *
s_mult`이므로 s_mult=1이면 수학적으로 순수 AdaRound `quantize()`와 동일해진다.

**결과(seed 2, 484장, 실제 LVIS GT)**:

```
                   AP       AP50
FP32             0.2259   0.3152
adaround         0.2232   0.3135
combined         0.2161   0.3071
combined(s=1)    0.2232   0.3135   <- adaround와 소수점까지 완전 일치
```

**s_mult=1로 되돌리는 순간 AdaRound와 AP가 정확히 같아짐 -> Combined가 LVIS에서
AdaRound보다 나쁜 원인의 100%가 학습된 s_mult(평균 0.988, 그러나 개별 conv
최저 0.31까지 극단적으로 떨어짐 -- min=0.3129, max=1.1932)이고 rounding
쪽에는 문제가 없다는 게 가설이 아니라 확정됨.**

### 7.6 재설계 방향 (§7.5 확정 이후 정리)

원인이 "s_mult가 conv당 스칼라 하나뿐이라 class-selective 조정이 구조적으로
불가능"으로 확정됐으므로, 재설계는 이 파라미터화를 직접 겨냥해야 한다:

1. **conv당 스칼라 -> per-channel 벡터.** `[out_channels]` 크기로 확장하면
   text embedding이 채널마다 다른 가중치를 갖는다는 사실을 이용해 class별로
   다른 영향을 줄 자유도가 생김. 파라미터 증가로 인한 overfit 위험은 `(s-1)^2`
   정규화로 완화.
2. **적용 범위 축소.** 지금은 양자화된 conv 71개 전부에 답다(`list_adaround_
   convs` 전체) -- cv4 바로 앞 region-feature conv 1~2개로 좁혀서 얕은
   backbone에서 시작된 편향이 여러 층을 거치며 증폭되는 것을 방지, 1번의
   파라미터 수도 관리 가능한 수준으로 유지.
3. **(가장 저렴, 우선 시도) 정규화 추가.** 구조 변경 없이 `loss += reg_w *
   (s_mult - 1)^2`만 추가해도 "값싸게 전역적으로 줄이는" 지름길이 막혀서
   최적화가 더 국소적인 해법을 찾도록 유도될 수 있음.

추천 순서: 3(정규화, 구조 변경 없음) 먼저 시도 -> 부족하면 1+2(per-channel +
범위 축소). 50/51(native vs transplant 대조)이 끝나면 "이 문제가 이식
시나리오 특유인지, native 학습에서도 반복되는 구조적 한계인지"까지 확인한
뒤 재설계 착수 여부/우선순위를 최종 결정.

### 7.7 대조 실험 결과 — native에서도 재현됨 (재설계 확정)

50(LVIS-native)이 완료되어 51(이식)과 나란히 비교 가능해짐. **s_mult가 세
시나리오 전부에서 1.0 미만으로 수렴**:

| 시나리오 | s_mult 최종 평균 |
|---|---|
| COCO-80 native (45번) | 0.977 |
| COCO→LVIS 이식 (51번) | 0.988 |
| LVIS native (50번) | 0.971 |

LVIS-native AP(seed 2, 489장):

```
           overall AP   S_AP    H_eval_AP   UPIR
naive        0.2159    0.2140    0.2756    0.81%
adaround     0.2162    0.2096    0.2800    1.15%
qdrop        0.2172    0.2086    0.2802    1.08%
brecq        0.2130    0.2067    0.2765    1.22%
combined     0.2149    0.2145    0.2773    2.16%
```

Combined는 S_AP(자신이 직접 학습한 40개 LVIS class)에서 유일하게 1등이라
margin_loss 자체는 작동하지만, **overall AP는 naive보다도 낮고, 논문의 핵심
동기 지표인 UPIR은 6개 중 최악**(2.16% vs 나머지 0.8~1.2%)이다. 이식(51번)
때처럼 "6개 중 완전 최악"까지는 아니지만(brecq보다는 나음), naive를 못
이긴다는 본질은 동일.

**최종 결론**: 이 문제는 "재학습 없는 이식" 시나리오에 국한된 게 아니라
neighbor-hinge 설계(conv당 단일 스칼라, §7.5에서 인과관계 확정) 자체가
vocabulary와 무관하게 구조적으로 갖는 한계다. **재설계 착수를 확정**하고,
§7.6에서 정리한 우선순위(1. `(s_mult-1)^2` 정규화 먼저 시도 -> 2. 부족하면
per-channel 벡터화 + 적용 범위를 cv4 직전 layer로 축소)대로 진행한다.

## 코드 구조 (§6 이후 추가분)

```
scripts/47_lvis_generality.py       -- zero-shot 이식, pseudo-label flip/margin
scripts/48_w4a4_probe.py            -- W4A4 스모크 테스트(완전 붕괴, AP=0 확인 후 폐기)
scripts/49_lvis_ap.py               -- zero-shot 이식, 실제 LVIS GT AP(lvis-api)
scripts/50_lvis_native.py           -- LVIS-native 재학습, 전체 지표
scripts/51_lvis_transplant_full.py  -- zero-shot 이식, 전체 적용 가능 지표(AP+flip+GT)
scripts/52_smult_ablation.py        -- s_mult=1 강제 ablation, 인과관계 확정
scripts/53_scale_reg_sweep.py       -- (s_mult-1)^2 정규화 강도 스윕(1-seed)
scripts/54_combined_reg_6seed.py    -- 정규화판 Combined 6-seed 전체 검증
scripts/55_calib_size_sweep.py      -- calib 크기 스윕(정규화 없는 원본, 대안 가설 검증)
/data/taeho/lvis_datasets/annotations/lvis_v1_val.json -- 공식 LVIS v1 val GT(신규 다운로드)
runs/{47_lvis,48_w4a4,49_lvis_ap,50_lvis_native,51_lvis_transplant,52_smult_ablation,54_combined_reg_6seed,55_calib_size_sweep*}/ -- 각 실행 로그
```

---

## 8. 정규화 접근의 6-seed 검증 — 부족하다는 결론

### 8.1 대안 가설 기각 — calib 크기는 원인이 아님

`scripts/55_calib_size_sweep.py`(정규화 없는 원본 Combined, calib 32/64/128/256,
seed 2)로 "s_mult 드리프트가 단순히 calibration 데이터 부족 때문 아닐까"를
확인. 결과: calib을 8배(32→256) 늘려도 s_mult 평균이 1.0에 가까워지지 않고
오히려 더 멀어짐(0.988→0.958~0.964, 뚜렷한 추세 없이 등락). LVIS AP도
calib=128에서 반짝 좋았다가(0.2297) calib=256에서 다시 낮아져서(0.2211)
일관된 개선 추세가 아니라 노이즈로 판단됨. **calibration 부족은 원인이
아니고, §7.5~7.7의 구조적 진단(conv당 단일 스칼라)이 재확인됨.**

### 8.2 (s_mult-1)^2 정규화 6-seed 검증 — 트레이드오프이지 해법이 아님

`scripts/54_combined_reg_6seed.py`(`scale_reg_weight=200`, 1-seed 스윕에서
가장 유망했던 값)을 COCO-80 6-seed 표준 세트로 전체 지표 재검증:

```
COCO-80 (6-seed)          AP      UPIR%   GT_MRR   lost   Top1_flip%
naive                    35.06    0.232   0.9267   41.0   0.820
AdaRound                 35.05    0.218   0.9272   35.0   0.690
QDrop                    35.08    0.207   0.9274   35.0   0.680
BRECQ                    35.01    0.165   0.9276   30.0   0.630
Combined(rw=0, 원조합)    36.46    0.142*  0.9280*  31.2   0.655
Combined(rw=200)         35.31    0.225   0.9266   36.8   0.646
(* = 5개 중 최선)

LVIS(6-seed)              AP        Top1_flip%   lost
naive                    0.2175     5.68         113
AdaRound                 0.2232*    5.19         116
QDrop                    0.2202     5.03         113
BRECQ                    0.2217     4.90         120
Combined(rw=200)         0.2191     5.10         121.3(6개 중 최악)
```

**결론: 정규화는 문제를 "해결"한 게 아니라 트레이드오프를 이동시켰을 뿐이다.**
LVIS AP는 naive를 근소하게 넘지만(+0.0016) AdaRound/QDrop/BRECQ에는 여전히
못 미치고 LVIS lost는 오히려 6개 중 최악이다. 그 대가로 COCO-80에서 Combined의
**가장 강력한 원래 결과였던 UPIR(0.142%, 5개 중 최선)이 0.225%로 뛰어
baseline 수준으로 후퇴**했고 GT_MRR/lost도 더 이상 최선이 아니다. 즉
"논문의 핵심 주장(UPIR 최선)을 포기하고 LVIS에서 그저 그런 성적을 받는"
교환이 되어, 논문에 실을 만한 결과가 아니라는 게 6-seed 규모로 확정됨.

**원인**: §7.6에서 예견한 대로, `(s_mult-1)^2` 정규화는 conv당 단일 스칼라의
근본 제약(class-selective 조정 불가)을 없애지 못하고 그 스칼라가 움직일 수
있는 "폭"만 줄인다 -- 이득과 부작용이 같은 자유도를 공유하는 한, 정규화는
둘을 동시에 줄이는 다이얼일 뿐 둘을 분리하는 스위치가 될 수 없다.

**다음**: per-channel 재설계(§7.6의 옵션 1+2: s_mult를 conv당 스칼라에서
`[out_channels]` 벡터로, 적용 범위를 cv4 직전 layer로 축소) 착수.
