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
