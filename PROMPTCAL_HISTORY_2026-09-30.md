# PromptCal-PTQ 연구 경위 — 처음부터 프로토콜 확정까지 (2026-08-18 ~ 10-02)

**대상 독자:** 이 연구를 이어받거나 논문을 쓰는 공저자.

**목적:** 각 단계에서 **무엇을 왜 했고, 결과가 어땠고, 그래서 왜 다음 단계로 넘어갔는지**를 한 줄기로 따라갈 수 있게 한다.

**출처 표기 방식**
- 수치와 판단은 각 단계의 원문서에 있다. 괄호 안에 claim 번호, `runs/` 번호, 커밋 해시를 적었다.
- 09-28 이전은 문서(MASTER_SUMMARY, 진행 기록, CLAIMS, BASELINE_FIDELITY, PAPER_DESIGN)와 git 로그를 읽고 재구성했다.
- 09-29~10-02는 직접 수행한 기록이다.

**현재 상태 요약 문서**
- 방법: [`PROMPTCAL_METHOD_LOWBIT_2026-09-29.md`](PROMPTCAL_METHOD_LOWBIT_2026-09-29.md)
- 결과: [`docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md`](docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md) (10-01 이후 결과는 방법 문서 §3.3~§3.4, §4.5, §9)
- 논문 개요: [`PAPER_OUTLINE_2026-10-01.md`](PAPER_OUTLINE_2026-10-01.md)
- 코드 구성과 실행: [`pipeline/README.md`](pipeline/README.md)

---

## 0. 한 장 요약

| # | 기간 | 단계 | 핵심 결과 | 다음으로 넘어간 이유 |
|---|---|---|---|---|
| 1 | 08-18 ~ 08-21 | 관찰과 baseline 구축 (국면 A~D) | AP는 멀쩡해도 경계·held-out 판정이 무너진다. 강한 PTQ로도 안 막힌다 | 판정을 보존하는 방법이 필요 |
| 2 | 09-02 ~ 09-06 | PromptCal 초기 설계 (국면 E) | rounding을 판정 손실로 학습하면 실패. 연속 activation scale(s_mult)이 첫 긍정 신호 → Combined | 평가 지표와 데이터 설정을 넓혀 검증 |
| 3 | 09-07 ~ 09-14 | 평가 확장과 일반화 | LVIS로 옮기면 Combined가 최악. per-channel·정규화로 재설계. 공식 데이터 설정 전환 | 설계를 공정성 관점에서 재점검 |
| 4 | 09-15 ~ 09-17 | 설계 감사 (claim 1~11) | per-tensor s_mult, identity-aware margin 채택. LVIS 평가 버그 수정 | baseline 쪽 공정성 문제 발견 |
| 5 | 09-17 ~ 09-24 | **baseline 재설계** (claim 12~17, 충실도 작업) | baseline에 LSQ와 공식 설정 복원 → BRECQ가 가장 강함. Combined는 판정 지표 0/6 | 격차를 좁히는 시도 |
| 6 | 09-24 ~ 09-27 | 격차 좁히기 시도 (claim 18·19 등 8건) | 전부 실패 | 원인 재검토 |
| 7 | 09-27 ~ 09-28 | 기반 교정과 저비트 첫 시도 (claim 20~23) | Stage 1을 BRECQ로 바꾸니 W8A8에서 LVIS 6/6 개선. 하지만 폭이 작고(+0.4점) W4A8에서는 모든 변형이 실패 | 저비트용 새 설계 필요 |
| 8 | 09-29 | 저비트 원인 규명과 양자화기 처방 | 손실 설계(vocab metric) 실패 → 손상이 텍스트 융합 conv에 집중됨을 규명 → 보호 + 이전(PM) | 6-seed 확정과 새로움 |
| 9 | 09-30 | 6-seed 확정과 구조 전용 설계 | PM이 4개 비트 설정 × 6 seed 전승. **텍스트 게이트 교환(G)** 발견 → 저비트에서 GPM이 최선 | 일반화, W4A8 확정, 배포 |
| 10 | 10-01 ~ 10-02 | **엄격한 프로토콜 확정과 일반화** | attention·contrastive 8bit에서도 결론 유지. m 불안정 → A16. A16이 드러낸 LSQ 버그 수정. M은 모델별 사전 검사로 채택. s·v2·m 모두 BRECQ보다 개선 | 최종 6 seed 일괄, YOLOE |

**흐름을 한 줄로**
> 판정 손상 관찰 → 판정 보존 학습(PromptCal) → baseline을 공정하게 만들자 학습의 이득이 사라짐 → 재구성 위에 학습을 얹어 W8A8에서 작은 이득 → 저비트에서는 학습이 전부 실패 → **손상의 구조적 위치를 규명하고 양자화기 수준에서 지키는 방향으로 전환** → open-vocab 구조 전용 설계(게이트 교환) → 엄격한 프로토콜(행렬 연산 ~100% 정수)로 확정하고 세 모델로 일반화.

---

## 1. 관찰과 baseline 구축 (08-18 ~ 08-21, 국면 A~D)

**연구 질문.** open-vocab 검출기(YOLO-World)를 양자화하면 region–prompt 판정이 어떻게 손상되고, 그것을 보존하는 PTQ가 가능한가.

**한 일과 결과**
- **국면 A (발판):** FP32를 재현했다(COCO mAP 37.8, 공식 37.7). 유사도 행렬을 캡처하는 측정 도구 `SimilarityHarness`를 검증했다.
- **국면 B (고정 프롬프트 손상):** naive W8A8에서 AP는 −0.5뿐이다. 하지만 경계 판정(margin < 0.1)의 flip이 60%다. **손상은 margin이 작은 곳에 집중된다.**
- **국면 C (프롬프트 축 손상):**
  - flip의 69%가 의미적 이웃 클래스로 간다.
  - 정답 프롬프트를 빼면(held-out) flip이 0.6%에서 7.2%로 는다.
  - 어휘를 LVIS-1203으로 키우면 flip이 0.5%에서 4.3%로 는다.
  - → **margin이 줄어드는 조건이면 어디서든 손상이 커진다.**
- **국면 D (강한 baseline):** AdaRound, QDrop, BRECQ를 재구현했다. 세 모델에서 비교해도 재구성 기반 PTQ는 held-out 판정 손상을 막지 못했다.
- **head 양자화 정도 비교:** cv3(region 임베딩 분기)의 activation만 A11로 두면 held-out flip이 −41%다(`docs/PROMPTCAL_V3_MIXED_PRECISION.md`). **"어디를 몇 bit로 지키느냐"가 효과를 낸 첫 사례**이고, 09-29 방향 전환의 먼 선행 근거가 된다.
- **첫 판정 손실(decision loss) 시도:** 실패했다(08-21, 09-02 "E2 failed").

**넘어간 이유.** 재구성만으로는 판정이 안 지켜지므로, 판정 자체를 보존하는 목적함수를 설계하기로 했다.

---

## 2. PromptCal 초기 설계 (09-02 ~ 09-06, 국면 E)

**한 일**
- **rounding(α)을 margin/decision 손실로 학습:** 실패했다.
  - 원인 중 하나는 최적화 중 soft/STE 플래그가 내내 꺼져 있던 버그였다(09-05 수정).
  - 이전 RankSafe/phase1 계열도 실패로 정리했다.
- **방향 C, 연속 activation scale 배율(s_mult):** weight rounding은 AdaRound로 고정하고, conv별 activation scale 배율만 학습한다. 첫 긍정 신호가 나왔다.
- **Combined 구성:** AdaRound + s_mult. 손실은 calibration 프롬프트 S의 top-k margin 보존 + 비대칭 이웃 hinge(텍스트 임베딩상 이웃 클래스가 부풀려지는 방향만 벌점).
- **baseline 버그 수정:** AdaRound와 BRECQ에 앞 층의 누적 오차가 반영되지 않던 버그를 고쳤다(09-06).

**결과.** AP 기준으로 6/6 seed가 AdaRound를 이겼다(09-06). 이후 6-seed 전체 비교에서 AP, 표준 Top1_flip, UPIR가 모든 baseline을 이기거나 동등했다. 다만 masked H_eval flip은 BRECQ가 더 좋았다.

**넘어간 이유.** 평가 지표가 좁고 데이터 설정이 비공식이어서, 일반성과 지표를 확장해 검증하기로 했다.

---

## 3. 평가 확장과 일반화 (09-07 ~ 09-14)

**한 일과 결과**
- **표준 지표 추가:** Top1_flip, GT_MRR/R@1, lost, UPIR, corrective rate.
- **LVIS 일반화:** COCO에서 학습한 Combined를 LVIS 어휘로 옮기면 **LVIS AP가 여섯 조건 중 최악**이었다.
  - s_mult를 1로 되돌리는 인과 ablation으로 원인이 s_mult임을 확인했다.
  - conv당 스칼라인 s_mult가 전역적으로 하향 편향되어, 보지 않은 클래스 전체에 영향을 줬다.
- **재설계 시도:**
  - (s_mult−1)² 정규화: 6-seed 실패.
  - **per-channel s_mult + 정규화:** "어디서도 최악이 아닌" 첫 프로파일(09-08).
- **공식 데이터 설정으로 전환(09-08):** calibration은 train2017 256장, LVIS 평가는 공식 minival 4809장.
- **버그 수정:**
  - H_eval이 neighbor-hinge 학습에 새던 누수(09-09)
  - 메모리 폭증(RSS, DataLoader fork)
  - probe 스트리밍
- **cal_weight 도입(09-14):** margin 손실을 H_cal까지 확장했다.
- **남은 문제:** Combined가 LVIS APr(드문 클래스)에서 다섯 조건 중 최악이었다(09-10).

**넘어간 이유.** 설계 요소마다 공정성과 타당성을 체계적으로 감사할 필요가 생겼다.

---

## 4. 설계 감사 (09-15 ~ 09-17, claim 1~11)

- **claim4:** Combined만 per-channel activation 자유도를 쓰는 건 비교 교란이다. 배포 시 표준 INT8 커널로 재현할 수도 없다. → **s_mult를 per-tensor(conv당 스칼라)로 바꿨다.**
- **claim5:** margin 손실이 클래스 정체성을 무시했다. → **identity-aware margin 채택**(09-16, 기본값 전환).
- claim1~3, 6~10: anchor 선정 기준, neighbor 후보 퇴화, 클래스별 AP 매핑 버그, vocabulary 전환 버그, 버전 고정 등을 수정했다.
- **claim11:** LVIS AP가 공개 수치의 절반이었다. 원인은 NMS multi_label 불일치였고, Fixed AP 프로토콜을 채택했다(09-17). 이 수정 뒤 FP32 LVIS AP가 0.126에서 0.259가 됐다.
- **설계 문서:** `docs/PROMPTCAL_CURRENT_MODEL_V2.md`가 이 시점의 기준 문서다.

**넘어간 이유.** 사용자가 baseline 쪽의 비대칭(claim12)을 발견했다. "우리 방법만 손잡이가 있는 것 아니냐"는 문제다.

---

## 5. baseline 재설계 (09-17 ~ 09-24, claim 12~17 + 충실도 작업)

**동기.** 비교가 공정하지 않으면 어떤 이득도 주장할 수 없다. 그래서 baseline을 **원 논문과 공식 코드에 최대한 맞추는 작업**을 했다(`PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md`).

| claim / 작업 | 발견 | 조치 |
|---|---|---|
| claim12 | baseline(QDrop/BRECQ)에는 activation scale을 학습하는 손잡이가 없다. 원 논문에는 LSQ가 있다 | LSQ를 넣었다. 09-21에 s_mult와 완전히 분리된 독립 구현 `LSQActQuant` + 공식 lr |
| claim13 | 재구성 손실 정규화가 공식보다 C_out배 작아서 α가 거의 안 움직였다. lr과 reg_weight도 공식값이 아니었다 | 공식 `lp_loss` 정규화, lr 1e-3, reg 0.01, warmup 20% |
| claim14 | Combined의 1단계(AdaRound)가 2단계 보호 영역 밖으로 손상을 흘린다 | **1단계를 제거**(stage1 = none, naive 기반). → 이것이 이후 실패 8건의 원인이 됨(claim20) |
| claim15 | 확정 표가 LSQ가 꺼진 baseline과 비교한 것이었다 | 공정 비교로 재측정 |
| claim16 | 남은 격차를 좁히는 두 시도(aux_mse, neighbor_of_cal) | 둘 다 실패 |
| claim17 | W8A8 손상의 거의 전부가 activation 양자화에서 온다(W8A32 −0.05, W32A8 −3.29 COCO AP; 당시 min-max observer) | "activation 쪽을 고치는 방법"이라는 근거 |
| 09-21 | 공식 BRECQ/QDrop은 MSE activation observer를 쓴다 | `--act-observer mse` 기본값. 이것으로 naive W8A8 손상 대부분이 사라졌다(COCO AP 33.5 → 36.6) |
| 09-23 (`9125d1d`) | weight 양자화가 채널별 대칭 max-abs였다(공식은 비대칭 MSE). torch-seed가 모든 seed에서 고정돼 있었다 | 채널별 비대칭 MSE, 독립 LSQ, seed 버그 수정. 환경 재구축(`.venv`) |
| 09-24 (`9da7d99`) | CLIP vision tower의 conv가 양자화에 섞여 있었다(모델 크기 12.13 → 실제 9.88 MiB). AdaRound iters가 1000으로 교란이었다 | CLIP 제외, AdaRound iters 2000. **확정 baseline `runs/97` 6-seed** |
| 09-24 (`1162c98`) | Combined만 재실행이 비결정적이었다 | `--deterministic` |

**결과(`runs/97`, W8A8, head 제외).** BRECQ가 판정 지표에서 가장 강한 baseline이 됐다(Heval_flip 3.71%, lost 166). 당시 Combined(naive 기반)는 **판정 지표에서 AdaRound·QDrop·BRECQ에 0/6 전패**였다.

**넘어간 이유.** 공정한 baseline 앞에서 우리 방법이 지고 있으므로, 격차를 좁히는 방법을 찾아야 했다.

---

## 6. 격차 좁히기 시도 (09-24 ~ 09-27, claim 18·19 등, 전부 실패)

| 시도 | 결과 |
|---|---|
| rounding α를 s_mult와 함께 margin으로 학습 (claim18-a) | W8A8 10/10 지표 악화 |
| activation 범위(클리핑) 완화 (claim18-b, range_blend) | 10개 중 7개 악화 |
| 논문 §4.3 utility 항(l_thresh, l_box) (claim19) | 절반이 죽은 코드(인덱싱 버그)였다. 고쳐도 LVIS APr 트레이드오프만 있고 순이득 없음 |
| region_dir(임베딩 방향 항), 단측 margin, 랜덤 샘플링, per-group anchor, local recon | 이득 없음 / 악화 |

모두 8건이고, 전부 실패했다.

**넘어간 이유.** 사용자의 질문이 전환점이 됐다: "논문은 **재구성만으로는 부족하니 ranking 보존을 추가**하자는 건데, 그게 잘 살려졌나?"

---

## 7. 기반 교정과 저비트 첫 시도 (09-27 ~ 09-28, claim 20~23)

**claim20 (기반이 잘못돼 있었다).**
- claim14에서 1단계를 뺀 뒤로 확정 설계가 **"naive + ranking"**이었다. 논문이 주장하는 "재구성 + ranking"이 아니었다. 8건의 실패는 모두 이 기반 위에서 나왔다.
- **Stage 1 = BRECQ**로 바꿨다(`runs/112`, W8A8, head 제외, 6 seed). LVIS_flip −18%, LVIS_lost −26%, LVIS AP +0.2점으로 **LVIS 세 지표가 6/6 개선**됐다. Heval_flip은 1/6(악화)이었다.
- "BRECQ 위에 얹으면 새로움이 약하다"는 우려에 대해: PTQ 문헌에서 표준적인 패턴이다(QDrop = BRECQ + drop, PD-Quant = BRECQ + 예측 차이).

**claim23 (head 포함, fully quantized, 6 seed, `runs/115`·`121`).**
- BRECQ 대비 LVIS AP +0.41점(6/6), COCO AP +0.29(6/6)였다.
- 하지만 Heval_flip과 lost는 0/6(악화)이었다.
- head를 포함하면 BRECQ도 naive보다 LVIS AP가 낮았다.
- AdaRound min-max의 붕괴는 observer 문제로 판정했다(`runs/116`·`117`).

**GPU 번호 불일치 수정(`1d71eb5`).** `--device N`이 nvidia-smi 번호와 달랐다. `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 고정했다.

**판단: 이득이 너무 작다** (LVIS AP +0.4점, COCO AP +0.3점). → 격차가 더 큰 **저비트(W4A8)**로 가기로 했다.
- **s_mult만 (`runs/113`):** BRECQ에 0/3 악화.
- **공동 최적화, α + 블록 재구성 (claim21, `runs/114`):** 붕괴(COCO AP 33.5 → 18.5).
- **α bias (conv당 스칼라 rounding 보정, 자유도 52개; claim22, `runs/120`):** 붕괴(33.5 → 27.8).
- → 교훈: **W4에서 rounding을 판정 손실로 건드리면 자유도와 무관하게 무너진다.**

**다음 설계 (`5ce7871`, 09-28 요약 문서).**
- 판정 손실 대신 **재구성 손실의 metric을 "임의 어휘에서의 유사도 오차"로 바꾸는** vocabulary-metric 재구성(brecq_vm)을 설계했다.
- 저비트 표준 관례(첫/마지막 레이어 8bit)를 추가했다.

---

## 8. 저비트 원인 규명과 양자화기 처방 (09-29)

**8.1 요약 문서 계획 실행 (E0~E4, `runs/123`~`127`)**
- **E0:** 첫/마지막 레이어를 8bit로 둬도 W4A4와 **W8A4**가 모두 붕괴했다. → 원인은 per-tensor 4bit activation 자체다.
- **E1:** lam_mean=1일 때 C의 유효 차원이 1.7/512다. 평균 방향 하나가 지배한다.
- **E2~E4:** brecq_vm은 W4A8에서 **모든 변형**(lam 0/1, identity, anchor 가중, mix)이 LVIS AP를 −0.1~−0.65점 악화시켰다. W8A8에서는 identity만 +0.2점이었다.
- → **손실(감독)을 바꾸는 계열은 저비트에서 종료.**

**8.2 원인 규명 (`runs/128`, `139`)**
- 학습 없는 W4 민감도 진단: **12.cv2**(C2fAttn concat → 1×1 conv) 하나가 naive 손상의 60~83%를 차지한다.
- 12.cv2 분해:
  - 입력 네 분기 중 **텍스트 attention 분기**만 activation(최대 20)과 weight(최대 3.3)가 동시에 크다.
  - 이 outlier가 공유 scale을 오염시켜 출력이 파괴된다(상대 오차 0.83).
- 가설 검증:
  - **H3(attention 분기):** 지지됨. attention 열만 8bit로 두면 LVIS flip이 79%에서 6.9%로 회복된다.
  - **H1(어휘 밀도):** 보조 요인.
  - **H2(오차 방향), H5(guide):** 기각.
- **모델 간 재현(`runs/137`):** s-v1, s-v2, m-v1 모두 C2fAttn concat conv와 1.conv가 최상위다.

**8.3 양자화기 처방 (`runs/129`~`136`)**
- **블록·conv·채널 단위 W8, 입력 채널 scale 이전(독립 → 배포 가능한 공유 제약):**
  - 채널 계보 추적기: 측정 기반, 배포형 FP 등가성 5.5e-6.
- **평가 데이터 누수 발견과 수정:** 처음 conv 선택을 평가 이미지와 LVIS로 했다. 누수 없는 선택(train2017, COCO 어휘)으로 바꿔도 효과가 같았다.
- **대조군:**
  - 같은 크기의 **무작위 보호는 효과 없음.** 위치 선택이 결정적이다.
  - 선택 기준(임베딩, 출력 MSE, COCO flip, 로컬 오차)은 대부분 같은 conv를 고른다. → **기준은 기여가 아니다.**
- **PromptCal을 새 양자화기 위에 얹기(`runs/140`):** 이득 없음. held-out 지표가 2/2 악화 → 음성 결과.
- **W8A8에서 공유 제약 이전:** LVIS_flip −30~33%. 비트 비용 0.
- **A4 누적 복구 곡선(`runs/142`):** conv의 76%를 A8로 되돌려도 붕괴한다. activation 비트 스윕(`runs/143`)에서는 A5부터 풀린다. → W4A6, W4A5를 새 설정으로 추가.
- **P0 (방법을 규칙으로 고정, `9ba8655`):**
  - 보호 예산 1.5%, 선택 기준은 COCO flip(사용자 결정), 자동 선택 `--protect-budget`.
  - 이렇게 **PM(보호 + 공유 제약 이전 + BRECQ)**이 확정됐다.

---

## 9. 6-seed 확정과 구조 전용 설계 (09-30)

**9.1 실험 인프라 (`95bb9ce`)**
- **조건별 RNG 복원:** 조건을 묶어도 순서의 영향이 없다. bit-identical로 검증했다.
- **조건 접미사**(`+P/+M/+R/+H/+G`)와 GPU 작업자 대기열.

**9.2 주 결과 6-seed (`runs/147_main`)**

| 설정 | PM vs BRECQ | 비고 |
|---|---|---|
| W8A8 | LVIS AP +0.32점, LVIS_flip −33% (6/6) | COCO AP FP 수준 |
| W4A8 | +1.36점, −34% (6/6) | 크기 +1.3% |
| W4A6 | +1.41점, −25% (6/6) | 이전 단독은 0/6 악화 |
| W4A5 | +2.45점, −26% (6/6) | 이전 단독은 2/6 |

**baseline**
- QDrop은 LVIS_flip 0/6이다.
- Combined(PromptCal)는 LVIS AP 0/6이다.
- AdaRound는 W4A8에서 붕괴한다. MSE observer로도 붕괴해서, 09-28의 "observer 탓" 판정은 W4에는 해당하지 않는다.
- **QDrop 위 PM도 6/6**(plug-in).

**9.3 새로움 탐색**
- **선행 연구:** 직접 경쟁 연구는 **QATMA**(arXiv 2603.05964)다. QAT이고, YOLO-World M/L/X, W4A4·W4A5를 다루며, PTQ는 W4A4에서 완전 실패로 보고했다. 층별 원인 분석은 없다.
- **A안(이전 고려 보호, `runs/148`):** 보호 목록의 마지막 한 자리만 바뀐다. 대신 **"이전은 취약성을 12.cv2에서 1.conv로 옮긴다"**는 메커니즘을 얻었다. → 나중에.
- **방향 2(분기 인식 concat):** A4 붕괴는 해결되지 않았다 → 기각. W4A4는 한계로 명시.
- **방향 1: 텍스트 게이트 교환(G, `e3b342f`)**
  - C2fAttn의 `cv2(cat(y0,y1,y2, aw⊙p))`를 `W_main·cat(y0,y1,y2) + Σ_h aw_h⊙(W_h·p_h)`로 재배치한다.
  - FP 등가이고 비트 비용은 0이다.
  - naive에서 12.cv2 손상 약 90%를 제거한다.
  - **BRECQ 위, 6 seed 결과:**
    - G 단독: W4A6·W4A5에서 모든 지표 6/6 개선.
    - **GPM(G + 남는 conv 보호 + 이전, 크기 +0.3%):** PM(+1.3%)보다 held-out 판정 지표가 6/6 더 좋다. W4A5에서는 COCO AP도 6/6이다.
    - W4A8에서는 PM과 GPM의 차이가 Holm 보정 후 유의하지 않다(10-01 짝 Δ 분석).
  - 게이트 교환은 **이전을 쓸 수 있게 만든다.** A6·A5에서 G → GM이 개선된다.

---

## 10. 엄격한 프로토콜 확정과 일반화 (10-01 ~ 10-02)

**출발점.** 기본 프로토콜은 conv만 양자화하고 attention Linear/matmul, head contrastive matmul, DFL을 FP로 두었다. "그래도 되는가"라는 질문(QATMA는 attention을 8bit로 양자화)에서 시작했다.

**10-01: attention·contrastive 8bit (runs/153)**
- `--attn-quant`를 구현했다. 실행 중 값은 per-tensor A8, 텍스트 상수는 클래스별 8bit, 실행 중 입력을 받는 Linear는 W8A8로 양자화한다.
- 처음에는 양자화를 `no_grad` 안에서 해서 BRECQ 재구성의 gradient가 끊겼다. STE로 고쳤다.
- 연산량을 다시 세어 보니, LVIS 어휘에서는 contrastive matmul이 전체 행렬 연산의 약 24%(5.2 GMAC)였다. 처음 "FP는 0.75%"라고 한 답은 이것을 빠뜨린 것이었다.
- **6 seed (W4A8·W4A5): 짝 차이가 그대로였다.** GPM − BRECQ는 W4A8 +1.09 → +1.21, W4A5 +2.62 → +2.62(seed 0 제외)다. W4A5 GPM − PM LVIS AP가 +0.39로 유의해졌다.
- **W4A5 seed 0에서 BRECQ만 붕괴했다**(COCO 18.7). 진단 네 가지 결과, attention 양자화가 직접 만든 손상도, 12.cv2도 아니었다. 블록 15 이후 재구성 전반이 나쁜 해로 간 것으로, 원인 conv는 특정하지 못했다.

**10-01: m 불안정과 A16 (runs/151·152)**
- 일반화 2 seed: v2는 강하게 재현됐고(+5~6점), m은 PM·GPM이 BRECQ보다 나빴다.
- m naive W8A8가 seed에 따라 40.4 / 24.3으로 갈렸다. 이분 탐색으로 head `cv3.0` 마지막 1×1의 입력 범위(MSE observer 무작위 부분추출)를 원인으로 찾았다.
- 그 입력만 A16으로 두면 40.7로 고쳐졌다. **m에만 쓰면 평가를 보고 프로토콜을 바꾸는 셈이라, 모든 모델에 같게 적용하기로 했다.**

**10-01 밤 ~ 10-02 새벽: 확정 프로토콜 점검과 LSQ 버그 (runs/154·155)**
- 보호 진단도 새 프로토콜로 다시 했다. 보호 목록은 세 모델 모두 그대로였다.
- **처음 결과에서도 m의 PM·GPM은 나빴다.** M을 끄는 사전 검사를 시험하던 중 이상한 값이 나왔다. BRECQ+M 모델은 직접 forward하면 임베딩 오차가 약 1.0이었다.
- 층별로 쪼개 보니 head cv3 마지막 conv(A16) 한 레벨이 FP와 반대 방향의 출력을 냈다. weight만 양자화하면 정상이었다.
- **원인:** BRECQ가 16bit activation의 step까지 LSQ(Adam)로 학습해서 step이 발산했다. `LSQ_MAX_BITS = 16`으로 고치고, 새 프로토콜 결과를 전부 다시 돌렸다(이전 로그는 `old_lsqbug/`).
- **고친 뒤에도 m에서 M은 해로웠다**(PM −1.5 ~ −3.5점). 그러니 M 자체가 m을 해친다는 것이 확정됐다.

**10-02: M 채택 규칙 (runs/155~157)**
- calibration **밖** train2017 200장과 COCO 어휘로 M 켬/끔 모델의 flip을 비교하는 검사(`mcheck_flip`)를 만들었다.
- calibration 이미지를 쓰면 BRECQ가 이미 맞춘 이미지라 M의 해가 가려질 수 있어서, 처음부터 calibration 밖 이미지를 썼다.
- **검증:** m은 2~7배로 늘어 "끄기", s는 1.06배 이하로 "켜기"였다. 실제 평가와 모두 맞았다.
- v2 W8A8에서 M이 LVIS flip만 나빠지는 경우는 COCO 어휘 검사로 잡히지 않았다. 한계로 기록했다.
- 임베딩·logit 오차는 기준이 될 수 없었다. m에서도 M이 이 오차들을 줄였다. 평균 오차가 아니라 판정을 직접 봐야 했다.
- **비용:** `+A`(BRECQ 두 번)는 빌드 시간이 2.2~2.4배다. 짧은 BRECQ(200 iter) 검사는 절약이 작고 m W4A5에서 틀렸다. 그래서 **모델당 한 번 검사하고 그 결정을 모든 설정에 쓰는 방식**으로 정했다.

**확정 프로토콜 점검 결과 (2 seed, LVIS AP, BRECQ 대비)**
- s: W4A8 PM +1.3 / +1.2점, W4A5 GPM +2.3점(seed 1). seed 0은 BRECQ가 다시 붕괴했다.
- v2: W4A8 PM +6.5 / +6.4점, W4A5 PM +7.3 / +7.9점.
- m: G(= GP) W4A8 +0.8 / +1.5점, W4A5 +3.2 / +2.3점.
- **세 모델 모두 BRECQ보다 좋아졌다.** 일반화를 주장할 근거가 생겼다.

**정리 (10-02):** 확정 프로토콜 인자는 `pipeline/scripts/protocol.sh`에 있다. 기각된 vocab-metric 스크립트는 `pipeline/legacy/`로 옮겼다.

**10-03: 표 1 확정 (runs/158).** 확정 프로토콜로 s 주 결과를 6 seed로 다시 냈다. W4에서는 결론이 그대로였다(W4A8 PM +1.15, W4A6 GPM +1.66, W4A5 GPM +2.53점, seed 0 제외). W8A8의 M 효과는 +0.20점으로 이전보다 작아졌다. W4A5 seed 0에서는 BRECQ와 QDrop이 함께 무너졌다. 같은 날 표 2(W4A5 ablation: G +1.37, P +1.04, M −0.75점, M은 G·P 위에서만 이득)와 표 3(무작위 보호 +0.21 vs 진단 기반 +1.05점, QDrop 위 plug-in W4A5 +5.4점)도 6 seed로 확정했다. 같은 날 방법을 GPM 하나로 고정하고(비트별 선택 안 함), W8A8 GPM과 QDrop + GPM을 추가했다(runs/159). W8A8 GPM은 flip −24%로 M 단독보다 확실히 나았고, QDrop + GPM은 BRECQ + GPM과 같아졌다. 결과의 기준 문서는 `docs/PROMPTCAL_RESULTS_2026-10-03.md`다.

---

## 11. 돌아보면: 반복해서 나온 교훈

1. **공정성과 충실도가 결론을 바꾼다.** baseline에 LSQ, 공식 손실, 공식 양자화 격자를 복원하자 Combined의 이득이 사라졌다(5단계). 설계 기반이 바뀌면(claim20) 결론도 바뀌었다. 평가 누수(09-29)와 GPU 번호 불일치(09-28)도 같은 종류의 함정이었다.
2. **calibration 어휘에 맞추는 학습은 held-out 순위를 해친다.** 반복해서 확인됐다: margin/decision 손실, s_mult, rounding 공동 학습, vocab metric, PromptCal을 새 양자화기 위에 얹기.
3. **정밀도 배분과 구조는 통한다.** cv3 A11(08월) → 보호(PM) → 게이트 교환. "어디를 어떻게 양자화하느냐"로 푼 시도는 대부분 효과가 있었다.
4. **원인을 층 단위로 규명하자 설계가 나왔다.** 12.cv2 분기 분해(텍스트 분기 outlier + 공유 scale)에서 게이트 교환이 곧바로 도출됐다.
5. **짝비교와 seed 수.** 1~2 seed 결론이 6 seed에서 뒤집힌 사례가 여러 번 있었다(09-05 "2승 1패", claim15 1-seed). 그래서 조건별 RNG 복원과 6-seed 짝비교를 기본으로 삼았다.
6. **평가를 보고 모델별로 규칙을 바꾸지 않는다.** A16과 M 채택 모두 같은 원칙으로 정했다. 프로토콜은 모든 모델에 같게 적용하고, 모델별 결정은 누수 없는 검사로만 내린다.
7. **새 설정은 기존 코드의 숨은 가정을 깬다.** A16은 "activation step은 LSQ로 학습해도 된다"는 가정을 깼다. 이상한 숫자가 나오면 결론부터 내지 말고 층 단위로 쪼개 확인한다.
8. **판정 기준이 평균 오차보다 정직하다.** M의 해는 임베딩 오차가 줄어드는 가운데 판정을 뒤집는 형태였다.

---

## 12. 현재 위치와 남은 일 (10-02)

- **방법:** ⓪ 게이트 교환 + ① 누수 없는 보호 + ② 공유 제약 이전(모델별 사전 검사로 채택) + ③ BRECQ(또는 QDrop).
  - s: W8A8 M, W4A8 PM, W4A6·W4A5 GPM. v2: PM / GPM. m: GP.
- **확정 프로토콜:** head 포함, 첫·마지막 8bit, attention·contrastive 8bit, head 마지막 conv 입력 A16.
- **남은 일**
  1. ~~확정 프로토콜로 s 표 1·2·3~~ 완료(10-03)
  2. v2·m 3 seed 확장, YOLOE-v8s 일반화
  3. QATMA 조건(YOLO-World-L, 첫·마지막 FP), SmoothQuant/AWQ·Reg-PTQ 비교
  4. 실제 엣지 배포 검증, 논문 그림
- **논문 논지:** "open-vocab 검출기 PTQ에서 무너지는 것은 held-out 어휘의 판정이고, 그 손상은 텍스트가 융합되는 구조적 지점과 초반 backbone에 몰려 있다. 그 지점을 구조적으로 재배치(게이트 교환)하고 소수 conv만 지키면, 학습 없는 PTQ로 행렬 연산을 거의 전부 정수화한 W4A5까지 판정을 지킨다. 세 YOLO-World 모델에서 재현된다. calibration 어휘에 맞춘 학습은 오히려 held-out 판정을 해친다."
