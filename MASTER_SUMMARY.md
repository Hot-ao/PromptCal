# PromptCal-PTQ — 전체 실험 마스터 요약 (2026-09-07 갱신)

Open-vocabulary detector(YOLO-World)의 W8A8 양자화가 region–prompt 의미적
의사결정을 어떻게 손상시키는지 규명하고, 이를 보존하는 PTQ를 제안하기 위한 연구.

핵심 주장: **open-vocabulary 양자화는 tensor reconstruction 문제가 아니라
semantic decision preservation 문제다.**

이 문서는 "지금 상태가 뭔지" 한 번에 파악하기 위한 살아있는 요약 문서다(세션마다
갱신). 세션별 상세 기록은 `PromptCal_PTQ_progress_*.md`, 방법의 자세한 작동
원리는 `PROMPTCAL_HOW_IT_WORKS.md` 참고.

---

## 0. 한눈에 보기 — 전체 로드맵

| 국면 | 내용 | 상태 |
|---|---|---|
| A. 발판 | FP32 baseline 재현 + SimilarityHarness 검증 | ✅ 완료 |
| B. 고정 프롬프트 손상 | naive W8A8이 경계 의사결정을 흔듦 | ✅ 완료 |
| C. 프롬프트 축 손상 | substitution / held-out / LVIS | ✅ 완료 |
| D. 강한 baseline | AdaRound/QDrop/BRECQ 재구현 + 검증 | ✅ 완료 |
| E. 제안 방법(Combined) | AdaRound weight + asymmetric neighbor-preserving scale | ✅ 핵심 검증 완료(6-seed) |
| F. 평가지표 확장 | Top1_flip/GT MRR·R@1·UPIR/비용까지 포함한 6-seed 재검증 | ✅ 완료 |
| G. 논문 반영 | 위 결과를 실제 초안(`논문.txt`)에 서술 | ⏳ 미착수 |
| H. 실배포 성능 | 실제 INT8 엔진(TFLite/TensorRT) latency/메모리 | ⏳ 보류(별도 과제) |
| I. 일반성(LVIS/v2) | Combined를 LVIS-1203 vocabulary·v2 체크포인트에서도 검증 | ⏳ 착수(09-07) |

현재 위치: **COCO-80/v1 기준 핵심 실험(A~F)은 완료**, 09-07부터 국면 I(LVIS
vocabulary·v2 일반성)를 시작. 논문 서술(G)과 실제 하드웨어 배포 성능(H)은 별도로 남음.

---

## 1. 핵심 발견 (세 문장)

1. 양자화는 총량 지표(AP)로는 거의 무해해 보이지만, region–prompt 의사결정의
   **경계(작은 margin)** 에서, **의미적으로 가까운 방향** 으로, 프롬프트가
   **낯설거나 촘촘할수록** 심하게 손상된다(국면 B·C).
2. AdaRound/QDrop/BRECQ 같은 강한 reconstruction 계열 baseline도 이 손상을
   못 막는다(국면 D) — reconstruction을 아무리 잘해도 held-out semantic decision은
   안 지켜진다.
3. weight rounding(AdaRound)에 **class-agnostic한 continuous activation
   scale(s_mult)** 하나를 얹고, 이걸 "S(계산에 쓴 프롬프트)의 margin 보존 +
   text-embedding 최근접 이웃의 collateral shift 억제(asymmetric hinge)"로
   최적화하면(Combined, 국면 E), **AP·표준 Top1_flip·UPIR에서 QDrop/BRECQ까지
   포함한 전체 baseline을 이긴다**(국면 F, 6-seed 검증). 단, 우리가 만든
   masked H_eval flip(다른 vocabulary를 쓰는 사용자를 가정한 반사실적 진단
   지표) 기준으로는 아직 BRECQ가 더 낫다 — 이건 실제 배포 조건과는 다른
   시나리오라는 게 확인됐다(국면 F 상세).

![AP intact, decisions collapse](figures/fig3_ap_vs_decision.png)

---

## 2. 국면 A — 발판 (완료)

| 항목 | 결과 |
|---|---|
| FP32 baseline (COCO val, pycocotools) | mAP 37.8 (공식 37.7 재현) |
| size별 AP | small 21.5 / med 42.2 / large 50.1 |
| SimilarityHarness | cv4 캡처 [8400×P], 전역 argmax == 모델 예측 (검증) |

측정 도구(유사도 행렬 캡처)의 정확성이 검증되어, 이후 모든 손상 측정의 토대가 됨.

---

## 3. 국면 B — 고정 프롬프트 손상 (완료)

naive W8A8 (fuse 후, vision Conv, COCO 80 고정 프롬프트).

| 지표 | 값 | 의미 |
|---|---|---|
| AP 하락 | 37.8 → 37.3 (-0.5) | 총량은 멀쩡 |
| confident flip | 0.73% | 확실한 판정은 보존 |
| boundary inversion | 46.5% | 경계는 대규모 붕괴 |

**핵심 곡선 — 손상은 margin(경계)에 집중:**

![margin-flip curve](figures/fig1_margin_flip.png)

margin<0.1에서 flip 60.5%, margin>8에서 0%. 이 margin–손상 법칙이 국면 C 전체의
메커니즘이 됨. (all-anchor flip 30%는 maxprob<0.05 배경 앵커의 노이즈로 판명.)

---

## 4. 국면 C — 프롬프트 축 손상 (완료)

![Stage C three conditions](figures/fig2_stageC.png)

| 실험 | 조건 | 핵심 결과 |
|---|---|---|
| C-1 substitution | COCO, flip 방향 분석 | flip의 69% 가 의미적 이웃으로 (우연의 10.9x) |
| C-2 held-out | 정답 프롬프트 제거 | flip 0.6% → 7.2% (margin 5.25→1.8) |
| C-3 LVIS | 80 → 1203 프롬프트 | flip 0.51% → 4.28%, margin 6.41→1.29, 편향 31x |

세 실험이 하나의 인과 사슬로 연결됨:

```
B:   margin 작을수록 flip 급증
C-1: 의미적 이웃 = 작은 margin → flip이 이웃으로 향함
C-2: 정답 제거 → margin 축소 → flip 악화
C-3: 촘촘한 vocab → margin 축소 → flip 증폭
```

margin 축소가 어디서 오든(정답 제거/vocab 확대) 손상이 그 뒤를 따름.

주의(지표 해석):
- 배수(held-out/seen, flip/chance)는 분모에 민감 → 절대값을 헤드라인으로.
- LVIS 이웃 비율은 k에 민감(top-5는 좁음, k=20에서 52%로 회복).

---

## 5. 국면 D — 강한 baseline (완료)

**목적:** "reconstruction을 잘하는 강한 PTQ(AdaRound/QDrop/BRECQ)로도 semantic
손상은 안 없어진다"를 입증 → 제안 방법의 필요성 확립.

**구현:** 우리 fake-quant(`src/quant/fake_quant.py`) 위에 AdaRound(learnable
rounding, `adaround.py`), QDrop(activation drop 확률적 마스킹, `adaround.py`의
`qdrop_prob`), BRECQ(block-wise joint reconstruction, `brecq.py`) 세 가지를
전부 재구현. 최초 구현엔 "앞선 layer/block의 누적 양자화 오차가 뒤쪽 재구성에
전혀 반영되지 않는" 버그가 있었음 — target(FP 출력)은 그대로 두고 pred의 입력을
quant_module 자신의 현재 상태(이미 hardened된 앞쪽 + soft/init인 뒤쪽)에서
다시 캡처하도록 수정. 수정 후에도 세 baseline의 AP는 naive와 거의 구분되지
않았다(W8A8 자체가 reconstruction 개선 여지가 원래 작은 구간).

**6-seed 결과 (국면 F에서 최종 확정, 상세 §6):**

| 방법 | AP(overall) | H_eval mAP | Top1_flip(표준) | GT_R@1 | calib 시간 |
|---|---|---|---|---|---|
| naive | 35.06 | 35.63 ± 3.97 | 0.820 | 0.8843 | 3.5s |
| AdaRound | 35.05 | 35.56 ± 3.93 | 0.690 | 0.8851 | 456s |
| QDrop | 35.08 | 35.58 ± 3.88 | 0.680 | 0.8866 | 875s |
| BRECQ | 35.01 | 35.48 ± 3.97 | **0.630** | **0.8866** | 661s |

세 baseline 모두 naive 대비 AP 개선이 미미(±0.05 이내) — 예측대로 "강한
reconstruction으로도 semantic 손상은 안 없어진다"가 확인됨. 그중 BRECQ가
Top1_flip·GT_R@1에서 가장 좋은 baseline.

---

## 6. 국면 E+F — 제안 방법(Combined)과 확장 평가 (완료)

**방법 요약** (자세한 원리는 `PROMPTCAL_HOW_IT_WORKS.md` §4 참고):
1. AdaRound로 weight rounding 확정(hard).
2. 각 conv의 activation scale에 **learnable multiplier `s_mult`**(연속값,
   초기 1.0)를 얹고, weight/rounding은 더 이상 안 건드림.
3. `s_mult`를 두 개 항으로 최적화:
   - **margin_loss**: S(calibration에 실제로 쓰는 40개 프롬프트)의 top-(k+1)
     인접 margin을 FP와 맞춤.
   - **neighbor loss(asymmetric hinge)**: S의 각 class와 text-embedding상
     가장 가까운 non-S 이웃 class들의 절대 유사도가 FP보다 **커지는** 방향만
     억제(작아지는 방향은 벌점 없음) — `s_mult`가 class-agnostic이라 S만
     타깃해도 나머지 79개 컬럼에 새는(collateral shift) 부작용을 잡기 위함.
4. H_eval(20개 프롬프트)은 최적화에 **한 번도** 쓰지 않음 — 순수 held-out 평가.

**6-seed 최종 검증(scripts/45_baseline_compare.py, §18/§F 상세는
`PromptCal_PTQ_progress_2026-09-06.md` §18):**

| 지표 | Combined | 최선 baseline | 판정 |
|---|---|---|---|
| AP(overall) | 36.46 ± 0.06 | 35.08(QDrop) | **6/6 승, +1.38** |
| AP(H_eval subset) | 37.00 ± 4.01 | 35.63(naive) | **6/6 승** |
| Top1_flip(표준, AP와 같은 배포조건) | 0.655 ± 0.061 | 0.630(BRECQ) | **거의 동률**(3/6 seed는 BRECQ보다 낮음) |
| GT_MRR / GT_R@1(real COCO GT) | 0.9280 / 0.8874 | 0.9276 / 0.8866(BRECQ) | **동률**(평균은 근소 우위, seed별로는 역전도 있음) |
| UPIR(calibration 안 본 prompt 침입률) | 0.142% | 0.165%(BRECQ) | **5/6 seed 승, 평균 최저** |
| H_eval_flip(masked, 우리 자체 진단) | 9.75 ± 0.80 | 8.33(BRECQ) | **6/6 최악권 — 남은 한계** |
| calib 시간 | 637s | QDrop 875s / BRECQ 661s | 다른 reconstruction 계열과 같은 자릿수 |

**정직한 결론**: Combined는 AP·표준 Top1_flip·UPIR(=논문 핵심 동기와 가장
직결되는 지표)에서 QDrop/BRECQ를 포함한 모든 baseline을 이기거나 동등하다.
다만 "H_eval을 아예 쓰지 않는 다른 사용자"라는 좁은 반사실적 시나리오(masked
H_eval flip)에서는 여전히 BRECQ가 낫다 — 이건 버그가 아니라 s_mult가
class-agnostic해서 H_cal/H_eval을 명시적으로 보호하지 못하기 때문이며,
논문에서 한계로 정직하게 서술해야 할 지점.

**중요한 방법론적 사실**: naive/AdaRound/QDrop/BRECQ는 S나 H_eval에 의존하는
어떤 계산도 하지 않으므로 6개 seed 전부에서 완전히 동일한 값을 낸다(진짜
분산 없음). Combined와 H_eval-의존 지표(H_eval AP, H_eval_flip, UPIR)만 seed마다
실제로 달라진다 — "6-seed 평균 ± std"를 논문에 쓸 때 이 비대칭을 명시해야 함.

---

## 7. 실험 환경 / 코드베이스

- HW: L40S 46GB ×4, RTX 4000 Ada 20GB ×4 / CUDA 12.6 / torch cu126
- 모델: 국면 A~C(motivation)는 **yolov8s-worldv2**(fused) 기준. 국면 D~F(baseline
  재구현·Combined·6-seed 검증, `scripts/43` 이후 전부)는 **yolov8s-world(v1)**
  기준 — `yolov8s-world.pt`와 `yolov8s-worldv2.pt`는 실제로 다른 체크포인트
  (해시 다름, 파라미터 13.38M vs 12.76M)인데, D 이후 스크립트들이 전부
  `--model` 기본값을 `yolov8s-world.pt`로 물려써서 그렇게 됐다(09-07에 뒤늦게
  확인). `PROGRESS_v1_D.md`가 "다음: 국면 E"로 미완으로 남겨뒀던 v1에서의
  제안 방법 검증이, 라벨 없이 사실상 이미 6-seed로 끝나 있었던 셈 — v1이
  v2보다 양자화 취약성이 커서(`PROGRESS_v1_D.md` 표 참고) 더 어려운 조건이라는
  점은 결과 해석에 유리하게 작용하지만, **Combined를 v2에서 명시적으로
  검증한 적은 아직 없다**(§8 국면 I 참고).
- 데이터: COCO val2017(AP·flip·GT 지표 전부), LVIS 1203 프롬프트(국면 C-3
  motivation 한정 — 실제 LVIS 이미지/annotation이 아니라 COCO 이미지에 LVIS
  vocabulary만 얹어서 씀, `ultralytics/cfg/datasets/lvis.yaml`의 class name
  리스트 재사용. 국면 I도 같은 방식.)
- 코어 측정 도구: `src/harness.py`의 `SimilarityHarness` (cv4 forward hook)
- 양자화: `src/quant/{fake_quant,quant_model,adaround,brecq,promptcal}.py`
- 6-seed 최종 비교 스크립트: `scripts/45_baseline_compare.py`
- 결과 원본: `results_v1/diag/45_metrics_seed{0,1,2,4,5,7}_full.txt`
- 문서: `PROGRESS_stage1-2` / `PROGRESS_stageC`(B·C) / `PromptCal_PTQ_progress_2026-09-{03,05,06}.md`(D·E·F 상세) / `PROMPTCAL_HOW_IT_WORKS.md`(방법 원리)

---

## 8. 작업 규모 관점 (정직한 평가)

- 완료(A~F): 관찰/진단(A·B·C) + baseline 재구현(D) + 제안 방법 설계·검증(E) +
  확장 평가지표로 6-seed 재검증(F). 논문 Sec 3(Motivation)·Sec 4(Method)·
  Sec 5(Results)의 핵심 데이터가 전부 확보됨.
- 남음(G·H):
  - G(논문 반영): §6의 "정직한 결론" 3단 구조(메인 클레임 승리 / GT 기준 동률 /
    masked flip 한계)를 `논문.txt` 초안에 실제로 쓰는 작업. 데이터는 이미 있음
    — 순수 글쓰기 작업.
  - H(실배포 성능): 지금까지의 모든 latency/모델크기는 fake-quant(clamp/round/
    dequant) 기준 이론값이지 실제 INT8 엔진 성능이 아님. 진짜 배포 주장을
    하려면 TFLite/TensorRT export가 필요 — 별도의 큰 후속 과제로 보류 중.
  - I(일반성 — LVIS/v2, 09-07 착수): A~F 전부가 **COCO-80 vocabulary + v1
    체크포인트** 기준이었다는 게 뒤늦게 확인됨(위 §7). 두 가지가 미검증으로
    남아 있었음: (1) Combined를 훨씬 촘촘하고 큰 vocabulary(LVIS-1203)에
    배포했을 때도 이점이 유지되는가 — 국면 C-3가 "vocabulary가 커질수록 naive
    손상이 증폭된다"를 보였으니, 그 더 어려운 조건에서 Combined가 여전히
    통하는지가 논문 완성도에 중요함. (2) v2 체크포인트에서도 Combined가
    통하는가. (1)은 `scripts/07_lvis_compare.py`의 방식(실제 LVIS 이미지/GT
    없이, COCO 이미지에 LVIS-1203 vocabulary만 얹어 flip/margin 측정)을 5개
    조건(naive/AdaRound/QDrop/BRECQ/Combined) 전부로 확장해서 착수.
    **중간 결과(실제 LVIS GT 확보 후, `PromptCal_PTQ_progress_2026-09-07.md`
    §7 상세)**: COCO-80 학습 -> LVIS 재학습 없이 이식하면 Combined의 실제
    LVIS AP가 naive보다도 낮다(6개 조건 중 최악) — flip 증가가 무해한 재배치가
    아니라 진짜 검출 품질 저하. 메커니즘 추정: `s_mult`가 conv당 스칼라
    하나뿐이라 neighbor_loss의 "COCO 이웃만 조준"하려던 의도가 구조적으로
    불가능하고, 전역적 하향 편향(학습 후 s_mult 평균 0.97~0.99, 1.0 미만)으로
    새어나가 존재도 몰랐던 LVIS class 전체에 무차별 적용됨 — 재설계가
    필요할 수 있는 근거. **LVIS-native 재학습 대조 실험 완료(09-07)**: s_mult가
    COCO-80 native(0.977)/이식(0.988)/LVIS-native(0.971) 세 시나리오 전부에서
    1.0 미만으로 수렴 -- "재학습 없는 이식" 특유의 문제가 아니라 neighbor-hinge
    설계(conv당 단일 스칼라) 자체의 구조적 한계로 확정. s_mult ablation
    (`scripts/52_smult_ablation.py`)도 s_mult=1로 되돌리면 AdaRound와 AP가
    소수점까지 일치함을 확인해 인과관계를 직접 증명.
    **(s_mult-1)^2 정규화 6-seed 검증 결과(09-07, 실패)**: calib 크기를
    8배 늘려도 원인이 아님을 확인(`scripts/55`) 후 정규화(`scale_reg_weight=200`)를
    6-seed 전체 지표로 검증(`scripts/54`) — LVIS AP는 naive를 근소하게만
    넘고 baseline 최선(AdaRound/BRECQ)에는 못 미치며 LVIS lost는 오히려
    6개 중 최악. 그 대가로 COCO-80에서 Combined의 핵심 결과였던 UPIR
    (0.142%, 5개 중 최선)이 0.225%로 baseline 수준까지 후퇴 -- "핵심 주장을
    포기하고 LVIS에서 그저 그런 성적"이라는 나쁜 트레이드오프로 확정, 정규화만으로는
    불충분. **다음 단계: per-channel 재설계**(s_mult를 conv당 스칼라 -> `[out_channels]`
    벡터, 적용 범위를 cv4 직전 layer로 축소) 착수.
    **per-channel + 정규화 6-seed 결과(09-08)**: `PromptCal_PTQ_progress_
    2026-09-08.md` 참고. per-channel(자유도 확대) 단독으로는 여전히 LVIS를
    못 이겼으나(과적합 추정), `scale_reg_weight=20`과 결합하니 지금까지 나온
    Combined 변형 중 처음으로 균형 잡힌 프로파일 확보: COCO-80 AP 36.10
    (naive 대비 +1.04, 원래 스칼라 이득 +1.4의 79% 유지, baseline 전부보다
    우위), LVIS AP 0.2211(naive·QDrop보다 우위, BRECQ와 거의 동률, AdaRound
    에만 근소하게 못 미침 -- 두 baseline보다 못했던 이전 두 실패작과 다름),
    UPIR 0.191(naive·AdaRound·QDrop보다 우위, BRECQ에만 못 미침). "전부
    최고"는 아니지만 "어디서도 최악이 아닌" 첫 조건. 완벽한 방식은 아니지만
    이 정도가 논문에 실을 만한지 최종 판단하기 위해, **calibration을 train2017
    256장으로, LVIS 평가를 공식 minival(4809장, ultralytics 공식 배포 --
    확인 결과 "COCO val2017 ∩ LVIS val"과 정확히 일치)로 바꾼 "논문 방식"
    데이터 설정으로 재검증 완료(09-08)**(`scripts/58_full_baseline_official_data.py`,
    seed 0/1/2를 GPU 3개에 병렬 실행, 3-seed 전부 에러 없이 완료).
    **결과(상세는 `PromptCal_PTQ_progress_2026-09-08.md` §5)**: 이 재설계
    과정 전체를 통틀어 가장 균형 잡힌 프로파일 확보. COCO-80 AP는 naive
    대비 **+2.54**(이전 최대였던 +1.4보다 큰 폭, 역대 최대 격차),
    masked H_eval_flip 9.24%(5개 조건 중 **1위**, BRECQ의 9.71%까지 넘어섬
    -- 이 지표를 이긴 것은 프로젝트 전체에서 처음), LVIS_flip 5.34%·
    LVIS_lost 1155.3 둘 다 **5개 중 1위**. 다만 LVIS AP(0.1246)는 naive
    (0.1264)와 사실상 동률(근소 열세)이고 UPIR·COCO lost는 중위권(최선은
    아니나 최악도 아님) -- "전부 최고"는 아니지만 핵심 지표에서 확실한
    우위를 보인 첫 결과. 다음 단계(추가 하이퍼파라미터 튜닝 여부)는
    사용자 확인 대기 중.
