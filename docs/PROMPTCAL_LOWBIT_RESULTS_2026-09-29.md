# PromptCal 저비트 실험 결과 정리 (2026-09-29)

> `CONVERSATION_SUMMARY_2026-09-28.txt`의 실행 계획(E0~E5)에서 출발한 하루치 실험 기록과 이후 계획.
> run: `runs/123` ~ `runs/144`. 코드: 커밋 `bd2f735`, `aace6dc` + 미커밋 변경(§11).
> 최종 갱신: 09-29 저녁 (runs/144 W4A6는 진행 중 — §9.3).

---

## 0. 한 줄 요약

- **vocabulary-metric 재구성(brecq_vm)은 W4A8에서 실패**했다. 어떤 metric을 써도 LVIS AP가 떨어졌다.
- **W4A8 손상은 C2fAttn concat을 받는 conv(특히 `12.cv2`)의 텍스트 attention 분기**에 집중된다.
  - YOLOv8s-World v1, YOLOv8s-WorldV2, YOLOv8m-World 세 모델에서 모두 재현된다.
- **최종 방법 후보:** "민감 conv의 weight를 8bit로 보호 + 공유 제약 scale 이전 + BRECQ". 같은 절차를 모든 비트에 적용한다.
  - **W8A8:** 보호 단계는 자동으로 비용 0이 된다. BRECQ 대비 LVIS_flip −30~33%, COCO AP가 FP 수준. 모든 baseline(6-seed 평균)보다 좋다(2 seed).
  - **W4A8:** conv 4개 보호(+1.3% 크기). BRECQ 대비 LVIS AP +1.43~1.48점, LVIS_flip −32~35%(2/2 seed). 같은 크기에서 무작위 선택은 효과 없음.
- **PromptCal(Combined)을 새 양자화기 위에 얹어도 이득이 없고**, held-out 지표(Heval_flip)는 2/2 seed에서 악화된다 → ablation의 음성 결과.
- **A4는 per-tensor activation의 절벽이다.** conv의 76%를 A8로 되돌려도 붕괴한다. A5부터 붕괴가 풀린다.

**논문 논지 후보:** "open-vocab 검출기 PTQ에서 무너지는 것은 held-out 어휘의 판정 순위이고, 그것은 특정 구조적 지점(텍스트 융합 concat, 초반 backbone)에서 일어난다. 그 지점을 싸게 지키면 순위가 보존된다. calibration 어휘에 맞춘 학습(PromptCal)은 오히려 held-out 순위를 해친다."

---

## 1. 공통 설정

| 항목 | 설정 |
|---|---|
| 모델 | YOLOv8s-World(v1, ContrastiveHead). 일반화 확인에 v2(BNContrastiveHead)와 m 사용 |
| 양자화 범위 | conv 70개 전부, head 포함(`--no-skip-head`). CLIP, DFL, attention Linear/matmul은 FP |
| weight | 출력 채널별 비대칭, MSE(L2.4) scale |
| activation | conv 입력 per-tensor 비대칭, MSE observer |
| 첫/마지막 레이어 | stem 첫 conv, head cv2/cv3 마지막 1×1 = 8bit (`--first-last-bits 8`, W4 설정에만 의미) |
| PTQ | BRECQ (AdaRound rounding 2000 iter + LSQ), calibration train2017 256장 |
| 평가 | COCO val2017 AP, LVIS minival AP(held-out vocabulary), FP 대비 flip/lost |
| 구현 | fake-quant 시뮬레이션 (실제 정수 커널 아님) |
| GPU | 본 비교는 전부 RTX 4000 Ada(GPU 4~7). 예외는 표에 표시 |

---

## 2. 최종 비교표 (현재까지)

### 2.1 W8A8 (head 포함)

| 방법 | seed | COCO AP | LVIS AP | LVIS_flip | LVIS_lost | Heval_flip |
|---|---|---|---|---|---|---|
| FP32 | – | 36.80 | 0.2589 | – | – | – |
| naive | 6-seed 평균 | 36.36 | 0.2540 | 5.39% | 1197 | 7.94% |
| AdaRound (MSE) | 6-seed 평균 | 36.50 | 0.2521 | 4.64% | 1049 | 7.01% |
| QDrop | 6-seed 평균 | 36.54 | 0.2538 | 3.88% | 840 | 6.35% |
| BRECQ | 6-seed 평균 | 36.29 | 0.2514 | 3.97% | 861 | 6.52% |
| Combined (PromptCal) | 6-seed 평균 | 36.58 | 0.2555 | 3.87% | 717 | 7.56% |
| **BRECQ + 공유 제약 scale 이전 (우리)** | s0 / s1 | **36.86 / 36.76** | **0.2577 / 0.2547** | **2.67 / 2.69%** | **624 / 577** | **3.98 / 4.07%** |

baseline 행은 BASELINE_STATUS §1.1(runs/115·121)의 6-seed 평균이고, 우리 행은 2 seed(runs/135)다. 6-seed 확장이 필요하다(§12).

### 2.2 W4A8 (head 포함, 첫/마지막 8bit)

| 방법 | 크기 | seed | COCO AP | LVIS AP | APr | LVIS_flip | LVIS_lost | Heval_flip |
|---|---|---|---|---|---|---|---|---|
| FP32 | 약 49 MiB | – | 36.80 | 0.2589 | 0.1767 | – | – | – |
| naive | 6.14 MiB | – | 붕괴 (약 1.1)* | 약 0.01* | – | – | – | – |
| BRECQ | 6.14 | s0 / s1 | 33.35 / 33.19 | 0.2250 / 0.2243 | 0.1463 / 0.1524 | 18.38 / 18.01% | 3956 / 3848 | 25.82 / 23.77% |
| + scale 이전 (공유 제약) | 6.14 | s0 / s1 | 33.98 / 33.79 | 0.2271 / 0.2286 | – | 16.59 / 16.64% | 3393 / 3501 | – |
| + conv W8 보호 (누수 없음) | 6.22 | s0 / s1 | 34.69 / 34.57 | 0.2375 / 0.2361 | 0.1658 / 0.1742 | 12.31 / 11.98% | 2569 / 2442 | 18.84 / 18.67% |
| **+ conv W8 보호 + scale 이전 (우리)** | **6.22** | s0 / s1 | **34.81 / 34.90** | **0.2398 / 0.2386** | 0.1738 / 0.1726 | **11.92 / 12.18%** | **2531 / 2563** | **17.48 / 18.57%** |
| 같은 크기 무작위 conv W8 (3개 조합) | 6.23~6.24 | s0 | 33.3~33.4 | 0.2263~0.2276 | – | 17.5~18.3% | 3837~3929 | – |

\* naive는 runs/113(head 제외 조건) 값이다. head 포함이면 더 나쁘다. **같은 프로토콜의 naive, AdaRound, QDrop, Combined 행이 아직 없다**(§12 P2).

conv W8 보호의 seed1 값은 runs/140 w4a8_convw8_seed1의 BRECQ 행이다.

---

## 3. E0~E4: 요약 문서 계획대로 진행한 실험

### 3.1 E0 — A4는 첫/마지막 레이어 8bit로도 살아나지 않음 (`runs/123`, seed0, eval-cap 500)

| COCO AP / LVIS AP | naive | qdrop | brecq |
|---|---|---|---|
| W4A4 | 0.00 / 0.0000 | 0.03 / 0.0024 | 0.05 / 0.0012 |
| W8A4 | 0.00 / 0.0004 | 0.02 / 0.0004 | 0.00 / 0.0002 |

### 3.2 E1 — 텍스트 부분공간 진단 (W4A8, 200장)
- lam_mean=1일 때 C의 유효 차원은 1.7/512다. 평균 방향 하나가 지배한다.
- naive: err_iso 1.209, rho 0.622
- BRECQ: err_iso 0.0897, rho 0.868 → 진행 기준은 통과했다.

### 3.3 E2~E4 — brecq_vm은 W4A8에서 전부 실패 (`runs/123`, `124`, `126`)

| W4A8 (brecq → brecq_vm) | seed | LVIS AP 변화 | LVIS_flip 변화 |
|---|---|---|---|
| lam_mean=1 (generic vocab) | 0 / 1 | −0.33 / −0.43점 | +0.47 / +0.52%p |
| lam_mean=0 | 0 / 1 | −0.33 / −0.30점 | −0.06 / +0.20%p |
| identity (C=I) | 0 | −0.65점 | +0.41%p |
| lam0 + anchor uniform / mix 0 / mix 1 | 0 | −0.12 / −0.30 / −0.15점 | +0.24 / −0.05 / +0.03%p |

- COCO AP는 모든 변형에서 +0.2~0.4 올랐지만 LVIS는 모두 나빠졌다.
- W8A8 (runs/125·127): identity +0.22 / +0.24점, lam0 +0.07 / +0.03점. vocabulary metric이 identity보다 나은 경우는 없었다.

---

## 4. W4 손상은 어디서 오는가

### 4.1 학습 없는 W4 민감도 진단 (`runs/128`: val2017 + LVIS 가중 → 선택 용도로는 누수)
- drop view: 블록 12가 LVIS_flip +67.8%p로 최대였고, conv 단위로는 `12.cv2` 하나가 +66.5%p다.
- restore view: `12.cv2` 하나가 회복분 28.6%p 중 27.0%p를 차지했다.
- 입력 채널별 weight 크기 편차(최대/중앙값)는 `12.cv2`가 13.3배로 모델 최대였다.

### 4.2 `12.cv2` 분해 (FP 입력, 출력 상대 오차)
입력 = concat(cv1 조각 0, cv1 조각 1, bottleneck m.0, **텍스트 attention 출력**), 분기당 128채널.

| 분기 | activation 평균 \|x\| | activation 최대 | weight 열 최대 (중앙 / 최대) |
|---|---|---|---|
| cv1 조각 0 / 1 | 0.44 / 0.34 | 7.7 / 7.0 | 0.27 / 0.81, 0.25 / 0.64 |
| bottleneck | 0.45 | 9.3 | 0.22 / 0.89 |
| **attention** | **1.62** | **20.4** | 0.25 / **3.32** |

| 설정 | 출력 상대 오차 |
|---|---|
| W8A8 | 0.005 |
| W4A8 (현재) | **0.83** |
| attention 분기만 W4 (나머지 FP) | 0.80 |
| 분기별 weight scale | 0.24 |
| \|W\|·\|x\| 상위 2채널만 W8 | 0.073 |

### 4.3 원인 분석 (`runs/139`, 학습 없음, W8A8 기준, train2017 64장, LVIS guide + LVIS 평가)

| 변형 | LVIS flip (P3 / P4 / P5 \| 전체) | rho |
|---|---|---|
| 기준 W8A8 | 6.8 / 6.6 / 5.9 \| 6.5% | 1.28 |
| 12.cv2 W4 | 54 / 87 / 94 \| 79.2% | 1.07 |
| **12.cv2 W4 + attention 분기만 W8** | 6.6 / 8.1 / 5.7 \| **6.9%** | 1.29 |
| 12.cv2의 attention 분기만 W4 (분기별 scale) | 18 / 30 / 26 \| 25.2% | 1.36 |
| 15.cv2 W4 | 18 / 10 / 6 \| 11.4% | 0.91 |

- **H3(텍스트 attention 분기): 지지된다.** attention 분기 열(모델 weight의 0.26%)만 W8로 두면 거의 완전히 회복된다. 손상은 attention 분기 자체의 오차와, 공유 scale을 통해 다른 분기까지 망가지는 효과가 겹친 것이다.
- **H1(어휘 밀도): 보조 요인이다.** 점수 차 <0.5 anchor가 LVIS 약 880개, COCO 약 50개다. 다만 12.cv2 W4는 점수 차 >2인 판정도 70% 넘게 뒤집는다(임베딩이 통째로 망가짐).
- **H4(위치): 부분 지지된다.** 12.cv2 손상은 P4/P5에, 15.cv2 손상은 P3에 몰린다.
- **H2(오차 방향): 원인이 아니다.** 12.cv2 오차의 rho는 기준보다 낮다.
- **H5(guide vocabulary): 영향 없다.** COCO guide 80.3% vs LVIS guide 79.2%.

### 4.4 모델 간 재현 (`runs/137`, 누수 없음: train2017 200장, COCO 가중, 임베딩 오차 순위)

| 모델 | 상위 conv |
|---|---|
| s-v1 (runs/134) | 12.cv2, 1.conv, 2.m.0.cv2, 3.conv, 15.cv2 |
| s-v2 | 12.cv2, 15.cv2, 18.cv2, 1.conv, 21.cv2 |
| m-v1 | 15.cv2, 12.cv2, 23.cv3.0.1, 1.conv, 2.cv2 |

### 4.5 outlier 채널의 출처 (채널 계보 추적)
- 모델 전체 1·2위 outlier는 12.cv2의 텍스트 attention 채널이다(conv 안 중앙값의 87배, 79배).
- 상위 1% 안에서 attention 출력 채널은 전체 비중 대비 3.0배로 과대표현된다.

---

## 5. W4A8 처방 비교 상세 (BRECQ 위)

| 처방 | 크기 | seed | COCO AP | LVIS AP | LVIS_flip | 출처 |
|---|---|---|---|---|---|---|
| scale 이전 (독립, 상한) | 6.14 | 0 / 1 | 34.17 / 34.19 | 0.2261 / 0.2275 | 16.22 / 15.72% | 130 |
| 채널 W8 0.5 / 1 / 2 / 5% | 6.18~6.45 | 0 | 33.95~34.43 | 0.2281~0.2340 | 14.07~12.54% | 131, 133 |
| 채널 1% + 이전 (독립 / 공유) | 6.21 | 0 | 34.78 / 34.38 | 0.2362 / 0.2318 | 12.90 / 13.99% | 131, 132 |
| conv W8: 12.cv2 하나 | 6.20 | 0 / 1 | 33.92 / 33.76 | 0.2262 / 0.2268 | 14.48 / 14.25% | 133 |
| conv W8: 누수 없음 · 임베딩 기준 | 6.23 | 0 / 1 | 34.54 / 34.38 | 0.2331 / 0.2340 | 12.65 / 12.48% | 134 |
| conv W8: 누수 없음 · COCO flip 기준 | 6.22 | 0 | 34.69 | 0.2375 | 12.31% | 134 |
| conv W8: 누수 있음 | 6.23 | 0 / 1 | 34.46 / 34.35 | 0.2355 / 0.2390 | 13.20 / 12.84% | 133 |
| fusion 지점 네 곳 W8 (12/15/19/22.cv2) * | 6.53 | 0 | 33.99 | 0.2315 | 13.61% | 138 |
| 블록 W8: 1, 2, 4, 12 | 6.62 | 0 | 35.16 | 0.2411 | 10.39% | 129 |

\* GPU 1(L40S)에서 측정.

**선택 기준 비교 (`runs/136`, v1, 누수 없음) — 기준은 차별화 요소가 아니다**

| 기준 | 예산 내 선택 | 임베딩 기준과 Spearman |
|---|---|---|
| 임베딩 방향 오차 | 12.cv2, 1, 2.m.0.cv1, 2.m.0.cv2 | – |
| 출력 MSE (Hessian 근사) | 12.cv2, 1, 15.cv2 | +0.99 |
| COCO flip | 12.cv2, 1, 2.cv2, 4.cv2 | +0.62 |
| 로컬 출력 오차 | 12.cv2, 2.m.0.cv1, 1, 2.cv2, … | +0.31 |
| weight 오차 | head cv2들 | −0.14 |

v2에서는 모든 기준이 같은 3개(12.cv2, 15.cv2, 1)를 고른다.

**해석**
1. 위치 선택이 결정적이다. 무작위 선택은 효과가 없다.
2. 누수 없는 선택이 누수 있는 선택만큼 좋다.
3. 채널 W8은 LVIS AP 약 0.234에서 포화한다.
4. fusion 지점만 지키면 상한은 +0.65점이다. 나머지 이득은 초반 backbone(1.conv, 2.x)에서 온다.
5. conv W8 보호 위에 공유 제약 scale 이전을 더하면 AP는 2/2 seed에서 소폭 오른다(LVIS +0.23 / +0.25점). 순위 지표는 1/2 seed만 개선됐다(runs/141).

---

## 6. W8A8 처방: 공유 제약 scale 이전 (`runs/135`)
§2.1의 우리 행이다.
- 2/2 seed에서 LVIS_flip −30~33%, LVIS_lost −28~32%, Heval_flip −31%, COCO AP +0.4~0.7이다.
- LVIS AP는 +0.23 / +0.53점으로 seed 편차가 있다.
- 비트 비용이 없다.

---

## 7. PromptCal(Combined)을 새 양자화기 위에 얹기 (`runs/140`)

같은 run 안에서 `combined − brecq`를 비교했다. 양자화기 옵션은 두 조건에 똑같이 적용했다.

| 설정 | seed | COCO AP | LVIS AP | LVIS_flip | LVIS_lost | Heval_flip | lost |
|---|---|---|---|---|---|---|---|
| W4A8 + conv W8 | 0 | +0.09 | +0.02 | +0.39%p ✗ | +9 ✗ | +0.92%p ✗ | +98 ✗ |
| | 1 | +0.04 | +0.08 | +0.76%p ✗ | +275 ✗ | +2.12%p ✗ | +83 ✗ |
| W8A8 + scale 이전 | 0 | −0.07 | +0.01 | −0.15%p | −70 | +0.28%p ✗ | −1 |
| | 1 | 0.00 | +0.21 | −0.19%p | −63 | +0.42%p ✗ | +42 ✗ |

- W4A8: AP는 그대로이고 held-out 판정 지표가 2/2 seed에서 악화됐다.
- W8A8: LVIS 쪽은 미세하게 개선됐지만 Heval_flip은 2/2 악화됐다.
- §1.1(W8A8 6-seed)에서 Combined가 보인 약점과 같은 패턴이다. calibration 어휘에 맞추는 보정은 held-out 순위를 해친다.
- **논문에서의 위치:** ablation의 음성 결과로 쓴다.
- **참고:** 이 run의 BRECQ 행은 runs/134와 0.1점 이내로 다르다(adaptive max pool backward의 비결정성).

---

## 8. 한 방법으로서의 정리
- **방법:** ① 누수 없는 진단으로 예산 B 안의 민감 conv weight를 8bit로 보호 ② 공유 제약 scale 이전 ③ BRECQ
- **W8A8:** ①은 규칙상 자동으로 비용 0이 된다(이미 8bit).
- **ablation으로 보일 것:**

  | | BRECQ | + scale 이전 | + conv 보호 | + 둘 다 |
  |---|---|---|---|---|
  | W4A8 | runs/123·140 | runs/132 | runs/134·140 | runs/141 |
  | W8A8 | runs/125·127 | runs/135 | 해당 없음 | = scale 이전 |

- **비트별 진단과의 대응:** W8A8의 병목은 activation이라 scale 이전이, W4A8의 병목은 weight라 conv 보호가 주로 기여한다.

---

## 9. A4 / A5 / A6

### 9.1 activation 민감도 (`runs/138`, W8A8에서 conv 하나씩 입력만 A4)
- 상위 conv: 12.cv1(COCO +30, LVIS +62), 22.cv1, 12.cv2, 8.cv2 — 모두 concat을 입력으로 받는다. 상위 5개가 합계의 66%다.

### 9.2 누적 복구 곡선 (`runs/142`, 기준 W8A4, train2017 200장)

| A8로 되돌린 conv 비율 | COCO flip | LVIS flip |
|---|---|---|
| 0% | 84.5% | 99.3% |
| 19% | 80~85% | 98% |
| 51% | 72~79% | 90~91% |
| **76%** | **69~72%** | **88~89%** |
| 100% (= W8A8) | 0.8~0.9% | 5.9~6.2% |

→ **A4 activation이 하나라도 남으면 누적되어 붕괴한다.** 소수 conv를 A8로 두는 방식으로는 W4A4를 열 수 없다.

### 9.3 activation 비트 스윕 (`runs/143`, naive, weight 8bit, train2017 200장)

| 설정 | COCO flip | LVIS flip |
|---|---|---|
| W8A4 | 84.5% | 99.3% |
| W8A5 | 17.8% | 47.9% |
| W8A6 | 10.2% | 34.0% |
| W8A7 | 1.5% | 8.2% |
| W8A8 | 0.8% | 5.0% |

- A5부터 붕괴가 풀린다.
- 모든 비트에서 LVIS가 COCO보다 세 배 이상 민감하다.
- **진행 중 (`runs/144`, seed0):** BRECQ W4A6 / QDrop W4A6 / 우리 방법 W4A6 / BRECQ W4A5

---

## 10. 구현 검증 기록
- **기본 플래그의 기존 동작 보존:** 기본값에서는 수정 전 HEAD와 bit-identical이다(naive, 짧은 BRECQ, W4A8).
- **scale 이전 FP 등가성:** TF32를 끄면 상대 오차가 1e-5 수준이다. 파이프라인은 TF32 conv로 돈다(별도 확인 필요).
- **공유 제약 이전의 배포형 등가성:** 생산자가 x/s를 내보내고 소비자는 W·s만 쓰는 형태가 상대 오차 5.5e-6으로 같은 함수다. attention 입력 앞에는 x·s 역양자화를 넣었다.
- **채널 계보 추적기:** 교란이 conv를 통과해 퍼지는 문제와, 텐서가 in-place로 덮어써지는 문제를 고쳤다. 여러 생산자로 묶이는 경우는 residual add뿐이다(최대 그룹 3채널).
- **PromptCal과의 결합:** scale 이전은 Stage 1이 BRECQ인 Combined와 함께 쓸 수 있다(assert 완화).

---

## 11. 코드 상태

**커밋됨 (브랜치 `baseline-confounds-6seed`)**
- `bd2f735`: 진단 스크립트, `--hi-wbit-blocks`, `--hi-col-frac`, `--mig-alphas`, `--mig-tied`, `quant/migrate.py`, `quant/channel_graph.py`
- `aace6dc`: `--hi-wbit-convs`

**미커밋**
- 진단 확장: `--conv-level`, `--eval-source`, `--weight-vocab`, `--rank-json`, `--drop-kind a`, `HeadSim`(v2 지원), 출력 MSE 지표
- `set_conv_abits` / `--hi-abit-convs`
- scale 이전 + Combined assert 완화
- 이 문서
- 분석 스크립트: `runs/134~143`의 `*.py`

모든 옵션은 기본값이 꺼짐이다. `pipeline/BASELINE_STATUS.md`와 `docs/`의 다른 문서에는 세션 전부터 있던 미커밋 수정이 섞여 있다.

---

## 12. 앞으로 할 일 (우선순위순)

### P0. 방법을 규칙으로 고정 — **완료 (09-29)**
- [x] **conv 보호 규칙:** 예산 B = 양자화 weight 파라미터의 1.5%, **선택 기준 = COCO flip**(calibration 어휘 top-1 flip 증가량, 사용자 결정).
  - 진단: train2017 앞 200장(calibration 부분집합), naive W8A8에서 conv 하나씩 W4.
  - LVIS는 로드조차 하지 않는다.
  - greedy 규칙: 예산의 110%까지 담고, 90% 이상 차면 멈춘다.
- [x] **scale 이전 하이퍼파라미터:** α ∈ {0, 0.25, 0.5, 0.75, 1}, 탐색 이미지 8장, 공유 제약(`--mig-tied`).
- [x] **자동화:** `--protect-budget 0.015 [--protect-criterion coco]`가 진단을 돌려 목록을 뽑는다.
  - 결과는 `configs/protect_cache/<모델>_n200_c256_img640.json`에 캐시한다.
  - 진단 전후로 전역 RNG를 저장·복원한다(검증: 보존됨).
  - W8 설정에서는 규칙상 아무것도 하지 않는다.
- [x] **검증 (`runs/145`):** 자동 선택이 runs/134 목록(12.cv2, 1.conv, 2.cv2, 4.cv1; 0.172M)을 그대로 재현했다. 캐시 사용과 W8 no-op 경로도 스모크 테스트로 확인했다.
- **최종 방법의 CLI:**
  - W4A8: `--protect-budget 0.015 --mig-alphas 0,0.25,0.5,0.75,1 --mig-tied --conditions brecq`
  - W8A8: 같은 옵션(보호는 자동으로 no-op)

### P1. 6-seed 확장 (주 결과, RTX 4000 Ada GPU 4~7에서만)
짝비교의 RNG 위치를 맞추려면 **비교할 조건을 각 run의 첫 조건으로 두어야** 한다(BASELINE_STATUS §5.2).

| 설정 | 조건 | 있는 seed | 필요한 run |
|---|---|---|---|
| W8A8 | BRECQ (첫 조건 단독) | 0, 1 (runs/125·127) | seed 2~5: 4개 |
| W8A8 | BRECQ + scale 이전 | 0, 1 (runs/135) | seed 2~5: 4개 |
| W4A8 | BRECQ | 0, 1 | seed 2~5: 4개 |
| W4A8 | + scale 이전 | 0, 1 (runs/132) | seed 2~5: 4개 |
| W4A8 | + conv 보호 | 0, 1 (runs/134·140) | seed 2~5: 4개 |
| W4A8 | + 둘 다 (우리) | 0, 1 (runs/141) | seed 2~5: 4개 |
| W4A8 | 무작위 conv 보호 (3개 조합) | 조합 A만 0, 1 | 조합 A seed 2~5 + 조합 B·C seed 1~5: 12개 |

- 합계 36 run. run당 약 45분, GPU 4장이면 **약 7시간**이다.
- **판정:** 짝비교 k/6과, 아무것도 바꾸지 않은 조건의 흔들림 대비 배수로 본다(§5.3 방식).

### P2. W4A8 baseline 행 채우기 (같은 프로토콜)
- [ ] naive, AdaRound(MSE), QDrop, Combined를 W4A8(head 포함, 첫/마지막 8bit)에서 6-seed로 돌린다. 5조건 run 기준 run당 약 2시간, 6개 → GPU 4장으로 **약 3시간**.
- [ ] **QDrop + 우리 방법:** BRECQ가 아닌 다른 PTQ 위에서도 통한다는 근거(plug-in 일반성). 2-seed.

### P3. 같은 예산의 혼합 정밀도 baseline
- [ ] HAWQ식(출력 MSE 기준) conv 선택으로 같은 예산 run을 2~6 seed 돌린다. 선택 목록이 우리와 같으면 "같다"고 명시하고, 다르면 성능을 비교한다.

### P4. 일반화
- [ ] YOLOv8s-WorldV2, YOLOv8m-World에서 BRECQ vs 우리 방법(W4A8, W8A8) 2-seed. m은 메모리가 20GB를 넘는지 먼저 확인한다.
- [ ] (선택) Grounding DINO 같은 early-fusion 검출기. 인프라를 새로 만들어야 한다.

### P5. W4A6 판단 (runs/144 결과 후)
- [ ] BRECQ W4A6가 붕괴하지 않고 우리 방법이 크게 회복하면, 논문 표에 "가장 어려운 설정"으로 추가하고 6-seed로 확장한다.
- [ ] W4A4는 한계로 명시한다(§9.2 곡선을 근거로).

### P6. 추가 baseline
- [ ] SmoothQuant / AWQ 정식 구현(공유 제약 scale 이전과 비교)
- [ ] Reg-PTQ(마지막 예측 레이어 FP 프로토콜 차이를 명시)

### P7. 배포 검증 (엣지 목적)
- [ ] 목표 기기를 정한다(Jetson + TensorRT INT8, 모바일 NPU + TFLite/ONNX 등).
- [ ] W8A8과 W4A8 모델을 실제로 내보내 정확도와 지연시간을 잰다. conv 보호(레이어별 정밀도)와 scale 이전(SiLU 뒤 채널별 곱셈)의 fusion 지원 여부를 확인한다.
- [ ] 결과에 따라 주 무대를 W8A8과 W4A8 중 무엇으로 할지 정한다.

### P8. 정리 작업
- [ ] 미커밋 코드와 이 문서를 커밋한다.
- [ ] TF32 conv가 FP 기준과 flip 지표에 주는 영향을 한 번 확인한다(TF32 끔 vs 켬, W8A8 BRECQ 1 seed).
- [ ] 논문 그림: 손상 지도(conv별 민감도), 12.cv2 분기 분해, 누적 복구 곡선, 비트별 LVIS/COCO 민감도.
