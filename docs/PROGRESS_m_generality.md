# PromptCal-PTQ — 크기 일반성 검증 (yolov8m-world / v1-medium)

목적: 핵심 관찰(프롬프트 축 semantic decision 손상)이 모델 크기와 무관하게 성립하는지 확인.
결론: **성립하며, 오히려 모델이 클수록 손상이 커진다.**

모델: yolov8m-world (v1, medium). 269 layers, 180M params, wrap 대상 vision Conv 91개.
harness/파이프라인은 s와 동일(cv4 3개, 유사도 [8400,80]) — 코드 무수정 재사용.

---

## 핵심: 세 모델 비교 (프롬프트 축)

| 지표 | v2 (worldv2-s) | v1-s | v1-m |
|---|---|---|---|
| C-1 substitution 편향 | 10.9× | 9.6× | 10.1× |
| C-2 held-out flip | 7.2% | 10.8% | 15.8% |
| C-3 LVIS flip (COCO→LVIS) | 0.51→4.28% | 0.81→6.12% | 0.97→16.22% |
| C-3 LVIS 이웃 편향 | 31.3× | 35.0× | 31.3× |

**모델이 커질수록(v2→s→m) held-out·LVIS 손상이 단조 증가.** substitution 편향은 세 모델 모두
~10×로 견고. 프롬프트 축 semantic 손상은 크기와 무관하게 성립하며, 큰 모델에서 더 심하다.

---

## m 상세 결과

### A. baseline
mAP 41.5 (mAP50 56.3, mAP75 45.1). harness OK (cv4 3개, 최고 앵커 tv 0.945).

### B. 양자화 AP / margin — ★주의: box 축 손상, semantic 축 아님
- W8A8 AP: 41.5 → 27.5 (−13.7). calib 64/256 동일 → calib 무관.
- ★ 이 AP 하락은 **box localization 축**의 손상. 우리가 보는 semantic decision(cv4)은 건강:
  - 레이어별 단독 양자화 민감도(00d): 전체 양자화 시 confident flip 0.95%(멀쩡),
    개별 레이어 최대 0.32% — outlier 레이어 없음.
  - margin 곡선은 배경 앵커(maxprob<0.05가 98.8%) 오염으로 맨 앞 bin이 낮게 보이나,
    margin 0.1~0.5 구간 flip 68%로 경계 집중 손상 자체는 존재.
- 교훈: AP(총량 지표)는 축을 구분 못 함. box가 망가져도 semantic은 멀쩡할 수 있고(m),
  그 반대도 가능. → semantic 축을 직접 봐야 함(C).

### C-1. substitution (2000장, k=5)
confident 110,163 / flip 1,130 (1.0%) / flip→이웃 63.7% / 우연 6.3% / **편향 10.1×**.

### C-2. held-out (5 seeds)
| seed | seen% | held-out% | 악화배수 |
|---|---|---|---|
| 0 | 1.31 | 14.71 | 11.2× |
| 1 | 1.31 | 13.91 | 10.6× |
| 2 | 1.65 | 15.04 | 9.1× |
| 3 | 0.86 | 20.72 | 24.0× |
| 4 | 0.72 | 14.41 | 19.9× |
| **mean** | **1.17** | **15.76** | **15.0×** |
| std | 0.34 | 2.51 | — |

### C-3. LVIS (500장, k=20)
| vocab | flip | margin | small-margin | 이웃 편향 |
|---|---|---|---|---|
| COCO-80 | 0.97% | 6.52 | 1.4% | 3.0× |
| LVIS-1203 | 16.22% | 1.38 | 28.2% | 31.3× |

---

## 논문 프레이밍 함의

세 모델의 AP 하락 폭이 제각각(v2 −0.5, s −3.5, m −13.7)인데 semantic 손상은 일관되게 성립
(오히려 크기와 함께 증가). 이는 **AP 같은 총량 지표가 semantic decision 손상을 신뢰성 있게
반영하지 못함**을 세 각도로 보여줌 — 과소평가(v2)와 과대평가/오귀인(m, box 축) 둘 다 발생.
→ open-vocabulary detector 양자화는 semantic decision 축을 직접 측정·보존해야 함.

주의(정직): m의 margin 곡선은 s만큼 깨끗하지 않음(배경 앵커 오염, 일부 비단조). 논문에선
m을 "크기 일반성 확인"으로 substitution+held-out+LVIS 중심 인용, s를 메인 모델로.

## 실험 메모
- 00d_m_layer_sensitivity.py: 레이어별 단독 양자화 민감도(outlier 탐지). m엔 outlier 없음.
- B에서 AP만 보면 오판. s(v1)에서도 같은 함정 → C로 넘어가 확인하는 것이 핵심.
- 다음: m D(baseline) 재현 — s처럼 네 방법이 held-out에서 수렴하는지.
