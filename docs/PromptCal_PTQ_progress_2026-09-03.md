# PromptCal-PTQ 연구 진행 정리 — 2026-09-03

## 0. 문서 목적

이 문서는 현재까지 진행한 PromptCal-PTQ 연구의 문제 정의, 실험 방향, 코드 구조, 실험 결과, 실패/의심 지점, 그리고 다음에 확인해야 할 사항을 기록한다.

현재 핵심 질문은 다음과 같다.

> **현재 PromptCal objective가 soft relaxation 상태에서 alpha를 움직이는 방향과, 최종 discrete quantized model에서 실제 held-out semantic decision을 보존하는 방향이 일치하는가?**

아직 새로운 objective를 확정한 상태가 아니며, 현재 가설과 실험 결과를 구분해서 기록한다.

---

# 1. 연구 배경

## 1.1 연구 대상

현재 연구 대상은 **Open-Vocabulary Object Detection (OVD)** 모델인 YOLO-World 계열이다.

주요 모델:
- `yolov8s-world.pt`
- YOLOv8s-world
- 약 249 layers
- 약 164.65M parameters
- 약 32.7 GFLOPs

주요 관심사는 PTQ(Post-Training Quantization) 상황에서 단순한 reconstruction/AP만으로는 드러나지 않는 **semantic decision의 보존**이다.

OVD에서는 이미지의 region/anchor와 prompt 간 similarity가 실제 semantic decision을 결정한다.

따라서 quantization이 전체 AP를 크게 변화시키지 않더라도 특정 region에서 FP 모델이 선택하던 prompt/class가 quantized model에서 다른 prompt/class로 바뀔 수 있다.

---

# 2. 핵심 문제의식

기존 PTQ 평가에서는 보통:
- AP
- reconstruction error
- activation/feature error
- output error

등을 사용한다.

하지만 OVD에서는 prompt-conditioned similarity가 semantic decision을 직접 결정한다.

따라서 연구의 관심은 단순히 feature/output error를 줄이는 것이 아니라:

> **양자화 후에도 FP 모델의 semantic decision structure를 최대한 보존하는 것**

이다.

현재 held-out flip은 이 현상을 평가하기 위한 지표로 사용하고 있으며, 최종 연구 방법 자체를 단순히 flip minimization으로 정의하려는 것은 아니다.

---

# 3. 연구의 큰 방향

현재 연구 방향은 다음과 같다.

1. 단순 quantization error minimization이 목적이 아니다.
2. OVD의 prompt-conditioned semantic decision을 보존한다.
3. calibration에 본 prompt와 보지 않은 prompt를 구분한다.
4. calibration에서 직접 사용하지 않은 held-out prompt에서 semantic decision이 유지되는지 평가한다.
5. 최종적으로는 문제를 측정하는 것(measurement)에 그치지 않고 실제 quantization parameter를 조정하는 positive method를 목표로 한다.

---

# 4. Prompt split

현재 최소 실험에서는 80개 COCO class prompt를 세 부분으로 나눈다.

```text
S       = Seen prompts
H_cal   = Held-out prompts used during calibration
H_eval  = Held-out prompts never used during calibration
```

현재 구현:

```python
perm = rng.permutation(80)

S      = perm[:40].tolist()
H_cal  = perm[40:60].tolist()
H_eval = perm[60:80].tolist()

pidx = S + H_cal
```

따라서:
- S = 40개
- H_cal = 20개
- H_eval = 20개
- PromptCal objective = S + H_cal = 60개
- H_eval = objective에서 제외

핵심 질문:

> H_cal에서 보존하도록 만든 semantic structure가 완전히 보지 않은 H_eval에서도 일반화되는가?

---

# 5. Held-out flip의 현재 정의

현재 `heldout_flip()`은 다음 방식으로 동작한다.

FP similarity:

```text
sf : [anchors, prompts]
```

Quantized similarity:

```text
sq : [anchors, prompts]
```

FP에서 confident top-1 prompt를 찾는다.

```python
prob = sf.sigmoid()
mp, c_fp = prob.max(-1)
conf_m = mp > conf
```

그리고 FP top-1 class가 H_eval에 속하는 confident anchor만 target으로 잡는다.

```python
target = conf_m & Hm[c_fp]
```

이후 H_eval prompt 자체를 masking한다.

```python
fp_K[:, Hm] = -1e9
q_K[:, Hm]  = -1e9
```

남은 prompt들에 대해 FP와 quantized model의 argmax가 달라지는 비율을 계산한다.

즉 현재 held-out flip은:

> **FP에서 H_eval prompt가 confident top-1이었던 anchor를 대상으로 H_eval prompt를 제거한 뒤, 남은 prompt들 사이에서 FP와 quantized model의 decision이 달라지는 비율**

이다.

일반적인 top-1 accuracy나 AP와는 다른 지표다.

---

# 6. 이전 RankSafe/09_phase1 계열의 실패

이전에는 ranking/top-k/distance/box/hardware 등 여러 목적을 결합하는 방향을 시도했다.

09_phase1(RankSafe)에서 핵심적으로 관찰한 문제:
- calibration objective가 좋아져도 held-out evaluation이 좋아진다는 보장이 없었다.
- calibration metric improvement와 held-out semantic utility 사이에 불일치가 있었다.
- 여러 loss를 섞으면서 무엇이 실제 효과를 만드는지 명확하지 않았다.

따라서 현재 국면에서는 복잡한 multi-objective를 다시 만드는 대신, 훨씬 단순한 가설부터 검증하고 있다.

---

# 7. 국면 E: 첫 관문 실험

실험 이름:

> **E 최소 실험: held-out margin 보존 vs baseline**

가설:

> H_cal을 calibration에 포함하고, H_cal prompt들의 top-k decision boundary margin을 FP와 맞추도록 quantization parameter를 최적화하면, 완전히 보지 않은 H_eval에서의 flip이 AdaRound baseline보다 감소할 것이다.

비교:
```text
FP
naive W8A8
AdaRound
PromptCal
```

초기 성공 판정 기준:
```text
PromptCal H_eval flip < AdaRound H_eval flip
```

그리고 여러 seed에서 일관되면 방향성이 성공했다고 판단하려 했다.

---

# 8. PromptCal 초기 구현

초기 PromptCal은 AdaRound의 rounding parameter `alpha`를 최적화했다.

핵심:
- weight 자체는 고정
- 기본 quantization 구조 유지
- AdaRound rounding parameter만 최적화
- FP cv4 similarity를 target으로 사용
- S + H_cal prompt subset에서 margin preservation
- confidence가 높은 anchor만 사용

현재 AdaRound convolution 수:
```text
71개
```

---

# 9. CV4 similarity capture

`pdquant.py`의 `_CV4Capture`를 사용한다.

```python
class _CV4Capture:
    def __init__(self, head):
        self.head = head
        self.buf = {}
        self.handles = []
        for i, sub in enumerate(head.cv4):
            def make(idx):
                def hook(_m, _inp, out):
                    self.buf[idx] = out
                return hook
            self.handles.append(sub.register_forward_hook(make(i)))

    def clear(self):
        self.buf = {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
```

각 level의 cv4 출력을 flatten하고 spatial dimension을 concat한 뒤 `[A, P]` 형태의 similarity matrix를 만든다.

---

# 10. 현재 margin loss

현재 margin loss는 top-(k+1) similarity를 이용한다.

계산 대상:
```text
top1 - top2
top2 - top3
...
topk - top(k+1)
```

FP와 quantized model의 margin 차이를 제곱 오차로 계산한다.

마지막 boundary인:
```text
top-k vs top-(k+1)
```
에 더 큰 weight를 준다.

현재 기본:
```text
k = 5
boundary_w = 3.0
```

---

# 11. 현재 decision loss

현재 코드에는 FP top-1을 pseudo-label로 사용하는 decision loss도 있다.

```python
target = sim_fp.argmax(dim=-1).detach()
return F.cross_entropy(sim_q, target)
```

목적:

> quantized similarity의 top-1 decision이 FP top-1 decision과 일치하도록 유도

---

# 12. 현재 PromptCal objective

현재 형태는 대략:

```text
loss =
    margin loss
  + rounding regularization
  + decision_weight * decision loss
```

초기에는 rounding을 0/1로 수렴시키기 위한 warmup도 사용했다.

warmup 동안:
- reg weight를 강하게 적용
- alpha를 discrete rounding에 가깝게 유도

이후:
- reg 완화
- margin/decision objective 중심

---

# 13. Soft vs Discrete 문제

현재 연구에서 가장 중요한 문제다.

AdaRound alpha 최적화 중에는:

```python
ac.soft = True
ac.ste = True
```

를 사용한다.

즉 optimization 과정은 실제 최종 discrete quantized weight와 완전히 동일한 상태가 아니라 미분 가능한 soft relaxation을 사용한다.

optimization 종료 후:

```python
ac.soft = False
ac.ste = False
```

로 되돌린다.

따라서 실제로 원하는 연결은:

```text
soft optimization
      ↓
alpha update
      ↓
discrete quantized weight
      ↓
actual held-out semantic decision
```

이다.

현재 결과는 이 연결이 깨질 가능성을 강하게 의심하게 한다.

---

# 14. GPU/CPU 관찰

실험 중 GPU utilization이 낮아 보이는 구간이 있었지만 `nvidia-smi dmon -s pucm`에서는 실제로 여러 GPU에서 높은 SM utilization이 관찰됐다.

특정 프로세스:
```text
PID 4014195
GPU memory 약 7704 MiB
```

CPU:
```text
python          약 62%
pt_autograd_0   약 34%
```

GPU 모니터링에서는 일부 구간에서:
```text
SM utilization 60~100%
```
가 확인됐다.

따라서 단일 순간의 GPU utilization만으로 GPU를 거의 사용하지 않는다고 판단하면 안 된다.

---

# 15. 실행 시간

PromptCal 최적화가 오래 걸려 iteration을 줄여 빠른 진단을 시도했다.

대표 실행:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/20_promptcal_minimal.py   --model yolov8s-world.pt   --coco-root /data/taeho/coco_datasets   --calib 32   --eval 1000   --seeds 0   --iters 300   --device 0   2>&1 | tee results_v1/decision/s5.txt
```

주의:
`CUDA_VISIBLE_DEVICES=6`이면 프로그램 내부에서 해당 GPU가 보통 logical `cuda:0`으로 매핑된다. 따라서 `--device 0`은 정상적인 조합이다.

서로 다른 CUDA 명령을 한 줄에 잘못 붙이면 argparse가 CUDA 환경변수를 일반 인자로 해석하여 오류가 날 수 있다.

---

# 16. 실험 결과 1 — 초기 PromptCal

결과:

```text
seed | naive | AdaRound | PromptCal | vs Ada

0    | 8.59% | 8.55%    | 9.45%     | -10.5%
```

PromptCal이 AdaRound보다 좋지 않았다.

---

# 17. 실험 결과 2 — decision-aware rounding

로그의 핵심:

```text
[promptcal] 32 calib, prompt subset 60개, alpha 71개,
decision-aware rounding(k=5)

150/1500
loss=1.0841
decision=0.9052
margin=0.1143
reg=0.6466
soft_flip=6.7%

300/1500
loss=0.1713
decision=0.0509
margin=0.0573
reg=0.6309
soft_flip=0.0%

...

1500/1500
loss=0.0964
decision=0.0008
margin=0.0378
reg=0.5782
soft_flip=0.0%
```

최종:

```text
naive     8.59%
AdaRound  8.76%
PromptCal 10.17%
```

즉 PromptCal은 AdaRound보다 1.41 percentage point 악화됐다.

---

# 18. 실험 결과 3 — discrete 상태 명시

`ac.soft=False`, `ac.ste=False`로 최종 discrete 상태를 명시했다.

결과:

```text
seed | naive | AdaRound | PromptCal | vs Ada

0    | 8.59% | 8.76%    | 9.04%     | -3.2%
```

최종 로그:

```text
150/1500
loss=1.1901
decision=0.9881
margin=0.1434
reg=0.5862
soft_flip=6.7%
h→0/1=16%

...

1200/1500
loss=1.6133
decision=1.4840
margin=0.1090
reg=0.2035
soft_flip=2.0%

...

1500/1500
loss=0.0902
decision=0.0008
margin=0.0732
reg=0.1620
soft_flip=0.0%
h→0/1=73%
```

최종:
```text
naive     8.59%
AdaRound  8.76%
PromptCal 9.04%
```

여전히 AdaRound보다 좋지 않다.

---

# 19. 가장 중요한 관찰

현재 로그에서는:

```text
soft_flip → 0%
decision loss → 거의 0
margin loss → 감소
```

하지만 최종 discrete held-out flip은:

```text
AdaRound   8.76%
PromptCal  9.04%
```

이다.

따라서 현재 확인된 사실은:

> **soft optimization에서 decision consistency가 좋아졌다는 사실만으로 최종 discrete quantized model의 held-out semantic decision이 좋아지는 것은 아니다.**

이것이 현재 가장 중요한 관찰이다.

---

# 20. 현재 가장 의심되는 원인

가장 의심되는 것은 **objective와 실제 discrete quantization 사이의 mismatch**다.

특히 alpha가 중간값에 있을 때:

```text
h ≈ 0.4
h ≈ 0.6
```

soft loss에서는 어느 방향으로 움직이는 것이 좋은지 gradient가 존재하지만 최종 discrete weight에서는 결국:

```text
round(h) = 0
```

또는

```text
round(h) = 1
```

중 하나가 된다.

따라서:

> continuous/soft alpha space에서의 좋은 방향과 실제 binary/discrete rounding에서의 좋은 방향이 다를 수 있다.

현재 실험은 이 가능성을 보여주고 있지만 아직 원인을 확정한 것은 아니다.

---

# 21. 현재 margin_loss의 추가 의심점

현재 구현은:

```python
fp_top, _ = sim_fp.topk(kk, dim=-1)
q_top, _ = sim_q.topk(kk, dim=-1)
```

처럼 FP와 quantized model에서 각각 독립적으로 top-k를 계산한다.

따라서 FP와 Q가 서로 다른 prompt를 top-k에 포함할 수 있다.

예:

```text
FP:
A > B > C > D > E > F

Q:
A > B > X > D > E > F
```

이 경우 단순히 두 top-k margin을 비교하는 것이 실제 semantic boundary preservation을 정확하게 의미한다고 보기 어렵다.

즉 현재 loss가:

> FP semantic ordering/decision boundary를 보존

하는 것과 정확히 같은 objective인지 별도 검증이 필요하다.

아직 이 부분을 수정한 실험은 하지 않았다.

---

# 22. 아직 확정하지 않은 것

## 확정되지 않음 1
margin preservation 자체가 틀렸다고 결론낼 수 없다.

## 확정되지 않음 2
H_cal → H_eval transfer가 실패했다고 결론낼 수 없다.

왜냐하면 H_cal에서 실제 discrete margin이 개선됐는지 아직 별도로 측정하지 않았기 때문이다.

## 확정되지 않음 3
AdaRound보다 PromptCal이 원리적으로 나쁘다고 결론낼 수 없다.

현재 objective와 soft/discrete 구현의 관계가 아직 완전히 검증되지 않았다.

---

# 23. 다음 진단 순서

새 objective를 바로 만들지 않는다.

먼저 세 질문을 분리한다.

## Q1. PromptCal은 실제 discrete H_cal margin을 개선하는가?

비교:

```text
AdaRound discrete
vs
PromptCal discrete
```

동일 calibration image와 동일 H_cal prompt에 대해 실제 similarity를 계산한다.

### Case A
```text
PromptCal discrete H_cal margin 개선
H_eval flip 악화
```

→ H_cal → H_eval generalization 문제 가능성.

### Case B
```text
PromptCal soft H_cal margin 개선
PromptCal discrete H_cal margin 개선 안 됨
```

→ soft-to-discrete mismatch 가능성.

### Case C
```text
PromptCal soft H_cal margin 개선 안 됨
PromptCal discrete H_cal margin 개선 안 됨
```

→ objective 자체가 실제 목표를 잘 표현하지 못할 가능성.

### Case D
```text
PromptCal H_cal margin 개선
PromptCal H_eval flip 개선
```

→ 현재 방향성이 유효할 가능성. 이후 seed 확대와 ablation 필요.

---

# 24. 왜 지금 새 loss를 만들면 안 되는가

현재는 원인이 분리되지 않았다.

예를 들어 PromptCal이 실패한 이유가:

```text
A. soft → discrete mismatch
B. H_cal → H_eval generalization 실패
C. margin loss 정의 문제
D. optimization instability
```

중 무엇인지 모른다.

이 상태에서 box loss, reconstruction loss, ranking loss 등을 계속 추가하면 어떤 요소가 실제로 문제를 해결했는지 해석할 수 없게 된다.

따라서 먼저 진단 실험을 해야 한다.

---

# 25. Flip의 연구 내 역할

현재 방향에서는 flip을 최종 방법론의 중심으로 단순화하지 않는다.

구분:

## Measurement
```text
quantization 때문에 semantic decision이 얼마나 바뀌었는가?
```

이는 문제를 보여주는 measurement다.

## Positive method
```text
quantization parameter를 어떻게 조정하면
semantic decision structure를 실제로 보존할 수 있는가?
```

현재 연구는 후자를 지향한다.

따라서 held-out flip은:

> **semantic decision preservation이 unseen prompt에서도 일반화되는지를 검증하는 evaluation signal**

로 보는 것이 더 적절하다.

---

# 26. 코드 구조

주요 파일:

```text
scripts/20_promptcal_minimal.py
src/quant/promptcal.py
src/quant/pdquant.py
src/quant/adaround.py
```

## scripts/20_promptcal_minimal.py

역할:
- COCO image load
- calibration/probe 구성
- FP model 생성
- naive model 생성
- AdaRound model 생성
- seed별 S/H_cal/H_eval 분할
- PromptCal model 생성
- held-out flip 평가

## src/quant/promptcal.py

역할:
- FP cv4 similarity cache
- quantized cv4 similarity capture
- margin loss
- decision loss
- AdaRound alpha optimization
- PromptCal optimization

## src/quant/pdquant.py

주요 helper:
```text
_find_head()
_CV4Capture()
```

## src/quant/adaround.py

주요 요소:
```text
AdaRoundQuantConv2d
list_adaround_convs
h_alpha
reg_loss
```

---

# 27. 현재 실험 조건

대표 조건:

```text
model: yolov8s-world.pt
calibration images: 32
evaluation/probe images: 1000
prompt count: 80
S: 40
H_cal: 20
H_eval: 20
PromptCal prompt subset: 60
AdaRound convs: 71
k: 5
```

현재 일부 로그에서는 `scripts/20_promptcal_minimal.py`의 command line `--iters`와 `build()` 내부 PromptCal 호출의 iteration 수가 일치하지 않는 상태가 있었으므로, 실제 실행 iteration은 로그의 `[150/1500]` 등으로 확인해야 한다.

---

# 28. 실험 결과 요약

| 실험 | Naive | AdaRound | PromptCal | 결과 |
|---|---:|---:|---:|---|
| 초기 | 8.59% | 8.55% | 9.45% | 실패 |
| decision-aware | 8.59% | 8.76% | 10.17% | 실패 |
| discrete 상태 명시 | 8.59% | 8.76% | 9.04% | 실패 |

현재까지 **PromptCal이 AdaRound baseline을 넘었다는 증거는 없다.**

오히려 PromptCal이 worse한 결과가 반복됐다.

---

# 29. 현재까지의 결론

현재까지의 실험은 단순히 "PromptCal 실패"라고 끝내기보다는 **실패 지점을 분해해야 하는 단계**다.

현재 확인된 것:

1. 단순 margin + decision optimization은 H_eval flip을 줄이지 못했다.
2. soft optimization 중 decision consistency가 좋아져도 최종 H_eval 결과가 좋아지지 않았다.
3. 따라서 soft objective와 실제 discrete quantization 결과의 관계를 검증해야 한다.
4. H_cal 자체에서 discrete semantic margin이 실제로 개선됐는지 확인해야 한다.
5. H_cal에서 개선되었다면 그 개선이 H_eval으로 일반화되는지 확인해야 한다.
6. 이 결과를 보기 전에는 새로운 loss를 계속 추가하는 것은 성급하다.

---

# 30. 다음 실험의 구체적인 목표

다음 실험은 성능 개선 실험이 아니라 **원인 규명 실험**이어야 한다.

확인할 흐름:

```text
AdaRound
   ↓
PromptCal optimization
   ↓
Discrete conversion
   ↓
H_cal margin 측정
   ↓
H_eval flip 측정
```

가능하면 다음 값을 모두 기록한다.

```text
AdaRound soft H_cal margin
AdaRound discrete H_cal margin

PromptCal soft H_cal margin
PromptCal discrete H_cal margin

AdaRound H_eval flip
PromptCal H_eval flip
```

이렇게 하면 현재 문제를:

```text
soft-to-discrete mismatch
```

와

```text
H_cal-to-H_eval generalization failure
```

로 분리할 수 있다.

---

# 31. 연구 방향의 원칙

앞으로의 설계에서 유지할 원칙:

> **"flip을 직접 최소화하는 방법"이 아니라 "semantic decision structure를 보존하는 quantization method"를 만든다.**

개념적으로:

```text
FP semantic decision structure
          ↓
quantization-aware preservation
          ↓
discrete quantized model
          ↓
unseen prompt generalization
```

---

# 32. 한 줄 요약

현재 PromptCal의 첫 시도는:

> **"H_cal의 FP top-k margin을 soft AdaRound alpha optimization으로 보존하면 H_eval semantic flip이 감소할 것이다."**

였지만 현재 결과는:

> **"soft objective는 잘 최적화되는데 최종 discrete model의 H_eval flip은 AdaRound보다 오히려 나빠진다."**

이다.

따라서 지금 당장 새로운 loss를 만드는 것보다 먼저:

> **"우리가 최적화한 것이 실제 discrete quantized model에 전달되는가?"**

와

> **"전달된다면 H_cal에서 개선된 semantic structure가 H_eval으로 일반화되는가?"**

를 분리해서 확인하는 것이 현재 가장 중요한 다음 단계다.
