# Baseline 확정 상태 및 재현 절차

**최종 갱신: 2026-09-24** · 대상 독자: 이 파이프라인으로 baseline 수치를 재현하거나,
baseline을 기준으로 제안 방법을 설계하는 사람.

이 문서는 **"지금 baseline이 어떤 상태이고, 어떻게 재현하며, 무엇이 남았는가"** 를 다룬다.
각 방법이 원 논문/공식 코드와 항목별로 어떻게 대응되는지는 저장소 루트의
[`PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md`](../PROMPTCAL_BASELINE_FIDELITY_2026-09-23.md)
(특히 09-24에 추가된 **§7 6-seed 확정 재측정**)를 볼 것. 여기서는 그 결론만 요약한다.

---

## 1. 확정 baseline (`runs/97_ada_iters2000_6seed`, 6-seed 평균)

| | COCO_AP | Heval_flip | Top1_flip | lost | LVIS_AP |
|---|---|---|---|---|---|
| naive | 36.617 | 5.945% | 0.667% | 291 | 0.2554 |
| AdaRound | 36.648 | 5.142% | 0.472% | 234 | **0.2577** |
| QDrop | **36.807** | 4.090% | 0.397% | 180 | 0.2556 |
| BRECQ | 36.755 | **3.705%** | **0.345%** | **166** | 0.2560 |
| *(참고) Combined* | 36.642 | 5.602% | 0.527% | 248 | 0.2576 |

- 이론적 모델 크기 **9.88 MiB** (W8, detector conv만. 이전에 보고하던 12.13 MiB는
  CLIP 인코더가 섞인 값이었다 — §3 참고)
- **제안 방법이 넘어야 할 선은 BRECQ의 `Heval_flip 3.705% / lost 166`.**
  seed별 짝비교에서 Combined는 naive에만 전승이고 AdaRound·QDrop·BRECQ에게
  decision 지표 0/6 전패다. 유일한 우세인 LVIS_AP(+0.0016~0.0020, 6/6)도
  AdaRound(0.2577)가 거의 같다.

---

## 2. 재현 절차

### 2.1 환경

**기존 editable ultralytics 설치가 가리키던 `/home/taeho/Mamba-YOLO`가 삭제되어
시스템 python으로는 import 자체가 안 된다.** 저장소 안의 전용 venv를 쓴다(gitignore됨).

```
ultralytics 8.4.121 | torch 2.10.0+cu128 | torchvision 0.25.0+cu128
numpy 1.26.4 | opencv 4.13.0 | lvis 0.5.3 | pycocotools 2.0.11 | faster_coco_eval 1.8.0
```

venv를 새로 만들어야 한다면:

```bash
cd /home/taeho/promptcal-ptq
python -m venv --system-site-packages .venv          # torch/numpy/opencv는 ~/.local 것을 물려받는다
.venv/bin/pip install --no-deps ultralytics==8.4.121 lvis pycocotools ftfy regex "filelock>=3.18"
.venv/bin/python -c "import torch,torchvision,numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
# 2.10.0+cu128 0.25.0+cu128 1.26.4 이어야 한다
```

**`--no-deps`가 중요하다.** 빼면 pip이 torch 2.14 스택 전체와 numpy 2.5.3을 venv에
따로 깔아 ~/.local의 torch 2.10과 충돌한다(`operator torchvision::nms does not exist`).

### 2.2 실행

```bash
export YOLO_AUTOINSTALL=false     # 없으면 ultralytics가 numpy 2.5.3을 다시 덮어쓴다

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python pipeline/run_comparison.py \
  --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
  --data configs/coco_local.yaml \
  --lvis-ann /data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json \
  --calib 256 --recon-iters-ada 2000 \
  --conditions naive,adaround,qdrop,brecq,combined \
  --seed <0..5> --device 0
```

**`--recon-iters-ada 2000`이 확정 설정이다**(기본값 1000이 아님, §4.1 참고).
나머지는 전부 기본값.

### 2.3 소요 시간 (RTX 4000 Ada 20GB 기준, 실측)

- 5개 seed 병렬(GPU 5장): **약 5시간 33분**
- 단독 실행 1개 seed: **약 1시간 25분**
- 6 seed를 GPU 5장으로 돌리면 총 **약 7시간**
- GPU당 peak 약 15GB, 프로세스당 CPU RAM 약 20GB (5-way에서 시스템 110/125GB)
- seed0 빌드 내역: naive 7s / AdaRound 4304s / QDrop 2988s / BRECQ 1742s / Combined 109s

---

## 3. 이번 세션에 바꾼 것

| 파일 | 변경 | 결과 영향 |
|---|---|---|
| `quant/quant_model.py` (+ `src/quant/` 사본) | `ALWAYS_SKIP_NAMES = {"clip_model"}` — `wrap_convs`가 CLIP 인코더로 재귀하지 않게 | **정확도 무영향**(6-seed로 naive bit-identical 확인). 모델 크기 12.13 → **9.88 MiB** |
| `run_comparison.py` | `--adaround-act-observer {minmax,mse}` 추가 | 기본 `minmax` = 기존 동작과 bit-identical |
| `run_comparison.py` | `--deterministic` 추가 | 기본 꺼짐, 무영향 |

**CLIP 건 상세**: `set_classes()`가 `WorldModel.clip_model`에 CLIP(ViT-B/32) 전체를
캐싱하는데, `wrap_convs`가 그 vision tower의 patch-embed conv
(`clip_model.model.visual.conv1`, 3→768 k32, weight 2,359,296)까지 `QuantConv2d`로
감싸고 있었다. detector forward에서 호출되지 않아 정확도에는 영향이 없었지만
`quantized_weight_mib()`에 2.25 MiB(22.8%)가 허수로 더해졌고, AdaRound가 매 run
`1/53 conv 입력 미포착` 경고를 내며 건너뛰던 것의 정체가 이거였다.
`scripts/` 아래 호출처가 50곳 가까이 되어 호출처마다 고치지 않고 `wrap_convs` 안에서
DFL과 같은 방식으로 막았다.

**논문에 모델 크기를 인용했다면 9.88 MiB로 정정할 것.**

---

## 4. baseline 확정 — 결정된 것 / 남은 것

### 4.1 결정됨

| 항목 | 결정 | 근거 |
|---|---|---|
| **AdaRound iteration 예산** | **2000** (QDrop/BRECQ와 동일) | 1000은 논문 충실도 근거가 없는 그냥 절반 예산이었다(논문 20k, AIMET 10k). 2000으로 맞추니 decision 지표가 실제 개선(Heval_flip 5.93→5.14%, lost 263→234) — 음성 대조군 대비 약 7배. 루트 문서 §7.1 |
| **AdaRound activation observer** | **min-max 유지**(as-published) | 교란변수가 아니라 **COCO↔LVIS 트레이드오프**였다. MSE로 바꾸면 COCO는 6/6 개선, LVIS는 6/6 악화(LVIS_lost +29%). min-max에서 AdaRound의 LVIS_AP 0.2577이 전 baseline 1위라 불이익도 아니다. MSE판은 ablation으로 병기. 루트 문서 §7.2 |
| **BRECQ iters 2000** | 유지 | 20000과 최종 지표 차이가 노이즈 수준, 비용만 6.2배. 단 alpha는 수렴하지 않았다(nearest 대비 flip 5.2%→12.9%) — "수렴해서 충분"이 아니라 "수렴 안 했는데도 지표가 같아서"로 서술할 것. 루트 문서 §6.2 |
| **BRECQ 2단계 분리** | **쓰지 않음**(공동 최적화) | 공식대로 켜면 Heval_flip 3.70→5.10%, lost 180→210으로 명확히 악화. 시간이 아니라 **성능** 때문이다. 루트 문서 §6.3 |
| **QDrop batch 2 / neck layer-wise** | **쓰지 않음** | 차이가 노이즈로 설명된다. 비용만 2.7배. 논문 부록 E와 어긋나는 점은 각주로 밝힐 것. 루트 문서 §6.3 |
| **CLIP 인코더 제외** | 적용 완료 | §3 |

### 4.2 남은 것

| 항목 | 상태 | 닫는 방법 | 비용 | 중요도 |
|---|---|---|---|---|
| **AdaRound batch** | 공식 32, 우리 1. 미측정 | `optimize_adaround(batch=...)`를 CLI로 노출하고 1-seed 비교. 유의미하면 6-seed | 1-seed ≈ 1.5h | 낮음 — AdaRound만 해당하고, 예산 정렬로 이미 개선됨 |
| **QDrop iters 수렴** | 미측정(BRECQ만 확인) | `--conditions naive,qdrop --recon-iters-strong 20000` 1-seed. QDrop은 drop 때문에 수렴 양상이 BRECQ와 다를 수 있다 | 1-seed ≈ 2h | **중간** — QDrop은 넘어야 할 상대라 신뢰도가 필요 |
| **AdaRound batch·QDrop iters 외 미검증 없음** | — | — | — | — |

### 4.3 확정 완료 판정

**넘어야 할 상대인 BRECQ/QDrop 쪽이 가장 단단하다** — 구현이 공식 소스와 수식 수준에서
일치하고(루트 문서 §2·§3), 설정도 재측정으로 닫혔으며, 실행 분산이 0이다.
남은 두 항목은 AdaRound batch(영향 낮음)와 QDrop iters(확인 권장)뿐이므로,
**baseline을 기준자로 쓰기 시작해도 되는 상태다.**

QDrop iters만 1-seed로 확인해 두면 baseline 쪽은 사실상 종결이다.

---

## 5. 측정 신뢰성 — 반드시 알고 있어야 할 3가지

### 5.1 baseline은 결정적이다 (Combined는 아니다)

naive/AdaRound/QDrop/BRECQ는 **독립된 run 사이에서 bit-identical**이다
(`runs/91`↔`92`, `runs/97`↔`98` 6-seed 전부 확인). 따라서 **6-seed 분산 전체가 seed
효과**(클래스 분할 + torch RNG)이고 실행 노이즈가 0이다 — 기준자로서 필요한 성질.

> **09-24 정정 — `--deterministic`을 켠 BRECQ는 예외다.** `runs/99`↔`runs/101`(5 run)에서
> `naive`는 7/7 bit-identical인데 `brecq`만 두 값 중 하나에 무작위로 떨어졌다
> (AP 36.74↔36.75, Heval_flip 3.72↔3.68, lost 190↔188). `--deterministic`에서도 결정적
> 구현이 없어 경고만 뜨는 `adaptive_max_pool2d_backward_cuda`(`ImagePoolingAttn`의
> `AdaptiveMaxPool2d`)가 유력하다 — BRECQ는 `ImagePoolingAttn`을 블록 타깃으로 재구성하므로
> 그 backward를 정면으로 통과한다. 위 6-seed 확정값(`runs/97`, `--deterministic` 미사용)은
> 영향 없고, 변동폭도 통상 측정 대상의 1/25 수준이다. 자세한 내용은 claim18-d.

**반면 Combined는 같은 seed·같은 코드·같은 RNG 스트림에서도 결과가 달라졌다**
(`runs/97`↔`98`에서 seed 2·4, 2/6). 원인을 격리한 결과(09-24):

| full scale(calib 256, iters 1500) Combined를 같은 seed로 2회 빌드 | s_mult 불일치 | max\|Δ\| | 빌드 |
|---|---|---|---|
| 기본 | **52/52 conv** | 2.48e-01 | 111s / 114s |
| `torch.use_deterministic_algorithms(True, warn_only=True)` | **0/52** | — | 114s / 115s |

**→ `--deterministic`으로 완전히 해결된다.** `cudnn.deterministic=True`는 conv
알고리즘만 고정할 뿐 backward의 atomicAdd 계열을 못 잡는다. (`--deterministic`
상태에서도 `adaptive_max_pool2d_backward_cuda`(ImagePoolingAttn의 AdaptiveMaxPool2d)만
결정적 구현이 없다는 경고가 남지만, 그건 범인이 아니다 — 그 상태로 0/52가 나온다.)

**baseline 수치는 `--deterministic`으로 바뀌지 않는다.** AdaRound/BRECQ/QDrop을
켜고/끄고 빌드해 비교하면 quant weight와 LSQ delta가 전부 bit-identical이다
(52/52 동일, iters=200·calib=64). **즉 §1의 확정 표는 그대로 유효하고 재측정이
필요 없다.** 빌드가 느려지는 비용만 있다(Combined +2~3%, baseline +13~47%
— baseline은 어차피 이미 결정적이라 켤 필요가 없다).

→ **Combined를 다루는 모든 실행에 `--deterministic`을 켤 것.** 켜면 Combined도
실행 분산 0이 되어, 설계 A vs B를 같은 seed에서 짝비교하면 차이가 전부 실재한다.

### 5.2 `--conditions` 목록이 RNG 스트림 위치를 정한다

`torch.manual_seed`는 실행 시작에 한 번만 걸리고 조건은 `--conditions` 순서대로 순차
빌드된다. **따라서 조건 목록이 다르면 같은 조건이라도 다른 난수를 받는다.**
예: qdrop은 `naive,adaround,qdrop,brecq,combined`에서 3번째, `naive,qdrop,brecq`에서
2번째다. 이 재추첨만으로 6-seed 분포만큼의 변동이 생긴다.

증거: **항상 1번째로 빌드되는 naive는 모든 run에서 bit-identical**인 반면 뒤 순서의
조건들은 그렇지 않다.

→ **두 run을 비교할 거면 `--conditions`를 반드시 동일하게 유지할 것.**

### 5.3 "6-seed 범위 밖"은 유의성 기준으로 너무 약하다

아무것도 바꾸지 않은 조건(qdrop의 lost, brecq의 Top1_flip)도 RNG 재추첨만으로 평균이
기존 6-seed 범위를 벗어났다. **변경하지 않은 조건들을 음성 대조군으로 두고 그 드리프트
대비 배수로 판정할 것.** (§4.1의 AdaRound 예산 판정이 그 방식이다 — 약 7배.)

---

## 6. run 인덱스

| run | 내용 | 결론 |
|---|---|---|
| `91_baseline_confirmed_6seed` | 이전 확정값 (AdaRound iters 1000) | `runs/97`로 대체됨 |
| `92_env_recheck` | `runs/91` seed0 재현 (환경 재구축 검증) | baseline 4개 bit-identical, Combined만 다름 |
| `93_iters_recheck` | BRECQ iters 20000, 확정 설정 | 2000과 노이즈 수준 차이 → 2000 유지 |
| `94_adaround_confound` | AdaRound iters/observer 1-seed 예비 | §7이 대체 (1-seed 결론은 부분적으로 틀렸다) |
| `95_official_defaults` | 2단계 + batch2 + neck layer-wise | 2단계만 유의미하게 악화 |
| `96_clipfix_verify` | CLIP 수정 결과 중립성 스모크 | 수정 전후 지표 동일, 크기 9.88 MiB |
| **`97_ada_iters2000_6seed`** | **확정 baseline** (AdaRound iters 2000, min-max) | **§1 표의 출처** |
| `98_ada_mseobs_6seed` | AdaRound observer를 MSE로 통일한 6-seed | ablation. COCO↔LVIS 트레이드오프 확인 |

이전 `runs/85`~`88`은 커밋 `9125d1d`(weight 양자화를 채널별 비대칭 MSE로 교체) **이전**
코드에서 측정된 것이라 현재 코드 판단의 근거로 쓸 수 없다.
