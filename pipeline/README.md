# pipeline/ — 저비트 PTQ(GPM) 실험 코드

**대상 독자:** 이 저장소에서 실험을 다시 돌리거나 코드를 고치는 사람.

**방법의 원리와 결과:** 루트 [`PROMPTCAL_METHOD_LOWBIT_2026-09-29.md`](../PROMPTCAL_METHOD_LOWBIT_2026-09-29.md)를 본다. 이 README는 "어떤 파일이 무엇을 하고, 어떻게 돌리는가"만 다룬다.

10-02 정리: 기각된 방향(vocab-metric)의 스크립트와 이전 README는 [`legacy/`](legacy/)로 옮겼다. 중복 모델 파일과 평가 출력은 `runs/_archive/pipeline_artifacts_2026-10-02/`(git 무시)로 옮겼다.

---

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `run_comparison.py` | **진입점.** FP·양자화 모델을 조건별로 빌드하고 COCO AP, LVIS AP, 판정 충실도(flip/lost)를 잰다 |
| `diag_w4_sensitivity.py` | ① 보호용 누수 없는 conv 진단(`rank_convs_leakfree`), 선택(`select_protected`), M 채택 사전 검사(`mcheck_flip`) |
| `harness.py` | 유사도/판정 측정 도우미 (`run_comparison`이 사용) |
| `scripts/protocol.sh` | **확정 프로토콜 공통 인자**(아래 §2). 작업자가 이 파일을 읽는다 |
| `scripts/worker.sh` | GPU 하나를 맡는 대기열 작업자(`<run_dir>/queue.tsv`) |
| `BASELINE_STATUS.md` | 기준선(BRECQ/QDrop/AdaRound/Combined) 충실도 기록 |
| `quant/fake_quant.py` | `QuantConv2d`: weight 출력 채널별 MSE, activation per-tensor MSE observer, 이전(`mig`), 채널별 비트 |
| `quant/quant_model.py` | conv 래핑, calibrate, 비트 지정(`set_first_last_bits`, `set_last_abits`, `set_conv_wbits` 등) |
| `quant/adaround.py`, `quant/brecq.py` | ③ AdaRound/BRECQ/QDrop 재구성. 16bit 이상 activation은 LSQ 학습 제외(`LSQ_MAX_BITS`) |
| `quant/fusion_quant.py` | ⓪ 텍스트 게이트 교환(`gate_commute_all`) |
| `quant/channel_graph.py`, `quant/migrate.py` | ② 공유 제약 scale 이전(채널 계보 추적, 공유 그룹, α 탐색) |
| `quant/attn_quant.py` | attention·contrastive matmul 8bit 양자화(`--attn-quant`) |
| `quant/promptcal.py`, `quant/semantic_calib.py`, `quant/pdquant.py`, `quant/vocab_metric.py` | Combined(PromptCal) 기준선, head 캡처, 어휘 임베딩 도우미 |

`src/quant/`는 `pipeline/quant/`의 사본이다. 고치면 같이 복사한다.

---

## 2. 확정 프로토콜 (2026-10-02)

`scripts/protocol.sh`:

```
--deterministic --calib 256 --no-skip-head --first-last-bits 8
--attn-quant attn_cls --last-abits 16
--combined-stage1 brecq --combined-recon-iters 2000
```

| 항목 | 설정 |
|---|---|
| 양자화 범위 | 모든 conv(head 포함). attention Linear/matmul과 head contrastive matmul도 8bit |
| 첫·마지막 레이어 | stem 첫 conv, head cv2/cv3 마지막 1×1은 W8A8 |
| head 마지막 conv 입력 | A16(`--last-abits 16`). m 모델의 seed 불안정 해결용으로, 모든 모델에 같게 적용한다 |
| FP로 남는 것 | LayerNorm, softmax, sigmoid, max, residual add, DFL, NMS |
| calibration | train2017 정렬 순서 앞 256장 |
| 보호 진단 | train2017 앞 200장, COCO 어휘. 캐시: `configs/protect_cache/<모델>_n200_c256_img640_la16_aqattn_cls.json` |
| 필수 환경 | `CUDA_DEVICE_ORDER=PCI_BUS_ID`. 한 비교의 모든 seed는 같은 GPU 종류(RTX 4000 Ada)에서 돌린다 |

---

## 3. 조건 이름

`<기반>+<접미사>`로 쓴다. 예: `brecq+GPM`. 기반은 `naive`, `brecq`, `qdrop`, `adaround`, `combined` 중 하나다.

| 접미사 | 의미 |
|---|---|
| `G` | ⓪ 텍스트 게이트 교환. 보호 목록에서 C2fAttn cv2를 뺀다 |
| `P` | ① 누수 없는 진단으로 고른 conv를 W8로 보호(예산 1.5%) |
| `M` | ② 공유 제약 scale 이전 |
| `A` | M을 사전 검사로 켤지 정한다. M 끔/켬 두 후보를 빌드하고 calibration 밖 COCO flip을 비교해, M 쪽이 1.5배를 넘으면 끈다. 빌드 시간은 약 2.2배 |
| `R` / `H` | 같은 예산의 무작위 보호 / HAWQ식(출력 MSE) 보호 (대조군) |

**모델별 최종 구성 (M 채택은 모델당 검사 1회로 결정, runs/154·156)**

| 모델 | 최종 구성 | M 결정 근거 |
|---|---|---|
| YOLOv8s-World | W4A8 `brecq+PM`, W4A6·W4A5 `brecq+GPM`, W8A8 `brecq+M` | 검사 통과(M 켬) |
| YOLOv8s-WorldV2 | `brecq+PM` / `brecq+GPM` | 검사 통과(M 켬) |
| YOLOv8m-World | `brecq+GP`. m은 보호 대상이 C2fAttn cv2뿐이라 `brecq+G`와 같다 | 검사 탈락(M을 켜면 flip 2~7배) |

조건마다 빌드 직전에 RNG를 복원한다. 그래서 한 run에 여러 조건을 묶어도, 따로 돌려도 결과가 bit 단위로 같다.

---

## 4. 실행

```bash
cd /home/taeho/promptcal-ptq
source pipeline/scripts/protocol.sh

# 한 번 실행
.venv/bin/python pipeline/run_comparison.py $PROTOCOL --model yolov8s-world.pt \
    --w-bits 4 --a-bits 5 --conditions naive,brecq,brecq+PM,brecq+GPM --seed 0 --device 4

# 대기열: runs/<이름>/queue.tsv에 "태그<TAB>인자"를 한 줄씩 쓰고 GPU마다 작업자를 띄운다
pipeline/scripts/worker.sh runs/<이름> 4 0-23 &
pipeline/scripts/worker.sh runs/<이름> 5 24-47 &
```

- **진단 훅:** `--post-build <스크립트>`를 주면 빌드 직후, 평가 전에 그 스크립트를 실행하고 끝난다. 스크립트에서 `models`, `fp`, `args`, `device`를 쓸 수 있다. 예: `runs/155_mcheck/calib_check.py`.
- **메모리:** m 모델은 한 run에 조건 2개까지만 묶는다. 20GB GPU에서 조건 3개는 메모리 한계에 걸린다.
- **소요 시간 (RTX 4000 Ada):** s 조건 하나 빌드에 약 20분, m은 약 23분이다. COCO와 LVIS 평가는 조건당 약 5분이다.

---

## 5. 주의 (과거 사고)

- `--device N`과 nvidia-smi 번호는 `CUDA_DEVICE_ORDER=PCI_BUS_ID`일 때만 같다.
- 16bit activation에 LSQ를 걸면 step이 발산한다. `LSQ_MAX_BITS`가 막고 있으니 지우지 말 것(runs/155).
- 보호 진단 캐시는 프로토콜(`_la16_aqattn_cls`)별로 따로 저장된다. 프로토콜을 바꾸면 새 캐시가 생긴다.
