# pipeline/ — 우리 모델(Combined PTQ)을 돌리는 데 필요한 전체 코드

`promptcal-ptq` 저장소에는 진단/탐색용 스크립트가 60개 넘게 있다(`scripts/00_*`
~ `scripts/61_*`). 이 디렉토리는 그중 **논문에 실제로 쓰이는 결과(baseline
비교 + 제안 방법 Combined + 확장 평가지표, 공식 데이터 설정)를 처음부터
재현하는 데 필요한 코드만** 한곳에 모은 것이다.

작동 원리를 개념적으로 설명한 문서는 저장소 루트의
[`PROMPTCAL_HOW_IT_WORKS.md`](../PROMPTCAL_HOW_IT_WORKS.md), **현재 확정된
최종 설계·하이퍼파라미터·성능 결과의 단일 진실 공급원**은
[`PROMPTCAL_CURRENT_MODEL.md`](../PROMPTCAL_CURRENT_MODEL.md)다. 이 README는
"어느 파일이 무슨 역할을 하는가"에 집중한다 — 수치를 인용할 땐 항상
`PROMPTCAL_CURRENT_MODEL.md`를 우선한다.

## 파일 지도

```
pipeline/
├── harness.py              -- SimilarityHarness: FP32/양자화 모델에 forward hook을
│                               걸어 WorldDetect.cv4(ContrastiveHead)의 region-prompt
│                               유사도 행렬 [anchors, prompts]을 캡처. 이 저장소 모든
│                               측정(AP 제외)의 공통 기반.
├── quant/
│   ├── fake_quant.py        -- naive W8A8. QuantConv2d(weight per-channel symmetric
│   │                            + activation per-tensor asymmetric int8 fake-quant),
│   │                            ActObserver(calibration으로 activation min/max 수집).
│   ├── quant_model.py       -- wrap_convs(모델의 모든 Conv2d를 QuantConv2d로 교체),
│   │                            calibrate(calibration 이미지로 activation scale 확정).
│   ├── adaround.py          -- AdaRound(학습 가능한 weight rounding, alpha) +
│   │                            QDrop(qdrop_prob로 확률적 activation drop, 같은 함수의
│   │                            옵션). AdaRoundQuantConv2d에 Combined가 쓰는
│   │                            s_mult(연속 activation scale multiplier, conv당
│   │                            스칼라가 아니라 in_channels별 벡터 --
│   │                            PROMPTCAL_CURRENT_MODEL.md §5.2 참고)도 정의돼 있음.
│   ├── brecq.py             -- BRECQ(block-wise joint reconstruction, C2fAttn 등
│   │                            다중 입력 블록 지원).
│   ├── pdquant.py           -- _find_head/_CV4Capture 헬퍼(promptcal.py가 사용).
│   │                            optimize_pdquant 자체는 현재 5-way 비교에 포함 안 됨
│   │                            (PD-Quant는 v1 시절 baseline, 지금 조건에서는 제외).
│   ├── semantic_calib.py    -- get_txt_feats/text_neighbor_order(text embedding
│   │                            최근접 이웃 계산, Combined의 neighbor preservation이
│   │                            사용) + utility_refinement_terms(§4.3 utility
│   │                            constraint, 현재 기본 비교에서는 미사용이지만
│   │                            promptcal.py의 utility 변형 함수가 참조).
│   └── promptcal.py         -- 제안 방법. 핵심은 optimize_promptcal_scale_neighbor:
│                               AdaRound로 확정한 weight 위에, 각 conv의 per-channel
│                               learnable activation scale(s_mult)을 (1) S 프롬프트의
│                               top-k margin 보존 + (2) H_cal에도 동일 margin_loss
│                               직접 적용(cal_weight) + (3) text-embedding 최근접
│                               이웃(H_eval 제외)의 절대 유사도가 FP보다 커지는
│                               방향만 억제(asymmetric hinge) + (4) (s_mult-1)^2
│                               정규화(scale_reg_weight)로 최적화.
└── run_comparison.py        -- 실행 진입점(`scripts/58_full_baseline_official_data.py`
                                포팅, 09-15). naive/AdaRound/QDrop/BRECQ/Combined
                                다섯 조건을 공식 데이터 설정(calib=train2017,
                                COCO-80 평가=val2017 전체, LVIS 평가=공식 minival)으로
                                빌드하고, COCO_AP(+S/H_eval subset)·LVIS_AP(+APr/APc/APf)·
                                Heval_flip(masked)·Top1_flip(표준)·GT_MRR/R@1·
                                lost/gained/lateral/corrective_rate·UPIR(COCO·LVIS
                                양쪽)·lost의 S/H_cal/H_eval 그룹별 분해·calib 시간·
                                이론적 모델 크기까지 전부 측정해서 표로 출력.
```

## 실행 방법

```bash
conda activate promptcal
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle GPU> python pipeline/run_comparison.py \
    --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
    --data configs/coco_local.yaml \
    --lvis-ann /data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json \
    --calib 256 --seed 0 --device 0
```

스모크 테스트(몇 분 안에 파이프라인 정상 동작만 확인, `--eval-cap`으로 probe
수 제한):

```bash
... --calib 8 --eval-cap 16 --iters 30 --recon-iters-ada 20 --recon-iters-strong 20 --seed 0 --device 0
```

- **데이터 설정(공식, 09-08 확정)**: calibration은 COCO **train2017**에서
  `--calib`장(기본 256, 평가 데이터와 완전 분리). COCO-80 평가는 val2017
  **전체**(5000장, `--eval-cap`으로 제한 안 하면). LVIS 평가는 공식
  `lvis_v1_minival.json`(ultralytics 공식 배포) — 확인 결과 "COCO val2017 ∩
  LVIS val"과 정확히 일치하는 4809장.
- `--scale-reg-weight`(기본 10.0)·`--cal-weight`(기본 1.0): 둘 다 확정값
  (`PROMPTCAL_CURRENT_MODEL.md` §5.5). `--cal-weight 0.0`을 주면 H_cal 직접
  보호를 끈 이전 동작으로 돌아감.
- `--model`: 저장소 루트의 `yolov8s-world.pt`(YOLO-World v1) 또는
  `yolov8s-worldv2.pt`. 지금까지 모든 확정 결과는 `yolov8s-world.pt` 기준.
- `--seed`: COCO-80 프롬프트를 S(40, margin_loss 직접 대상)/H_cal(20,
  cal_weight로 margin_loss 직접 대상 + neighbor-hinge 후보 풀)/H_eval(20,
  최적화에 전혀 안 씀, 평가 전용)로 나누는 난수 시드. 확정 6-seed 비교는
  `0 1 2 3 4 5`를 사용(예전 `PROMPTCAL_HOW_IT_WORKS.md` 시절 문서의
  `0 1 2 4 5 7`과는 다른 세트이니 혼동하지 말 것).
- GPU는 항상 `CUDA_DEVICE_ORDER=PCI_BUS_ID`와 함께 지정할 것 — 이 서버는
  `nvidia-smi` 인덱스와 CUDA 기본 열거 순서가 다르다(자세한 내용은
  `PROMPTCAL_HOW_IT_WORKS.md` §8.4). 공유 서버이므로 실행 전 항상
  `nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv`로
  다른 사용자 프로세스가 없는 GPU인지 확인할 것(메모리 0만 보고 판단하지 말 것).
- LVIS(P=1203)는 probe 전체를 sim 행렬 리스트로 들고 있으면 CPU RAM이
  터지므로(4809장 기준 이론상 ~194GB) `compute_lvis_flip_gt_streaming`이
  이미지 한 장씩 스트리밍 처리한다 — COCO-80 쪽(P=80)은 여전히 리스트 캐시
  방식(메모리 부담이 훨씬 작음).

## 이 디렉토리에 없는 것 (의도적으로 제외)

- **v1(최초 PromptCal, `src/quant/promptcal.py`의 `optimize_promptcal`,
  margin+decision CE+reg로 AdaRound alpha 최적화)**: 6-seed 재검증 결과 새
  지표로도 naive보다 나은 게 없어 폐기가 재확인된 방법(`PromptCal_PTQ_progress_
  2026-09-07.md` §6). 재현하려면 `scripts/20_promptcal_minimal.py` /
  `scripts/46_v1_metrics_check.py` 참고.
- **PD-Quant(`quant/pdquant.py`의 `optimize_pdquant`)**: v1 시절 baseline
  비교(`PROGRESS_v1_D.md`)에서만 쓰였고 지금 5-way 비교에는 포함 안 됨. 헬퍼
  함수(`_find_head`/`_CV4Capture`)만 `promptcal.py`가 재사용 중이라 파일 자체는
  남겨둠.
- **Utility-Constrained Refinement 변형(`optimize_promptcal_scale_neighbor_utility`,
  `promptcal.py`에 있지만 `run_comparison.py`가 호출 안 함)**: 검증 결과 기본
  가중치에서 중립적(뚜렷한 추가 이득 없음, `scripts/44_utility_ap_check.py`).
- 실제 하드웨어(TFLite/TensorRT) INT8 엔진 export/latency 측정: 별도 후속
  과제로 보류 중(`MASTER_SUMMARY.md` §8 국면 H).

## 유지 관리 메모

`quant/*.py`·`harness.py`는 `src/quant/*.py`·`src/harness.py`의 사본이고,
`run_comparison.py`는 `scripts/58_full_baseline_official_data.py`의 사본이다
(로직은 같고 import 경로만 이 디렉토리 기준으로 바뀜). 원본(`src/`, `scripts/`)이
계속 실험용으로 수정되는 반면 이 디렉토리는 "논문에 쓰이는 버전"으로 유지하는
게 목적이므로, 원본을 고친 뒤 그 변경을 여기에도 반영할지는 그때그때 판단 —
자동 동기화는 안 됨. **동기화 여부를 확인하려면 `diff src/quant/*.py
pipeline/quant/*.py`처럼 직접 diff를 떠서 확인할 것 — 이 문서의 "동기화 이력"
서술이 실제 diff보다 늦게 갱신될 수 있다(09-15에 실제로 그랬음, 아래 참고).**

**동기화 이력**:
- 2026-09-07 최초 스냅샷(scalar `s_mult`, calib=32/val2017 슬라이스, 6-seed
  결과는 `results_v1/diag/45_metrics_seed{0,1,2,4,5,7}_full.txt` —
  `PROMPTCAL_CURRENT_MODEL.md` §8.4가 지금 결과와 이걸 헷갈리지 말라고 경고함).
- 2026-09-09~09-12: `quant/adaround.py`(per-channel `s_mult`) ·
  `quant/promptcal.py`(`boundary_w` 노출, `scale_reg_weight` 정규화, H_eval을
  neighbor 후보 풀에서 제외하는 버그 수정, `cal_idx`/`cal_weight` H_cal 직접
  보호 추가)를 `src/quant/`와 동기화. **이 문서(README)는 이 구간의 싱크를
  09-09분까지만 기록해뒀었는데, 09-15에 `diff src/quant/*.py
  pipeline/quant/*.py`로 직접 확인해보니 실제로는 09-12분(H_eval 버그 수정 +
  cal_weight)까지 이미 다 반영돼 있었다** — 코드는 맞았고 기록만 밀려 있었던
  것. 앞으로는 이 문서의 서술보다 실제 diff를 신뢰할 것.
- **2026-09-15**: `run_comparison.py`를 `scripts/58_full_baseline_official_data.py`
  기준으로 전면 교체 — 이전까지는 quant/ 엔진만 최신이고 데이터 소스는 여전히
  val2017 슬라이스(calib 앞부분)에 LVIS 평가 자체가 아예 없었다. 이제
  calib=train2017/평가=val2017 전체+공식 LVIS minival, `--cal-weight`(기본
  1.0)·`--scale-reg-weight`(기본 10.0) 확정값, LVIS AP(+APr/APc/APf)/LVIS
  flip/GT/lost 스트리밍 계산까지 전부 반영됨. `--eval-cap` 스모크 테스트로
  end-to-end 정상 동작 확인 완료(GPU 7, calib=8/eval-cap=16).
- 최신 확정 하이퍼파라미터·공식 데이터 6-seed 결과의 단일 진실 공급원은
  저장소 루트의 `PROMPTCAL_CURRENT_MODEL.md`다. 이 README와 수치가 어긋나면
  그쪽을 따를 것.
