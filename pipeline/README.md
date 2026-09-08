# pipeline/ — 우리 모델(Combined PTQ)을 돌리는 데 필요한 전체 코드

`promptcal-ptq` 저장소에는 진단/탐색용 스크립트가 40개 넘게 있다(`scripts/00_*`
~ `scripts/46_*`). 이 디렉토리는 그중 **논문에 실제로 쓰이는 결과(baseline
비교 + 제안 방법 Combined + 확장 평가지표)를 처음부터 재현하는 데 필요한
코드만** 한곳에 모은 것이다.

작동 원리를 개념적으로 설명한 문서는 저장소 루트의
[`PROMPTCAL_HOW_IT_WORKS.md`](../PROMPTCAL_HOW_IT_WORKS.md), 지금까지의 전체
결과 요약은 [`MASTER_SUMMARY.md`](../MASTER_SUMMARY.md)를 참고. 이 README는
"어느 파일이 무슨 역할을 하는가"에 집중한다.

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
│   │                            s_mult(연속 activation scale multiplier, 09-08부터
│   │                            conv당 스칼라가 아니라 in_channels별 벡터 --
│   │                            PROMPTCAL_CURRENT_MODEL.md §5.2/5.7 참고)도 정의돼 있음.
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
│   └── promptcal.py         -- 제안 방법. 핵심은
│                               optimize_promptcal_scale_neighbor(asymmetric=True):
│                               AdaRound로 확정한 weight 위에, 각 conv의 learnable
│                               continuous activation scale(s_mult)을 (1) S 프롬프트의
│                               top-k margin 보존 + (2) text-embedding 최근접 이웃의
│                               절대 유사도가 FP보다 커지는 방향만 억제(asymmetric
│                               hinge, collateral shift 대응) + (3) scale_reg_weight>0이면
│                               (s_mult-1)^2 정규화(09-07 LVIS 일반성 검증 이후 추가,
│                               확정값 10.0)로 최적화.
└── run_comparison.py        -- 실행 진입점. naive/AdaRound/QDrop/BRECQ/Combined
                                다섯 조건을 전부 빌드하고, AP(전체+S/H_eval subset)·
                                Top1_flip(표준)·H_eval_flip(masked, 우리 진단용)·
                                GT MRR/R@1/lost/gained/UPIR(실제 COCO GT 기준)·
                                calib 시간·이론적 모델 크기까지 전부 측정해서 표로 출력.
```

`scripts/45_baseline_compare.py`(같은 로직, `src/` import 경로만 다름)로 이미
낸 6-seed 결과가 `results_v1/diag/45_metrics_seed{0,1,2,4,5,7}_full.txt`와
`PromptCal_PTQ_progress_2026-09-06.md` §18/`PromptCal_PTQ_progress_2026-09-07.md`
에 정리돼 있다. `pipeline/run_comparison.py`는 그 스크립트를 이 디렉토리 하나로
독립시킨 사본이었다 — **단, 09-08~09-09에 `quant/adaround.py`·`quant/promptcal.py`를
현재 확정 설계(per-channel `s_mult` + `scale_reg_weight=10` 정규화)로
동기화하면서, 이 파일들만 놓고 보면 더 이상 위 결과(scalar `s_mult`, 정규화
없음)와 완전히 같은 숫자를 재현하지는 않는다.** 그 원래 6-seed 결과를 그대로
재현하려면 git 기록에서 09-08 이전 시점의 `src/quant/{adaround,promptcal}.py`를
체크아웃해서 넣어야 한다. 지금 버전으로 새로 돌리면 **현재 확정된 최종
방법**(공식 데이터 설정 6-seed 결과는 `PROMPTCAL_CURRENT_MODEL.md` §8.1)이
재현된다 — 다만 `run_comparison.py` 자체의 calibration/평가 데이터 소스는 아직
예전 방식(`val2017`에서 앞 32장을 calib, 그다음 500장을 probe로 슬라이스)
그대로다. 공식 데이터 설정(calib=train2017 256장, LVIS 평가=공식 minival)을
그대로 재현하려면 `scripts/58_full_baseline_official_data.py`를 참고할 것 —
이 디렉토리로의 이식은 아직 안 함(§유지 관리 메모).

## 실행 방법

```bash
conda activate promptcal
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle GPU> python pipeline/run_comparison.py \
    --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
    --data configs/coco_local.yaml --calib 32 --eval 500 --seed 2 --device 0 \
    --scale-reg-weight 10.0
```

- `--scale-reg-weight`: 기본값 10.0(09-08 official-data 6-seed 스윕으로 확정).
  0.0을 주면 정규화 없이(옛 동작) 돌릴 수 있음.

- `--model`: 저장소 루트의 `yolov8s-world.pt`(YOLO-World v1) 또는
  `yolov8s-worldv2.pt`. 지금까지 모든 6-seed 결과는 `yolov8s-world.pt` 기준.
- `--seed`: COCO-80 프롬프트를 S(40, calibration에 실제 사용)/H_cal(20,
  Combined는 안 씀)/H_eval(20, 평가 전용)로 나누는 난수 시드. 기존 6-seed
  비교는 `0 1 2 4 5 7`을 사용.
- GPU는 항상 `CUDA_DEVICE_ORDER=PCI_BUS_ID`와 함께 지정할 것 — 이 서버는
  `nvidia-smi` 인덱스와 CUDA 기본 열거 순서가 다르다(자세한 내용은
  `PROMPTCAL_HOW_IT_WORKS.md` §8.4).

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

이 디렉토리의 파일들은 `src/harness.py`·`src/quant/*.py`·
`scripts/45_baseline_compare.py`를 복사한 것이다(로직은 같고 import 경로만 이
디렉토리 기준으로 바뀜). 원본(`src/`, `scripts/`)이 계속 실험용으로 수정되는
반면 이 디렉토리는 "논문에 쓰이는 버전"으로 유지하는 게 목적이므로, 원본을
고친 뒤 그 변경을 여기에도 반영할지는 그때그때 판단 — 자동 동기화는 안 됨.

**동기화 이력**:
- 2026-09-07 최초 스냅샷(scalar `s_mult`, calib=32/val2017 슬라이스, 6-seed
  결과는 `results_v1/diag/45_metrics_seed{0,1,2,4,5,7}_full.txt`).
- 2026-09-09: `quant/adaround.py`(per-channel `s_mult`) ·
  `quant/promptcal.py`(`boundary_w` 노출, `scale_reg_weight` 정규화)를
  `src/quant/`의 최종 확정 버전으로 재동기화. `run_comparison.py`도
  `--scale-reg-weight`(기본 10.0)/`--boundary-w`(기본 3.0) 인자를 추가해서
  `build()`에 실제로 전달하도록 수정. **아직 안 한 것**: calibration/평가
  데이터 소스는 여전히 예전 방식(val2017 슬라이스)이고, 공식 데이터 설정
  (calib=train2017 256장, LVIS=공식 minival, `scripts/58_full_baseline_official_data.py`
  기준)으로는 이식 안 함 — 필요해지면 다음 동기화 대상.
- 최신 확정 하이퍼파라미터·공식 데이터 6-seed 결과의 단일 진실 공급원은
  저장소 루트의 `PROMPTCAL_CURRENT_MODEL.md`다. 이 README와 수치가 어긋나면
  그쪽을 따를 것.
