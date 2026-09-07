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
│   │                            s_mult(연속 activation scale multiplier)도 정의돼 있음.
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
│                               hinge, collateral shift 대응)로 최적화.
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
독립시킨 사본이다 — 새로 실행해도 같은 결과가 나와야 한다(같은 파일, seed,
하이퍼파라미터).

## 실행 방법

```bash
conda activate promptcal
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle GPU> python pipeline/run_comparison.py \
    --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
    --data configs/coco_local.yaml --calib 32 --eval 500 --seed 2 --device 0
```

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
`scripts/45_baseline_compare.py`를 2026-09-07 시점 상태로 복사한 것이다(로직은
같고 import 경로만 이 디렉토리 기준으로 바뀜). 원본(`src/`, `scripts/`)이 계속
실험용으로 수정되는 반면 이 디렉토리는 "논문에 쓰인 버전"으로 고정해두는 게
목적이므로, 원본을 고친 뒤 그 변경을 여기에도 반영할지는 그때그때 판단 --
자동 동기화는 안 됨.
