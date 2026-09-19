# pipeline/ — 우리 모델(Combined PTQ)을 돌리는 데 필요한 전체 코드

`promptcal-ptq` 저장소에는 진단/탐색용 스크립트가 60개 넘게 있다(`scripts/00_*`
~ `scripts/61_*`). 이 디렉토리는 그중 **논문에 실제로 쓰이는 결과(baseline
비교 + 제안 방법 Combined + 확장 평가지표, 공식 데이터 설정)를 처음부터
재현하는 데 필요한 코드만** 한곳에 모은 것이다.

작동 원리를 개념적으로 설명한 문서는 저장소 루트의
[`PROMPTCAL_HOW_IT_WORKS.md`](../PROMPTCAL_HOW_IT_WORKS.md), **현재 확정된
최종 설계·하이퍼파라미터·성능 결과의 단일 진실 공급원**은 **09-17부터**
[`PROMPTCAL_CURRENT_MODEL_V2.md`](../PROMPTCAL_CURRENT_MODEL_V2.md)다(v1인
`PROMPTCAL_CURRENT_MODEL.md`는 per-channel 시절 전체 서사와 하이퍼파라미터
스윕 10개 절의 역사적 기록으로 보존됨 — v2가 그 결론만 깔끔하게 반영).
이 README는 "어느 파일이 무슨 역할을 하는가"에 집중한다 — 수치를 인용할 땐
항상 `PROMPTCAL_CURRENT_MODEL_V2.md`를 우선한다.

**09-16에 §8.1이 per-tensor `s_mult` + identity-aware `margin_loss`
설계로 갱신됐다** — 둘 다 이제 `run_comparison.py`의 **기본 동작**이다
(claim4/claim5, `--no-smult-per-tensor`/`--no-identity-aware-margin`으로
이전 설계로 되돌릴 수 있음). H_eval anchor-선정 리크 수정(claim1)은 opt-in도
아니고 항상 적용되는 버그 수정이다. 이 변경들의 전체 검증 경위·claim
2/3/6~10 등 아직 미확정인 부분은
[`PROMPTCAL_CLAIMS_2026-09-15.md`](../PROMPTCAL_CLAIMS_2026-09-15.md)에
정리돼 있다.

**09-18**: `adaround.py`/`brecq.py`의 재구성 loss 정규화가 공식 BRECQ 대비
축소돼 있던 버그가 발견·수정됐다(claim13) — `optimize_adaround`가 Combined
자신의 weight-rounding 단계에도 쓰이므로 baseline 4개뿐 아니라 Combined의
COCO_AP/S_AP/H_eval_AP까지 영향받았다. 6-seed 재측정 완료
(`runs/71_recon_fix_review/`) — AdaRound/QDrop/BRECQ는 여전히 naive보다
COCO_AP가 낮고(정확히 구현해도 그렇다는 게 확인된 실제 결과, 버그 아님),
decision-preservation 4개 지표는 이제 QDrop·BRECQ가 Combined를 이긴다.

**09-18 (이어서, claim14, 최신)**: claim13으로 Combined 자신의 1단계
(`optimize_adaround`)도 alpha를 크게 움직이게 됐는데, 그 목적함수가
2단계(margin_loss/s_mult)가 보호하는 영역과 무관해서 그 바깥으로 손상이
샌다는 게 확인돼, **1단계를 아예 생략(`--combined-recon-iters 0`, 이제
기본값)** 하는 걸로 확정했다 — round-to-nearest weight + margin_loss만으로
COCO_AP/LVIS_AP/APr/Heval_flip/LVIS_lost 5개 지표가 트레이드오프 없이
동시에 개선됐다(6-seed, `runs/72_combined_recon_diag/`). Combined는 이제
AdaRound 메커니즘을 전혀 안 쓴다. `PROMPTCAL_CURRENT_MODEL_V2.md` §4/§5.1/§8
갱신 완료.

**09-19 (claim15, 최신)**: §8 표가 **불공정 비교**였다는 게 드러났다 —
QDrop/BRECQ의 activation scale 학습(LSQ)이 꺼진 채로 Combined와 비교하고
있었다(사용자 지적). QDrop/BRECQ는 원 논문대로 LSQ가 기본 적용되도록
고치고(`--qdrop-brecq-learn-act-scale` 기본 True, AdaRound는 원 논문에
없어서 기본 False 유지), 그 공정 비교에서 `scale_reg_weight`를 10.0→1.0으로
재튜닝했다. 결과: Combined는 9개 지표 중 **4개(COCO_AP/LVIS_AP/APr/
LVIS_lost)만 BRECQ+LSQ를 이기고 나머지 5개는 아직 진다.** 그 외
`--w-bits`/`--a-bits`(bit-width 실험용), `--combined-stage1`(BRECQ
block-wise를 1단계로 쓰는 진단 실험용) 추가, `quant_weight()`의 죽은 alpha
재계산 캐싱. `PROMPTCAL_CURRENT_MODEL_V2.md` §4/§5.5/§8/§9 갱신 완료.

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
│   ├── adaround.py          -- AdaRound(학습 가능한 weight rounding, alpha).
│   │                            AdaRoundQuantConv2d에 Combined가 쓰는
│   │                            s_mult(연속 activation scale multiplier, conv당
│   │                            스칼라가 아니라 in_channels별 벡터 --
│   │                            PROMPTCAL_CURRENT_MODEL_V2.md §5.2 참고)도 정의돼 있음.
│   │                            09-15: channelwise_smult(기본 True) 추가 --
│   │                            False면 s_mult을 baseline과 동일한 per-tensor
│   │                            스칼라로 강제(claim4 대조 실험, opt-in).
│   │                            09-17: learn_act_scale(기본 False) 추가 -- True면
│   │                            AdaRound/QDrop/BRECQ에도 s_mult를 켜서 alpha와
│   │                            같은 reconstruction loss로 공동 최적화(claim12,
│   │                            원 BRECQ 논문의 AdaRound+LSQ 공동 최적화 재현).
│   │                            09-18: 재구성 loss를 공식 BRECQ의 lp_loss와 동일한
│   │                            정규화(lp_rec_loss, 채널축 sum)로 교체하고
│   │                            lr/reg_weight/warmup을 공식값(1e-3/0.01/0.2)으로
│   │                            수정(claim13 -- 이전 정규화는 공식 대비 C_out배
│   │                            작아서 rounding 정규화가 압도, alpha가 사실상
│   │                            round-to-nearest에서 못 움직였음). flip_rate()
│   │                            진단(nearest 대비 반올림이 실제로 바뀐 비율) 추가.
│   ├── brecq.py             -- BRECQ(block-wise joint reconstruction, C2fAttn 등
│   │                            다중 입력 블록 지원). 09-18: QDrop이 여기로 이전됨
│   │                            (`optimize_brecq(qdrop_prob=...)`) -- 이전엔
│   │                            adaround.py의 layer-wise 경로에 붙어있어 QDrop
│   │                            원 논문의 재구성 단위(block-wise)와 달랐다(claim13).
│   │                            block 내부 quantizer drop + block 입력 drop(input_prob)
│   │                            둘 다 구현. learn_act_scale도 adaround.py와 동일하게
│   │                            지원(claim12).
│   ├── pdquant.py           -- _find_head/_CV4Capture 헬퍼(promptcal.py가 사용).
│   │                            optimize_pdquant 자체는 현재 5-way 비교에 포함 안 됨
│   │                            (PD-Quant는 v1 시절 baseline, 지금 조건에서는 제외).
│   ├── semantic_calib.py    -- get_txt_feats/text_neighbor_order(text embedding
│   │                            최근접 이웃 계산, Combined의 neighbor preservation이
│   │                            사용) + utility_refinement_terms(§4.3 utility
│   │                            constraint, 현재 기본 비교에서는 미사용이지만
│   │                            promptcal.py의 utility 변형 함수가 참조).
│   └── promptcal.py         -- 제안 방법. 핵심은 optimize_promptcal_scale_neighbor:
│                               weight(09-18 claim14부터 round-to-nearest --
│                               AdaRound 1단계 생략, 아래 run_comparison.py 참고)
│                               위에, 각 conv의 per-channel
│                               learnable activation scale(s_mult)을 (1) S 프롬프트의
│                               top-k margin 보존 + (2) H_cal에도 동일 margin_loss
│                               직접 적용(cal_weight) + (3) text-embedding 최근접
│                               이웃(H_eval 제외)의 절대 유사도가 FP보다 커지는
│                               방향만 억제(asymmetric hinge) + (4) (s_mult-1)^2
│                               정규화(scale_reg_weight)로 최적화.
│                               09-15: (a) confident-anchor 선정을 80열 전체가
│                               아니라 train_cols(S∪H_cal)로 제한(claim1 버그
│                               수정, 항상 적용). (b) margin_loss에
│                               identity_aware(기본 False) 추가 -- True면
│                               fp_idx로 sim_q를 gather해서 class identity
│                               고정(claim5 대조 실험, opt-in).
│                               09-16: identity_aware=True가 FP top-(k+1) 밖의
│                               class가 Q에서 치솟는 경우(intrusion)를 아예 못
│                               보던 것 수정 -- 마지막 열(boundary)만 "top-k 밖
│                               전체 최댓값"으로 바꿔서 swap과 intrusion을 둘 다
│                               탐지(claim5-b). 이 수정 전에 돌린
│                               identity-aware 6-seed 결과(아래 실행 이력 참고)는
│                               재현하려면 재실행 필요.
└── run_comparison.py        -- 실행 진입점(`scripts/58_full_baseline_official_data.py`
                                포팅, 09-15). naive/AdaRound/QDrop/BRECQ/Combined
                                다섯 조건을 공식 데이터 설정(calib=train2017,
                                COCO-80 평가=val2017 전체, LVIS 평가=공식 minival)으로
                                빌드하고, COCO_AP(+S/H_eval subset)·LVIS_AP(+APr/APc/APf)·
                                Heval_flip(masked)·Top1_flip(표준)·GT_MRR/R@1·
                                lost/gained/lateral/corrective_rate·UPIR(COCO·LVIS
                                양쪽)·lost의 S/H_cal/H_eval 그룹별 분해·calib 시간·
                                이론적 모델 크기까지 전부 측정해서 표로 출력.
                                09-16: --smult-per-tensor(claim4)·
                                --identity-aware-margin(claim5) 플래그 추가,
                                이후 둘 다 §8.1 확정 설계로 채택되면서 기본값을
                                True로 전환(--no-* 로 이전 설계로 되돌릴 수 있음).
                                measure_ap의 클래스별 AP 매핑 버그(claim7)·
                                switch_vocab의 names/predictor 미갱신(claim8)
                                수정. --eval-cap이 COCO_AP/S_AP/H_eval_AP에는
                                적용 안 된다는 안내 문구 추가(claim10, LVIS AP는
                                적용됨) -- 자세한 내용은 PROMPTCAL_CLAIMS_2026-09-15.md.
                                09-16 추가: --cal-weight 0일 때 cal_idx까지
                                None으로 넘겨서 anchor 풀(train_cols)이 같이
                                줄어들던 confound 수정(claim5-a, cal_idx는 이제
                                항상 전달). main() 시작 시 print(vars(args))로
                                실행 플래그 로그에 남김, --conditions(쉼표 구분,
                                기본 5개 전부)로 Combined 변형만 볼 때 QDrop/BRECQ
                                재빌드 생략 가능(claim5-c).
                                09-17: --control-mse 추가(claim12) -- Combined
                                빌드 시 margin_loss/neighbor-hinge를 전부 끄고
                                s_mult만 순수 MSE reconstruction으로 최적화하는
                                control 실험(손잡이 존재 자체의 효과 분리용).
                                --learn-act-scale 추가(claim12) -- AdaRound/QDrop/
                                BRECQ 세 baseline에도 s_mult 공동 최적화를 추가
                                (naive/combined는 영향 없음, s_mult는 자동으로
                                per-tensor 강제).
                                09-18: QDrop 조건이 optimize_brecq(qdrop_prob=...)를
                                쓰도록 변경(claim13, 이전엔 layer-wise 경로라
                                brecq 조건과 재구성 단위가 달랐음).
                                09-18 (이어서, claim14): --combined-recon-iters
                                추가하고 기본값 0으로 확정 -- Combined의 1단계
                                (optimize_adaround)를 생략(round-to-nearest
                                weight)하는 게 트레이드오프 없이 5개 지표를
                                동시에 개선함을 6-seed로 확인. build()의
                                combined_recon_iters 기본값도 0.
                                09-19 (claim15, 최신): --learn-act-scale(단일
                                플래그)를 --adaround-learn-act-scale(기본 False)·
                                --qdrop-brecq-learn-act-scale(기본 True)로 분리 --
                                QDrop/BRECQ는 이제 플래그 없이도 원 논문대로 LSQ가
                                기본 적용됨(claim12를 옵트인으로 방치했던 실수
                                수정). --w-bits/--a-bits 추가(하드코딩된 8/8
                                해소, 메커니즘 실험용). --combined-stage1
                                (none/adaround/brecq) 추가 -- BRECQ의 block-wise
                                재구성을 Combined 1단계로 쓰는 진단 실험(제안
                                방법 변경 아님). --scale-reg-weight 기본값
                                10.0→1.0(공정 비교 기준 재스윕 결과).
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
- `--scale-reg-weight`(기본 **1.0**, 09-19 claim15로 10.0에서 하향)·
  `--cal-weight`(기본 1.0): 둘 다 확정값(`PROMPTCAL_CURRENT_MODEL_V2.md`
  §5.5). `--cal-weight 0.0`을 주면 H_cal 직접 보호를 끈 이전 동작으로
  돌아감. `scale_reg_weight`의 10.0은 claim14 이전(1단계가 AdaRound-refined
  weight였던 시절) 튜닝값이라 1단계가 naive로 바뀐 뒤(claim14)엔 과도한
  정규화였음 — 공정 비교(claim15, QDrop/BRECQ+LSQ 기준) 재스윕 결과 1.0이
  최적 구간(0.0/0.5는 LVIS_flip 악화).
- **`--smult-per-tensor`·`--identity-aware-margin`(09-16, 둘 다 §8.1 확정
  설계라 기본값 True)**: `argparse.BooleanOptionalAction`이라
  `--no-smult-per-tensor`/`--no-identity-aware-margin`으로 끌 수 있음 —
  이전(§8.11) 확정값인 per-channel `s_mult`·identity-unaware margin_loss로
  되돌리는 방법. `--smult-per-tensor`는 Combined의 `s_mult`을 baseline과
  동일한 per-tensor 스칼라로 쓰게 함(claim4, activation quantization
  granularity 공정성 + 표준 INT8 엔진 배포 가능성). `--identity-aware-margin`은
  margin_loss가 class identity를 무시하던 blind spot을 막음(claim5,
  claim5-b로 intrusion 탐지까지 보강). 즉 **위 "실행 방법" 예시 커맨드는
  플래그 추가 없이 그대로 §8.1 확정 설계를 재현**한다. 자세한 검증 경위는
  `PROMPTCAL_CLAIMS_2026-09-15.md` 참고.
- **`--control-mse`(09-17, claim12, 기본 False — 진단/실험용, Combined
  전용)**: margin_loss 대신 순수 MSE로 s_mult만 최적화. "Combined 우위가
  손잡이 존재 자체에서 오는가"를 분리 측정하는 용도, 확정 §8 표에는 안 들어감.
- **`--adaround-learn-act-scale`(기본 False)·`--qdrop-brecq-learn-act-scale`
  (기본 **True**, 09-19 claim15로 확정 — claim12를 09-17에 opt-in
  `--learn-act-scale`(기본 False, 단일 플래그)로 방치했던 걸 수정)**:
  QDrop/BRECQ 원 논문(Wei et al. ICLR'22 / Li et al. ICLR'21)은 activation
  scale 학습(LSQ)을 포함하고 AdaRound 원 논문(Nagel et al. ICML'20)은
  없다 — 그래서 QDrop/BRECQ만 기본으로 켠다. **이게 이제 확정 §8 표
  기준이다** — 이전엔 baseline 전부 LSQ가 꺼진 채로 Combined와 비교해서
  "방법 차이"가 아니라 "구현 축소와의 비교"였다(사용자 지적, claim15).
  `--no-qdrop-brecq-learn-act-scale`로 이전(claim12 이전, 축소 구현)
  동작으로 되돌릴 수 있음(ablation 목적, 확정 비교표엔 쓰지 말 것).
- **09-18(claim13) 재구성 loss 정합성 수정**: `adaround.py`/`brecq.py`의
  재구성 loss(공식 BRECQ `lp_loss`와 동일 정규화로 교체) · lr(1e-3) ·
  reg_weight(0.01) · warmup(0.2)이 전부 바뀌었다 — 이 CLI 플래그로 조정하는
  값이 아니라 라이브러리 기본값 자체가 바뀐 것이라, baseline 4개(naive
  제외)의 §8 수치가 이 수정 이후 영구적으로 달라졌다(6-seed 재측정 완료,
  `PROMPTCAL_CLAIMS_2026-09-15.md` claim13 참고).
- **`--combined-recon-iters`(09-18, claim14, 기본값 `0` — §8 확정 설계)**:
  claim13으로 Combined 자신의 1단계(`optimize_adaround`)도 alpha를 크게
  움직이게 됐는데, 그 목적함수가 2단계(margin_loss/s_mult)가 보호하는 영역
  밖으로 손상을 샌다는 게 확인돼 **1단계를 기본적으로 생략**한다(값을
  0보다 크게 주면 이전처럼 1단계를 되살릴 수 있음, opt-in). 자세한 경위는
  `PROMPTCAL_CLAIMS_2026-09-15.md` claim14 참고. **위 "실행 방법" 예시
  커맨드는 플래그 추가 없이 지금 §8(claim15까지 반영된 최신 버전)을 그대로
  재현**한다 — claim13 직후 한동안 재현 안 됐다가 claim14로 원복, claim15
  (LSQ 기본 적용 + scale_reg_weight=1.0)도 전부 기본값이라 여전히 재현됨.
- **`--combined-stage1`(none/adaround/brecq, 기본 `none` — 09-19 claim15,
  진단 전용, 제안 방법 아님)**: Combined의 1단계로 뭘 쓸지. `brecq`는 BRECQ의
  block-wise 재구성(alpha만, LSQ는 안 켬)을 1단계로 써서, "BRECQ+LSQ가
  margin_loss 없이도 decision-preservation을 이기는 게 block-wise 상관
  반영 때문인지 margin_loss가 그 위에 추가 기여를 하는지" 분리하는 통제
  실험용. 헤드라인 설계는 계속 `none`.
- **`--w-bits`/`--a-bits`(기본 8/8, 09-19 claim15)**: 이전엔
  `wrap_convs(m.model, 8, 8)`이 하드코딩돼 있어서 bit-width를 CLI로 조정할
  방법이 없었다. W8A32/W32A8 같은 조합으로 손상이 weight rounding 쪽인지
  activation range 쪽인지 분리하는 메커니즘 실험에 씀 — 스모크 테스트
  (calib=32)에서 naive의 FP32 대비 손상이 W8A32는 -0.06, W32A8은 -1.01로
  activation 쪽이 압도적임을 확인(정식 스케일 재측정은 아직 안 함).
- `--eval-cap`은 COCO_AP/S_AP/H_eval_AP(`measure_ap`가 `--data` yaml의 고정
  val split을 씀)에는 적용 안 되고, LVIS_AP/APr/APc/APf와 flip/GT/UPIR/lost
  등 나머지 전부에는 적용됨 — 실행 시 표 위에 이 안내가 자동 출력됨(claim10).
- `--conditions`(쉼표 구분, 기본 `naive,adaround,qdrop,brecq,combined`):
  Combined 변형만 튜닝/확인할 때 `--conditions naive,combined`로 QDrop/BRECQ
  재빌드(seed당 900~1400s대)를 생략할 수 있다 — 단, baseline 결과를 다른
  로그에서 재사용하려면 **같은 seed** 로그여야 한다(S_AP/H_eval_AP/Heval_flip/
  UPIR/lost_rate_by_group은 seed마다 partition이 바뀌어 값이 달라짐).
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
- **2026-09-16**: claim 1~10(리뷰어 공격 포인트 점검) 검증/수정 반영,
  커밋 `87157e5`. `src/quant/promptcal.py`의 confident-anchor 선정을
  train_cols(S∪H_cal)로 제한하는 버그 수정(claim1, 항상 적용)이
  `pipeline/quant/promptcal.py`에 동기화됨. `channelwise_smult`(claim4)·
  `identity_aware_margin`(claim5) opt-in 파라미터가 `adaround.py`/
  `promptcal.py`에 추가되고 `run_comparison.py`에 `--smult-per-tensor`/
  `--identity-aware-margin` 플래그로 노출됨. `measure_ap`의 클래스별 AP
  매핑 버그(claim7)와 `switch_vocab`의 names/predictor 미갱신(claim8)도
  수정. 전부 opt-in이거나(claim4/5) 항상-바른-방향인 버그 수정(claim1/7/8)이라
  기존 확정 결과 재현성은 안 깨짐. 실험 결과·6-seed 검증·미결정 사항은
  `PROMPTCAL_CLAIMS_2026-09-15.md`에 별도 정리.
- **2026-09-16 (이어서)**: 커밋 `645f910`. `--cal-weight 0`이 `cal_idx`까지
  `None`으로 넘겨서 anchor 풀(`train_cols`)이 40→60으로 같이 바뀌던
  confound 수정(claim5-a, `cal_idx`는 이제 항상 전달) — 이번 세션 결과는
  전부 `cal_weight` 기본값(1.0)을 썼으므로 영향 없음. `identity_aware=True`가
  FP top-(k+1) 밖의 class가 Q에서 치솟는 경우(intrusion)를 아예 못 보던
  버그 수정(claim5-b) — **이 수정 전에 돌린 identity-aware 6-seed 결과는
  재현하려면 재실행 필요**. `--conditions` 플래그와 `print(vars(args))`
  추가(claim5-c).
- **2026-09-16 (이어서 2)**: 커밋 `4629adf`(claims 문서), 이어서
  `--smult-per-tensor`/`--identity-aware-margin`을 `PROMPTCAL_CURRENT_MODEL.md`
  §8.1 확정 설계로 승격 — 풀스케일 6-seed(`runs/67_final_confirmed_fullscale/`)
  검증 완료 후 `argparse.BooleanOptionalAction`으로 바꿔서 두 플래그의
  **기본값을 True로 전환**(`--no-smult-per-tensor`/`--no-identity-aware-margin`으로
  이전 설계로 되돌릴 수 있음). `build()` 함수 자체의 기본값도 맞춰서 변경.
  `cal_weight`/`scale_reg_weight`가 이미 확정값을 기본으로 쓰는 것과
  일관성을 맞춘 것 — 이제 위 "실행 방법"의 예시 커맨드가 플래그 추가 없이
  §8.1을 그대로 재현한다.
- **2026-09-17**: `PROMPTCAL_CURRENT_MODEL.md`(v1)가 per-channel 시절
  서사와 하이퍼파라미터 스윕 10개 절이 누적돼 지금 설계를 파악하기 어려울
  만큼 두꺼워져서, 지금 확정 설계만 처음부터 깔끔하게 다시 쓴
  `PROMPTCAL_CURRENT_MODEL_V2.md`를 새로 작성 — v1은 역사적 감사 기록으로
  보존.
- **2026-09-17 (이어서)**: `--control-mse`/`--learn-act-scale` 추가(claim12,
  전부 opt-in, 기본 False). Combined의 margin_loss를 순수 MSE로 대체하는
  control 실험과, baseline(AdaRound/QDrop/BRECQ)에 activation scale 학습
  손잡이(s_mult)를 추가하는 실험 -- 둘 다 §8 확정 설계에는 영향 없음(진단용).
- **2026-09-18**: `adaround.py`/`brecq.py`의 재구성 loss 정규화·lr·reg_weight·
  warmup을 공식 BRECQ(Li et al. ICLR'21, `yhhhli/BRECQ` GitHub) 값으로
  전면 수정(claim13) -- 기존 `.pow(2).mean()`은 공식 `lp_loss` 대비 C_out배
  작아서 rounding 정규화가 압도, AdaRound의 alpha가 사실상 round-to-nearest에서
  못 움직이고 있었다(nearest 대비 flip 0.069%→5~7%대로 수정, `flip_rate()`
  진단으로 확인). **opt-in이 아니라 라이브러리 기본값 자체가 바뀐 것이라, 이
  수정은 §8 확정 표 전체(baseline 4개 + Combined의 weight-rounding 단계까지)에
  영향을 준다** -- 6-seed 재측정 완료(`runs/71_recon_fix_review/`). **결과:
  alpha가 실제로 훨씬 많이 움직이는데도 AdaRound/QDrop/BRECQ는 여전히
  naive보다 COCO_AP가 낮다** -- "정규화 버그가 원인"이라는 가설은 반박됐지만,
  이 수정 자체는 baseline을 원 논문대로 정확히 구현하기 위한 것이었지 성능을
  올리려던 게 아니므로(claim4/5와 같은 원칙) 결과 방향과 무관하게 유지한다.
  "정확히 구현해도 naive를 못 이긴다"는 이제 실제 결과로 §8/§9에 반영됨.
  QDrop을 layer-wise(`optimize_adaround`)에서 block-wise
  (`optimize_brecq(qdrop_prob=...)`)로 이전 -- 원 논문(Wei et al. ICLR'22)의
  재구성 단위와 일치시킴.
- **2026-09-18 (이어서, claim14)**: claim13으로 Combined 자신의
  1단계(`optimize_adaround`)도 alpha를 크게 움직이게 됐는데, 그 목적함수
  (순수 MSE reconstruction)가 2단계(margin_loss/s_mult)가 보호하는 영역
  (S∪H_cal+neighbor)과 무관해서 그 바깥(COCO 전체 Top1_flip/lost, LVIS)으로
  손상이 새는 부작용이 확인됐다. 처음엔 "Combined의 1단계를 BRECQ의
  block-wise 재구성으로 바꾸자"는 안이 나왔으나, 사용자가 "그러면
  reconstruction으로 decision error를 줄이자는 거 아니냐"고 지적 -- 논문
  핵심 주장과 충돌하는 프레이밍이라 철회. 대신 `--combined-recon-iters`
  플래그를 추가해 **1단계를 아예 생략(기본값 0, round-to-nearest weight로
  대체)**하는 쪽으로 확정 -- 6-seed 검증 결과 COCO_AP/LVIS_AP/APr/
  Heval_flip/LVIS_lost 5개 지표가 트레이드오프 없이 동시에 개선됐다
  (`runs/72_combined_recon_diag/`). Combined는 이제 AdaRound 메커니즘을
  전혀 안 쓴다. `build()`의 `combined_recon_iters` 기본값도 0.
- **2026-09-19 (claim15, 최신)**: §8 확정 표가 불공정 비교였음이 드러남 --
  QDrop/BRECQ의 activation scale 학습(LSQ)이 꺼진 채로 Combined와 비교
  중이었다(사용자 지적). `--learn-act-scale`(단일 플래그, 기본 False)를
  `--adaround-learn-act-scale`(기본 False, 원 논문에 LSQ 없음)·
  `--qdrop-brecq-learn-act-scale`(기본 **True**, 원 논문에 있음)로 분리 --
  claim12를 opt-in으로 방치했던 실수를 바로잡음. 이 공정 비교 6-seed
  (`runs/75_lsq_confirmed/`)에서 Combined는 9개 지표 중 COCO_AP·LVIS_AP
  2개만 BRECQ+LSQ를 이겼다. `scale_reg_weight`를 10.0→1.0으로 재튜닝(1-seed
  스윕 후 6-seed 확정, `runs/78_scalereg1_confirmed/`)해서 APr·LVIS_lost가
  추가로 뒤집혀 승리 지표 4개로 증가 -- 나머지 5개(Heval_flip/Top1_flip/
  UPIR/lost/LVIS_flip)는 아직 짐. `--w-bits`/`--a-bits`(하드코딩 8/8 해소,
  메커니즘 실험용), `--combined-stage1`(BRECQ block-wise를 1단계로 쓰는
  진단 실험, 제안 방법 아님) 추가. `quant_weight()`의 죽은 alpha 재계산도
  캐싱으로 제거. `--scale-reg-weight` 기본값 10.0→1.0.
- 최신 확정 하이퍼파라미터·공식 데이터 6-seed 결과의 단일 진실 공급원은
  저장소 루트의 `PROMPTCAL_CURRENT_MODEL_V2.md`다. 이 README와 수치가
  어긋나면 그쪽을 따를 것.
