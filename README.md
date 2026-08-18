# PromptCal-PTQ

Open-vocabulary detector(YOLO-World / OWLv2)의 W8A8 post-training quantization이
region–prompt **의미적 의사결정**을 어떻게 망가뜨리는지 측정하고, 이를 보존하는
PTQ 방법을 구현하기 위한 실험 코드베이스.

---

## 핵심 아이디어 (왜 이 구조인가)

이 논문의 모든 실험은 **하나의 측정 축**으로 환원된다:

> 같은 이미지 · 같은 prompt vocabulary에 대해,
> **FP32 detector와 양자화된 detector의 region–prompt 유사도 행렬을 각각 뽑아서 비교한다.**

YOLO-World에서 그 유사도 행렬은 detection head의 **classification logit** 그 자체다
(region feature ↔ text embedding contrastive 유사도 = cls logit).
따라서 head에 forward hook 하나를 걸면 `[num_regions × num_prompts]` 행렬이 그대로 잡힌다.

Top-1 flip, GT rank, boundary inversion, AP — 모든 지표가 이 행렬만 있으면 계산된다.
→ 그래서 코드의 코어는 `src/harness.py`의 `SimilarityHarness` 하나다.

---

## 모듈 맵

```
promptcal-ptq/
├── configs/
│   └── yoloworld_coco.yaml      # 모델/데이터/프롬프트/양자화 설정 한 곳에
├── scripts/
│   ├── 01_reproduce_baseline.py # [지금 여기] FP32 baseline AP 재현 (sanity check)
│   ├── 02_dump_similarity.py    # (다음) 유사도 행렬 덤프 하네스 검증
│   ├── 03_motivation_damage.py  # (Sec 3.2~3.4) FP vs INT8 semantic damage
│   └── 04_run_method.py         # (Sec 4) 제안 방법 학습/평가
├── src/
│   ├── harness.py               # ★ SimilarityHarness — 코어 추상화
│   ├── quant/                   # W8A8 fake-quant (naive N=1 → 나중에 AdaRound/QDrop)
│   ├── metrics/
│   │   └── semantic_metrics.py  # top-1 flip / GT rank / boundary inversion
│   └── eval/                    # pycocotools 래퍼 (AP, size-stratified 선택)
└── requirements.txt
```

## 실행 순서 (로드맵)

| 단계 | 스크립트 | 목적 | 방법 필요? |
|---|---|---|---|
| 1 | `01_reproduce_baseline.py` | FP32 baseline AP 재현 | X |
| 2 | `02_dump_similarity.py` | 유사도 행렬 hook 검증 | X |
| 3 | `03_motivation_damage.py` | FP vs INT8 의미 손상 (Sec 3) | X (naive 양자화만) |
| 4 | `04_run_method.py` | Semantic Calib + Utility Refine (Sec 4) | O |

**지금 할 것: 1단계.** 여기서 37.7 근처가 나오면 나머지가 다 이 위에서 돌아간다.

---

## 셋업 (conda + use-cuda)

핵심 원칙: **환경 관리는 conda, 패키지 설치는 전부 pip.** torch를 conda 채널로 깔면
pip으로 깐 ultralytics/pycocotools와 섞여 의존성이 깨지므로, torch도 pip으로 깐다.
그리고 **시스템 CUDA 툴킷(use-cuda)과 torch 휠의 CUDA 태그를 한 쌍으로 맞춘다.**

```bash
# 0) 드라이버가 지원하는 최대 CUDA 확인 (이 값이 상한선)
nvidia-smi                     # 우측 상단 "CUDA Version: X.Y"

# 1) 설치된 CUDA 툴킷 목록/현재 상태 확인
use-cuda --list
use-cuda --show

# 2) 이번 셸에서 쓸 CUDA 툴킷 선택 (드라이버 상한 이하로. 예: 12.6)
#    torch 휠도 이와 같은 버전(cu126)으로 맞출 것.
eval "$(use-cuda --session 12.6)"   # export 줄을 현재 셸에 적용
nvcc --version                       # 12.6으로 바뀌었는지 확인

# 3) conda 환경 생성 (Python 3.11)
conda create -n promptcal python=3.11 -y
conda activate promptcal

# 4) torch를 CUDA 맞춰 "먼저" pip 설치 (툴킷 12.6 ↔ cu126)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 5) torch가 GPU 잡는지 즉시 검증 (True 나와야 함)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 6) 나머지 설치 (torch는 이미 만족되므로 건드리지 않음)
pip install ultralytics pycocotools numpy

# 7) ultralytics 설치 후 torch가 CPU 빌드로 안 바뀌었는지 재확인
python -c "import torch; print('cuda ok:', torch.cuda.is_available())"
```

CUDA 태그 매칭: `use-cuda`로 12.6을 골랐으면 torch는 `cu126`, 12.8이면 `cu128`
(단 cu128 휠이 pytorch 인덱스에 있는지 먼저 확인), 11.8이면 `cu118`.
잘 모르겠으면 12.6 ↔ cu126 조합이 가장 무난하다.

> `use-cuda --session`은 이 셸에만 적용된다. 매번 새 셸/새 conda activate마다
> 다시 실행해야 하므로, 자주 쓰면 `use-cuda --user 12.6`으로 사용자 기본값을
> 저장해두는 것도 방법.

COCO val2017은 첫 `model.val()` 호출 시 Ultralytics가 자동 다운로드한다
(val 이미지 5000장 + annotation, 약 1GB). 이미 로컬에 있으면
`configs/yoloworld_coco.yaml`의 `coco_data_yaml` 경로를 기존 coco.yaml로 지정.

## 1단계 실행

```bash
python scripts/01_reproduce_baseline.py --model yolov8s-worldv2.pt --device 0
```

기대 출력: `mAP50-95 ≈ 37.7`. ±0.3 안이면 성공.
