# Vocabulary-metric 재구성 — 저비트(W4A8/W4A4) head 포함 설계 (2026-09-28)

W8A8 Combined(`s_mult` + top-k margin)는 저비트로 전이되지 않았다(설계 문서 §7, claim21·22).
이 문서는 그 대안으로 구현한 **재구성 손실의 metric 교체**를 정리한다. 아직 GPU 실측 전이다 —
아래 수치 칸은 전부 비어 있고, 실험 순서와 판정 기준만 정해 둔다.

## 1. 왜 이 방향인가

| 관찰 | 출처 |
|---|---|
| W4A8에서 손상의 주범은 weight rounding. `s_mult`(activation scale)만으로는 BRECQ에 −0.56 | claim22 |
| rounding을 decision loss로 건드리면 자유도가 수백만이든 52개든 똑같이 붕괴(flip ~6%) | claim21·22 |
| BRECQ(dense 재구성, FP에 고정)는 W4A8을 33.5까지 복구 | runs/113 |
| 그런데 W4A8 BRECQ도 LVIS_flip 20.5%, Heval_flip 27.8%, LVIS_AP 0.230(FP 0.259) | runs/113·120 |

즉 **sparse한 top-k 감독으로 weight를 움직이는 것**이 문제였고, dense 재구성 자체는 저비트에서도
안정적이다. 그렇다면 BRECQ의 틀(옵티마이저·자유도·예산)은 그대로 두고 **무엇을 재구성하느냐(metric)만**
open-vocabulary 결정에 맞게 바꾸는 게 남은 경로다.

## 2. 유도

ContrastiveHead: `sim_j = tau_l * <x̂, ŵ_j> + b_l` (x = cv3 출력 = cv4 입력, 레벨 l).
프롬프트 j의 오차는 `Δ_j ≈ tau_l <Δx̂, ŵ_j>`이고, 프롬프트 분포 V에 대해

```
E_{w~V}[Δ_j²] = tau_l² · Δx̂ᵀ C_V Δx̂,     C_V = E_{w~V}[w wᵀ] = Σ_V + μμᵀ
```

- **dense**: 모든 anchor·모든 방향에 대한 2차형식이라 BRECQ처럼 FP에 고정된다.
- **특정 vocabulary에 안 맞춘다**: C_V는 평가 vocabulary와 이름이 겹치지 않는 WordNet 구체 명사 4000개
  (`configs/vocab_generic.txt`)로 만든다. CLIP 텍스트 임베딩은 좁은 원뿔에 몰려 있어 C_V의 유효 차원이
  512보다 훨씬 작을 것으로 예상된다 — 그 밖의 오차는 순위에 거의 영향이 없다.
- `--vm-lam-mean`: `C = Σ + λμμᵀ`. λ=1은 비중심 2차 모멘트(순위 + 절대 점수), λ=0은 순위만.

## 3. 구현 (`pipeline/quant/vocab_metric.py`, `brecq.py::optimize_brecq(vocab_metric=...)`)

| target | 손실 | 비고 |
|---|---|---|
| head `cv3[l][-1]` (임베딩 1x1 conv) | **exact**: anchor 가중 평균 `tau² vᵀCv`, `v = normalize(q) − normalize(f)` | 출력이 유사도에만 쓰이므로 재구성 항 없음. 첫 iteration의 재구성/vocab 비율로 척도만 맞춤(round 정규화와의 균형 유지) |
| 그 외 backbone/neck 블록, cv3 앞단 conv | **Fisher**: `(1−mix)·재구성 + mix·Σ F⊙Δ²` | `F = E[(∂s/∂z)²]`, `s = Σ_a √w_a tau_l <x̂_a, u_a>`, `u_a ~ N(0, C)` → F는 끌어올린 metric `Σ_a w_a diag(J_aᵀ C J_a)`. per-sample, 평균 1로 정규화 |
| head `cv2` (box) | 순수 재구성 | 유사도 경로 밖 → gradient None으로 자동 판별 |

- anchor 가중치 `w_a = max_j σ(sim_fp[a, j]) + floor` (bank 전체에 대해, 기본 floor 0.05).
- Fisher 샘플링은 **전용 RNG**(`torch.Generator`)를 쓴다 — 전역 RNG를 소비하지 않아서 다른 조건의
  RNG 스트림 위치가 밀리지 않는다(toy 모델로 확인).
- `brecq_vm`/`qdrop_vm`은 대응 baseline과 **인자를 전부 같게** 넘기고 `vocab_metric`만 추가한다.
  짝비교가 metric 단일 변수다(Combined Stage 1의 neck_layerwise 교란을 반복하지 않음).
- `vocab_metric=None`이면 `optimize_brecq`의 기존 코드 경로 그대로다 → 기존 조건은 bit-identical.
- two_stage의 activation 단계에는 적용하지 않는다(기본값은 공동 최적화라 해당 없음).
- 비용: Fisher target마다 calib 이미지당 FP forward(grad) 1회 + backward K회(기본 4)가 추가된다.
  BRECQ 빌드 대비 대략 +1~1.5배로 예상(미측정). `--vm-samples 2`로 줄일 수 있다.

**toy 검증 (CPU)**: Fisher 추정량이 선형 사상에서 해석해 `w·tau²·diag(Jᵀ C J)`와 상대오차 1.8%(20k 샘플),
exact/fisher/재구성 분류가 구조대로(cv3 마지막 → exact, cv2 → 재구성), vm=None 경로 결정적,
qdrop_vm 동작.

**실제 아키텍처 스모크 (CPU, `yolov8s-world.yaml` 무작위 초기화, 128px, iters 3)**: head 포함 W4A8에서
래핑 70 conv, `--first-last-bits 8`이 stem + cv2/cv3 마지막 conv 6개를 잡음, vm 분류 exact 3 / fisher 23
(17블록 + cv3 앞단 6) / 재구성 9(cv2). `vocab_metric=None` 경로는 HEAD 코드와 alpha·LSQ delta가
**bit-identical**(qdrop 경로로 확인). **학습된 가중치·실데이터·GPU로는 아직 안 돌렸다** — 수치는 전부 미측정.

## 4. 첫/마지막 레이어 비트 (`--first-last-bits 8`)

지금까지 `wrap_convs`에 예외가 없어 W4A4에서 stem이 이미지 픽셀을 4bit로 받았다. BRECQ·QDrop·Reg-PTQ의
W4A4 표는 모두 첫/마지막 레이어를 8bit로 둔다. 이 플래그는 stem 첫 conv와(head 양자화 시)
`cv2[l][-1]`, `cv3[l][-1]`을 8bit로 고정한다. **모든 조건에 똑같이 적용**되고, calibrate 전에 적용돼
observer·weight scale도 그 비트로 잡힌다. 기본 0 = 꺼짐.

## 5. 실험 순서와 판정

공통: `--no-skip-head --first-last-bits 8 --deterministic --brecq-batch 1`(기본), 같은 GPU 종류.

| # | 실험 | 판정 |
|---|---|---|
| E0 | W4A4·W8A4 baseline: `--conditions naive,qdrop,brecq`, 1-seed `--eval-cap 500` | QDrop W4A4가 수십 AP로 살아나면 A4 축이 열린다. 여전히 0 근처면 activation 스킴 문제(per-channel 재파라미터화 필요) |
| E1 | `pipeline/legacy/diag_vocab_subspace.py` W4A8 head 포함, naive·brecq | C 유효 차원 ≪ 512이고 BRECQ의 rho ≲ 1이면 metric 교체로 옮길 용량이 있다. rho가 이미 크면 이득이 작을 것 |
| E2 | W4A8 `--conditions brecq,brecq_vm`, 2-seed full probe | LVIS_AP·LVIS_flip이 2/2 개선이면 6-seed로 |
| E3 | `--vm-vocab` ablation: `identity` / `coco` / 기본(generic) / `lvis`(oracle 상한) | "generic ≈ lvis > coco > identity"면 논문 핵심 표. identity만으로 같은 이득이면 기여는 "방향 보존 재구성"으로 축소 |
| E4 | `--vm-mix` {0.25, 0.5, 1.0}, `--vm-lam-mean` {0, 1} | box 경로 손실(COCO_AP) vs LVIS 트레이드오프 확인 |
| E5 | W4A4에서 `qdrop,qdrop_vm` (E0이 열렸을 때) | |

```bash
# E1
.venv/bin/python pipeline/legacy/diag_vocab_subspace.py --model yolov8s-world.pt --coco-root <coco> \
  --w-bits 4 --a-bits 8 --no-skip-head --first-last-bits 8 --modes naive,brecq \
  --calib 256 --n-eval 200 --device <gpu> --deterministic

# E2 (seed 0,1)
.venv/bin/python pipeline/run_comparison.py --model yolov8s-world.pt --device <gpu> --seed <s> \
  --deterministic --w-bits 4 --a-bits 8 --calib 256 --no-skip-head --first-last-bits 8 \
  --conditions brecq,brecq_vm --vm-vocab configs/vocab_generic.txt
```

## 6. 리스크 / 아직 모르는 것

- **E1이 부정적이면**(BRECQ 오차가 이미 텍스트 방향에 몰려 있으면) 재가중으로 얻을 이득은 작다.
- head 제외 프로토콜에서는 exact target이 없고 Fisher 경로만 작동한다(cv3가 FP).
- neck의 C2fAttn·ImagePoolingAttn은 calibration 중 COCO-80 텍스트로 guide된다. 모든 조건이 같으므로
  비교는 공정하지만, "vocabulary-agnostic" 주장에는 이 점을 각주로 밝혀야 한다.
- `configs/vocab_generic.txt`는 ultralytics yaml의 COCO/LVIS **이름**만으로 제외했다. 서버에서
  `pipeline/legacy/build_generic_vocab.py --lvis-ann <json>`으로 LVIS synonyms·synset까지 뺀 엄격판을 만들 수 있다
  (별도 파일로 두고 결과가 같은지 확인할 것).
