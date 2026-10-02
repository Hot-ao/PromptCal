"""
"논문 방식"에 가까운 공식 데이터 설정으로 5개 조건(naive/AdaRound/QDrop/BRECQ/
Combined) 전체 검증. `scripts/58_full_baseline_official_data.py`를 이 디렉토리
전용으로 포팅한 것(로직 동일, import 경로만 harness.py/quant/*로 변경, 09-15).
Combined는 현재 확정된 최종 설계 그대로다 — per-channel `s_mult` +
`scale_reg_weight=10`(정규화) + `cal_weight=1.0`(H_cal 직접 보호) + H_eval을
neighbor 후보 풀에서 제외(exclude_from_neighbors). 확정 경위와 전체
하이퍼파라미터·6-seed 결과는 저장소 루트의 `PROMPTCAL_CURRENT_MODEL.md`
(특히 §5.5/§8.1)가 단일 진실 공급원이다.

데이터 설정(이전 val2017-슬라이스 방식에서 09-08에 교체):
  - calibration: COCO **train2017**에서 256장(평가 데이터와 완전 분리).
  - COCO-80 평가: COCO val2017 **전체**(5000장).
  - LVIS 평가: 공식 **lvis_v1_minival.json**(ultralytics 공식 배포,
    lvis-labels-segments.zip에 포함) — 확인 결과 "COCO val2017 ∩ LVIS val"과
    정확히 일치하는 4809장.

측정 지표:
  - COCO_AP(전체+S/H_eval subset), LVIS_AP(+APr/APc/APf, 실제 LVIS GT)
  - Heval_flip(masked, 우리 진단용) / Top1_flip(표준, AP와 동일 배포 조건)
  - GT_MRR/R@1, lost/gained/lateral/corrective_rate, UPIR (COCO·LVIS 양쪽)
  - lost의 S/H_cal/H_eval 그룹별 분해(neighbor-hinge·cal_weight가 의도대로
    작동하는지 진단)
  - calibration 시간, 이론적 모델 크기

GT 앵커 매칭: GT 박스를 letterbox 변환으로 640 공간에 옮긴 뒤, 3개 FPN
level(80x80/40x40/20x20, stride 8/16/32)에서 그리드 셀 중심이 박스 안에 드는
후보를 모으고, 그중 FP32가 정답 class에 가장 confident한 anchor 하나를 GT의
대표 anchor로 선택한다(공식 TaskAlignedAssigner와 완전히 동일하지 않은 근사).

LVIS(P=1203) 쪽은 probe 전체(4809장)의 sim 행렬을 리스트로 들고 있으면 CPU
RAM이 터지므로(fp_sims 하나만 이론상 ~194GB) `compute_lvis_flip_gt_streaming`이
이미지 한 장씩 처리해서 즉시 소비·폐기하는 스트리밍 방식으로 flip/GT 지표를
누적한다. AP 자체는 lvis-api(`LVISEval`)로 별도 채점.

실행(GPU 3개에 각각 다른 --seed로, 기존 6-seed는 0~5):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python pipeline/run_comparison.py \
        --model yolov8s-world.pt --coco-root /data/taeho/coco_datasets \
        --data configs/coco_local.yaml \
        --lvis-ann /data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json \
        --calib 256 --seed 0 --device 0

스모크 테스트(--eval-cap으로 probe 수 제한, calib도 줄여서 몇 분 안에 확인):
    ... --calib 8 --eval-cap 16 --iters 30 --recon-iters-ada 20 --seed 0 --device 0
"""
import argparse, glob, os, sys, time

# 09-28: CUDA 장치 인덱스를 nvidia-smi 인덱스와 일치시킨다. torch 기본값
# CUDA_DEVICE_ORDER=FASTEST_FIRST는 이 서버(L40S 3장 + RTX 4000 Ada 5장)에서
# L40S를 먼저 정렬해버려서 --device 0이 실제로는 nvidia-smi GPU 1을,
# --device 1이 GPU 2를 잡는다(실측: 09-28 14:30, runs/116 seed0이 의도한 GPU 0
# 대신 GPU 1에 올라갔다). 공용 서버에서 남의 GPU를 덮칠 수 있는 사고라
# torch를 import하기 전에 PCI_BUS_ID로 고정한다. setdefault라 바깥에서
# 명시적으로 지정한 값은 그대로 존중한다.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import SimilarityHarness
from quant.quant_model import wrap_convs, calibrate, set_first_last_bits, set_last_abits, set_block_wbits, set_conv_wbits, set_conv_abits
from quant.adaround import convert_to_adaround, optimize_adaround, AdaRoundQuantConv2d, free_cpu_mem
from quant.fake_quant import QuantConv2d
from quant.brecq import optimize_brecq
from quant.vocab_metric import VocabMetric, load_vocab, encode_text_bank
from quant.promptcal import optimize_promptcal_scale_neighbor


def load_names(which):
    import ultralytics, yaml
    from pathlib import Path
    p = Path(ultralytics.__file__).parent / "cfg" / "datasets" / f"{which}.yaml"
    d = yaml.safe_load(open(p))
    n = d["names"]
    return [n[i] for i in range(len(n))]


def letterbox(im, new=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_r = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    return cv2.copyMakeBorder(im_r, top, new - nh - top, left, new - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess(path, imgsz, device):
    """CPU 텐서로 반환 -- probe가 5000장 규모라 전부 GPU에 올리면 OOM난다.
    run_image/calibrate/optimize_* 전부 내부에서 자체적으로 .to(device)를 하므로
    여기서 미리 옮길 필요가 없다(device 인자는 호환성을 위해 남겨두되 안 씀)."""
    im = letterbox(cv2.imread(path), imgsz)
    im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
    t = torch.from_numpy(im).unsqueeze(0).float()
    t.div_(255.0)          # in-place: out-of-place `/255.0`는 이미지당 여분의
    return t                # float32 버퍼를 만들고 버려서, probe 5000장 기준
                            # RSS가 이론치(~24GB)의 2배(~47GB)로 부풀었었다(09-08 발견).


def switch_vocab(model, names, device):
    # cache_clip_model=False를 쓰려고 내부 model.model.set_classes를 직접
    # 호출하는데(YOLOWorld.set_classes wrapper는 이 인자를 안 받음), 그러면
    # wrapper가 하는 model.model.names 갱신/predictor 리셋이 같이 스킵된다.
    # 지금은 verbose=False+예측 후처리가 nc=0으로 안전하게 동작해서 수치
    # 결과에는 영향 없지만(09-16 확인), 시각화/verbose를 켜면 이름-인덱스가
    # 어긋나 IndexError가 날 수 있어 직접 맞춰준다(09-16 수정).
    model.model.to("cpu")
    model.model.set_classes(names, cache_clip_model=False)
    model.model.names = list(names)
    model.predictor = None
    model.model.to(device).eval()


def measure_ap(model, data, imgsz, device):
    # workers=0: 기본값(8)이면 DataLoader가 os.fork()로 worker를 여러 개 띄우는데,
    # 이 시점 부모 프로세스가 이미 커져 있으면(q_sims 등) 그 메모리를 통째로
    # 복사해서 순식간에 수백 GB로 터질 수 있다(09-09 실측: RSS 66GB 부모에서
    # worker 8개가 fork되며 시스템 전체가 OOM 직전까지 감). single-process로
    # 강제해서 fork 자체를 없앤다.
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=False,
                        verbose=False, workers=0)
    overall = float(metrics.box.map) * 100, float(metrics.box.map50) * 100
    # 09-16 버그 수정: metrics.box.maps는 이미 클래스 id로 직접 인덱싱된
    # nc-길이 배열이라(ultralytics.utils.metrics.Metric.maps 참고) ap_class_index와
    # zip으로 "위치" 짝짓기하면 안 된다 -- ap_class_index가 [0,1,...,nc-1] 풀레인지일
    # 때만 우연히 맞는다. 클래스별로 직접 인덱싱해야 맞다. (또한 구버전 호환용
    # fallback이던 all_ap[:, 0]은 AP50이라 map(AP@0.5:0.95)과 지표 자체가 달라서
    # 같이 제거 -- 현재 ultralytics(8.4.121)는 .maps를 항상 갖고 있어 불필요.)
    per_class = {int(c): float(metrics.box.maps[c]) for c in metrics.box.ap_class_index}
    return overall, per_class


def subset_map(per_class, idx, scale=100.0):
    vals = [per_class[i] for i in idx if i in per_class]
    return float(np.mean(vals)) * scale if vals else float("nan")


def group_flip(fp_sims, q_sims, group_idx, conf=0.25):
    Gm = torch.zeros(80, dtype=torch.bool)
    Gm[group_idx] = True
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        target = conf_m & Gm[c_fp]
        if target.sum() == 0:
            continue
        idx = target.nonzero(as_tuple=True)[0]
        fp_K = sf.clone(); fp_K[:, Gm] = -1e9
        q_K = sq.clone();  q_K[:, Gm] = -1e9
        fl += int((fp_K.argmax(-1)[idx] != q_K.argmax(-1)[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def standard_flip(fp_sims, q_sims, conf=0.25):
    tot = fl = 0
    for sf, sq in zip(fp_sims, q_sims):
        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        if conf_m.sum() == 0:
            continue
        idx = conf_m.nonzero(as_tuple=True)[0]
        c_q = sq.argmax(-1)
        fl += int((c_fp[idx] != c_q[idx]).sum())
        tot += len(idx)
    return fl / max(tot, 1) * 100, tot


def load_coco_gt_by_path(ann_path, img_paths):
    from pycocotools.coco import COCO
    from ultralytics.data.converter import coco91_to_coco80_class
    coco = COCO(ann_path)
    map91to80 = coco91_to_coco80_class()
    fname_to_id = {info["file_name"]: img_id for img_id, info in coco.imgs.items()}
    out = {}
    for p in img_paths:
        fname = os.path.basename(p)
        img_id = fname_to_id.get(fname)
        boxes = []
        if img_id is not None:
            for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False)):
                cls80 = map91to80[a["category_id"] - 1]
                if cls80 is None:
                    continue
                x, y, w, h = a["bbox"]
                boxes.append((cls80, x + w / 2, y + h / 2, w, h))
        out[p] = boxes
    return out


def load_lvis_gt_by_path(lvis_gt, img_paths, img_ids):
    ann_by_img = {}
    for a in lvis_gt.dataset["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    out = {}
    for p, iid in zip(img_paths, img_ids):
        boxes = []
        for a in ann_by_img.get(iid, []):
            cls = a["category_id"] - 1
            x, y, w, h = a["bbox"]
            boxes.append((cls, x + w / 2, y + h / 2, w, h))
        out[p] = boxes
    return out


def get_grid_specs(h_fp, sample_img):
    h_fp.run_image(sample_img, -1)
    specs = []
    for i in sorted(h_fp._level_buf):
        B, P, H, W = h_fp._level_buf[i].shape
        specs.append((H, W))
    return specs


def build_gt_targets(fp_sims, img_paths, gt_by_path, grid_specs, imgsz):
    offsets = []
    off = 0
    for (H, W) in grid_specs:
        stride = imgsz // W
        offsets.append((off, H, W, stride))
        off += H * W
    targets = []
    for i, p in enumerate(img_paths):
        sf = fp_sims[i]
        im0 = cv2.imread(p)
        h0, w0 = im0.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
        img_targets = []
        for (cls, cx, cy, bw, bh) in gt_by_path.get(p, []):
            x0 = (cx - bw / 2) * r + left; x1 = (cx + bw / 2) * r + left
            y0 = (cy - bh / 2) * r + top;  y1 = (cy + bh / 2) * r + top
            candidates = []
            for (off0, H, W, s) in offsets:
                col0 = max(0, int(x0 / s - 0.5)); col1 = min(W - 1, int(x1 / s - 0.5) + 1)
                row0 = max(0, int(y0 / s - 0.5)); row1 = min(H - 1, int(y1 / s - 0.5) + 1)
                for row in range(row0, row1 + 1):
                    for col in range(col0, col1 + 1):
                        ccx, ccy = (col + 0.5) * s, (row + 0.5) * s
                        if x0 <= ccx <= x1 and y0 <= ccy <= y1:
                            candidates.append(off0 + row * W + col)
            if not candidates:
                continue
            best = max(candidates, key=lambda a: float(sf[a, cls]))
            img_targets.append((best, cls))
        targets.append(img_targets)
    return targets


def gt_metrics_for_method(fp_sims, q_sims, gt_targets, H_eval_set=None, S_set=None, H_cal_set=None):
    recip_ranks = []
    lost = gained = lateral = 0
    upir_num = upir_den = 0
    # S/H_cal/H_eval 그룹별 lost 분해 -- "COCO-80 lost가 H_eval(비보호
    # 클래스)에 몰리는가"라는 가설을 직접 확인하기 위함. denom은 "FP32는
    # 맞혔던(fp_rank==1) GT-anchor 수"(그룹별) -- lost가 나올 수 있는 최대
    # 후보군이라 이걸로 나눠야 그룹 크기(S=40,H_cal=20,H_eval=20) 차이를
    # 보정한 공정한 비율이 나온다.
    lost_by_group = {"S": 0, "H_cal": 0, "H_eval": 0}
    denom_by_group = {"S": 0, "H_cal": 0, "H_eval": 0}
    for i in range(len(fp_sims)):
        sf = fp_sims[i]; sq = q_sims[i]
        for (aidx, cls) in gt_targets[i]:
            fp_s = sf[aidx]; q_s = sq[aidx]
            fp_rank = int((fp_s > fp_s[cls]).sum()) + 1
            q_rank = int((q_s > q_s[cls]).sum()) + 1
            recip_ranks.append(1.0 / q_rank)
            group = None
            if S_set is not None and cls in S_set:
                group = "S"
            elif H_cal_set is not None and cls in H_cal_set:
                group = "H_cal"
            elif H_eval_set is not None and cls in H_eval_set:
                group = "H_eval"
            if fp_rank == 1 and q_rank != 1:
                lost += 1
                if group is not None:
                    lost_by_group[group] += 1
            if group is not None and fp_rank == 1:
                denom_by_group[group] += 1
            if fp_rank != 1 and q_rank == 1:
                gained += 1
            # lateral: 이 GT-anchor에서 FP/quant 둘 다 오답인데, quant의 예측
            # 자체가 FP와 다르게 바뀐 경우(오답->다른 오답 flip). lost/gained만
            # 보면 "GT-anchor에서 일어난 flip 중 실제로 정답이 되는 비율"을
            # 못 재는데(분모에 이 lateral이 빠짐), 이걸 포함해야
            # corrective_rate = gained/(lost+gained+lateral)이 정확해진다.
            if fp_rank != 1 and q_rank != 1 and int(fp_s.argmax()) != int(q_s.argmax()):
                lateral += 1
            if H_eval_set is not None and fp_rank == 1 and cls not in H_eval_set:
                upir_den += 1
                if int(q_s.argmax()) in H_eval_set:
                    upir_num += 1
    n = len(recip_ranks)
    mrr = sum(recip_ranks) / max(n, 1)
    r1 = sum(1 for r in recip_ranks if r == 1.0) / max(n, 1)
    total_flips = lost + gained + lateral
    corrective_rate = gained / max(total_flips, 1) * 100
    out = dict(mrr=mrr, r1=r1, lost=lost, gained=gained, lateral=lateral,
              total_flips=total_flips, corrective_rate=corrective_rate, n=n)
    if H_eval_set is not None:
        out["upir"] = upir_num / max(upir_den, 1) * 100
        out["upir_n"] = upir_den
    if S_set is not None and H_cal_set is not None:
        out["lost_by_group"] = lost_by_group
        out["denom_by_group"] = denom_by_group
        out["lost_rate_by_group"] = {
            g: lost_by_group[g] / max(denom_by_group[g], 1) * 100 for g in lost_by_group
        }
    return out


def predict_lvis_results(model, img_paths, img_ids, imgsz, device, conf=0.001, max_det=1000):
    # 09-17 버그 수정: DetectionValidator(COCO_AP가 쓰는 model.val() 경로)는
    # NMS를 multi_label=True로 호출하는데, DetectionPredictor(여기서 쓰는
    # model.predict() 경로)는 이 인자를 아예 안 넘겨서 non_max_suppression
    # 기본값 multi_label=False가 조용히 적용되고 있었다. LVIS처럼
    # 1203-class 동의어/federated annotation이 밀집한 vocabulary에서는
    # anchor당 top-1 class만 후보로 내는 multi_label=False가 recall을
    # 심하게 깎는다(실측: FP32 AP 0.126→0.233, APr 0.031→0.158, 4809장
    # 공식 minival 기준). model.predict()는 self.args에서 multi_label을
    # 아예 안 읽어서(DetectionPredictor.postprocess 확인) predict()에
    # multi_label=True를 인자로 넘겨도 무시된다 -- nms 모듈 함수 자체를
    # 임시로 patch해야 실제로 적용된다.
    from ultralytics.utils import nms
    import functools
    orig_nms = nms.non_max_suppression
    nms.non_max_suppression = functools.partial(orig_nms, multi_label=True)
    try:
        results = []
        for path, img_id in zip(img_paths, img_ids):
            r = model.predict(source=path, imgsz=imgsz, device=device, conf=conf,
                              max_det=max_det, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                results.append({"image_id": int(img_id), "category_id": int(c) + 1,
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(sc)})
    finally:
        nms.non_max_suppression = orig_nms
    return results


def run_lvis_eval(lvis_gt, results, img_ids):
    # 09-17: Fixed AP 프로토콜 채택(YOLO-World 논문 및 LVIS long-tail
    # 문헌 관행, Dave et al. -- 이미지당 dets 상한(표준 AP의 max_dets=300)이
    # rare class를 구조적으로 불리하게 만든다는 지적). 이미지당 상한 없이,
    # 클래스당 confidence 상위 10000개만 유지해서 채점한다. 위
    # multi_label=True 수정과 합쳐서 실측(FP32, 공식 4809장 minival):
    # AP 0.126→0.259, APr 0.031→0.177 -- 공개 수치(AP 0.243, APr 0.166)와
    # 6% 이내로 일치. **논문에 disclosure 필요**: 이건 버그 수정이 아니라
    # 프로토콜 선택이므로 "Fixed AP를 썼다"고 명시할 것.
    from lvis import LVISEval, LVISResults
    from collections import defaultdict
    if not results:
        return dict(AP=0.0, AP50=0.0)
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category_id"]].append(r)
    fixed = [r for rs in by_cat.values() for r in sorted(rs, key=lambda x: -x["score"])[:10000]]
    lvis_dt = LVISResults(lvis_gt, fixed, max_dets=-1)
    ev = LVISEval(lvis_gt, lvis_dt, iou_type="bbox")
    ev.params.img_ids = img_ids
    ev.run()
    return ev.get_results()


def compute_lvis_flip_gt_streaming(h_fp, h_models, probe_paths, gt_by_path, grid_specs, imgsz, conf=0.25):
    """LVIS(P=1203)는 8400 anchor x 1203 x probe장수 sim을 리스트로 다 들고 있으면
    (4809장 기준 fp_sims 하나만 ~194GB) CPU RAM이 터진다. 이미지 하나씩 처리해서
    FP/각 모델의 sim을 즉시 소비하고 버리는 스트리밍 방식으로 표준 flip과
    GT_MRR/R@1/lost/gained를 동시에 누적한다. h_models: {mode: SimilarityHarness}."""
    offsets = []
    off = 0
    for (H, W) in grid_specs:
        stride = imgsz // W
        offsets.append((off, H, W, stride))
        off += H * W

    tot = {m: 0 for m in h_models}; fl = {m: 0 for m in h_models}
    recip = {m: [] for m in h_models}
    lost = {m: 0 for m in h_models}; gained = {m: 0 for m in h_models}
    lateral = {m: 0 for m in h_models}

    for i, p in enumerate(probe_paths):
        t = preprocess(p, imgsz, "cpu")
        sf = h_fp.run_image(t, i).sim

        im0 = cv2.imread(p)
        h0, w0 = im0.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
        img_targets = []
        for (cls, cx, cy, bw, bh) in gt_by_path.get(p, []):
            x0 = (cx - bw / 2) * r + left; x1 = (cx + bw / 2) * r + left
            y0 = (cy - bh / 2) * r + top;  y1 = (cy + bh / 2) * r + top
            candidates = []
            for (off0, H, W, s) in offsets:
                col0 = max(0, int(x0 / s - 0.5)); col1 = min(W - 1, int(x1 / s - 0.5) + 1)
                row0 = max(0, int(y0 / s - 0.5)); row1 = min(H - 1, int(y1 / s - 0.5) + 1)
                for row in range(row0, row1 + 1):
                    for col in range(col0, col1 + 1):
                        ccx, ccy = (col + 0.5) * s, (row + 0.5) * s
                        if x0 <= ccx <= x1 and y0 <= ccy <= y1:
                            candidates.append(off0 + row * W + col)
            if not candidates:
                continue
            best = max(candidates, key=lambda a: float(sf[a, cls]))
            img_targets.append((best, cls))

        prob = sf.sigmoid(); mp, c_fp = prob.max(-1); conf_m = mp > conf
        conf_idx = conf_m.nonzero(as_tuple=True)[0]

        for mode, h_q in h_models.items():
            sq = h_q.run_image(t, i).sim
            if len(conf_idx) > 0:
                c_q = sq.argmax(-1)
                fl[mode] += int((c_fp[conf_idx] != c_q[conf_idx]).sum())
                tot[mode] += len(conf_idx)
            for (aidx, cls) in img_targets:
                fp_s = sf[aidx]; q_s = sq[aidx]
                fp_rank = int((fp_s > fp_s[cls]).sum()) + 1
                q_rank = int((q_s > q_s[cls]).sum()) + 1
                recip[mode].append(1.0 / q_rank)
                if fp_rank == 1 and q_rank != 1:
                    lost[mode] += 1
                if fp_rank != 1 and q_rank == 1:
                    gained[mode] += 1
                if fp_rank != 1 and q_rank != 1 and int(fp_s.argmax()) != int(q_s.argmax()):
                    lateral[mode] += 1        # 오답->다른 오답 flip (gt_metrics_for_method와 동일 정의)

        if (i + 1) % 500 == 0:
            print(f"    streaming {i+1}/{len(probe_paths)}장 처리")

    out = {}
    for mode in h_models:
        n = len(recip[mode])
        mrr = sum(recip[mode]) / max(n, 1)
        r1 = sum(1 for x in recip[mode] if x == 1.0) / max(n, 1)
        top1_flip = fl[mode] / max(tot[mode], 1) * 100
        total_flips = lost[mode] + gained[mode] + lateral[mode]
        corrective_rate = gained[mode] / max(total_flips, 1) * 100
        out[mode] = dict(top1_flip=top1_flip, n_flip=tot[mode],
                         mrr=mrr, r1=r1, lost=lost[mode], gained=gained[mode],
                         lateral=lateral[mode], total_flips=total_flips,
                         corrective_rate=corrective_rate, n=n)
    return out


def quantized_weight_mib(model_module):
    """09-19 (사용자 지적) 이전엔 total_elements / (1024**2)로 원소당 1바이트
    (=8bit)를 암묵 가정했다 -- --w-bits를 CLI로 조정 가능하게 만든 이상 이제
    실제 버그다(--w-bits 4면 실제 크기는 절반인데 8bit 기준으로 찍힘). 각
    conv가 이미 들고 있는 self.w_bits(QuantConv2d/AdaRoundQuantConv2d 둘 다
    생성자에서 저장)로 비트 수를 직접 계산한다."""
    total_bits = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            if getattr(m, "hi_cols", None) is not None:      # 09-29: 채널 단위 혼합 정밀도
                per_col = m.conv.weight.numel() // m.conv.weight.shape[1]
                n_hi = int(m.hi_cols.sum())
                total_bits += per_col * (n_hi * m.hi_bits + (m.conv.weight.shape[1] - n_hi) * m.w_bits)
            else:
                total_bits += m.conv.weight.numel() * m.w_bits
    return total_bits / 8 / (1024 * 1024)


MCHECK_RATIO = 1.5   # 10-02 +A: M을 켠 쪽 calib-밖 COCO flip이 이 배수를 넘으면 M을 끈다


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0, scale_reg_weight=1.0,
          h_eval=None, cal_idx=None, cal_weight=1.0,
          recon_iters_ada=1000, recon_iters_strong=2000, qdrop_prob=0.5,
          channelwise_smult=False, identity_aware_margin=True, control_mse=False,
          adaround_learn_act_scale=False, qdrop_brecq_learn_act_scale=True,
          combined_recon_iters=0, combined_stage1="none", w_bits=8, a_bits=8, act_observer="mse",
          brecq_two_stage=True, brecq_act_iters=5000, brecq_batch=2, skip_head=True,
          neck_layerwise=True, neighbor_of_cal=False, aux_mse_weight=0.0,
          adaround_act_observer="minmax",
          combined_learn_alpha=False, combined_alpha_lr=1e-2,
          combined_alpha_reg_weight=1e-2, combined_range_blend=0.0,
          combined_learn_alpha_bias=False, combined_alpha_bias_lr=1e-2,
          combined_alpha_bias_reg_weight=1e-3, combined_alpha_bias_limit=0.5,
          combined_utility_frac=0.0, combined_thresh_w=1.0, combined_box_w=0.5,
          combined_region_dir_weight=0.0, margin_one_sided=False,
          combined_random_sample=False, combined_per_group_anchors=False,
          combined_local_recon_weight=0.0, combined_block_recon_weight=0.0,
          first_last_bits=0, vocab_metric=None, hi_wbit_blocks=(), mig_alphas=(), mig_images=8, hi_col_frac=0.0, mig_tied=False, hi_wbit_convs=(), hi_abit_convs=(), gate_commute=False, attn_quant="none", last_abits=0):
    m = model_cls(w)
    m.set_classes(names)
    if mode == "fp":
        m.fuse(); m.model.to(device).eval()
        return m
    m.fuse()
    if gate_commute:
        # 09-30 설계 방향 1: C2fAttn의 텍스트 게이트를 cv2 뒤로 옮기는 FP 등가 재배치(quant/fusion_quant.py).
        # 양자화 전(wrap_convs 전)에 구조를 바꾼다. 재구성의 FP 기준도 같은 구조여야 하므로 아래 _fpref 참고.
        assert mode in ("naive", "brecq", "qdrop"), f"게이트 교환 미지원 조건: {mode}"
        from quant.fusion_quant import gate_commute_all
        gate_commute_all(m.model)
    # 09-23: CLIP 인코더(clip_model) 제외는 quant_model.ALWAYS_SKIP_NAMES가 담당한다
    # -- set_classes()가 캐싱하는 CLIP vision tower의 patch-embed conv가 양자화 대상에
    # 섞여 모델 크기를 22.8% 과대계상하던 버그(경위는 wrap_convs 주석 참고).
    wrap_convs(m.model, w_bits, a_bits,
               skip_modules=[m.model.model[-1]] if skip_head else None)
    if first_last_bits > 0:
        # 09-28: 저비트 표준 프로토콜 -- stem과 (head 양자화 시) cv2/cv3 마지막 conv를 고정 비트로.
        # 모든 조건에 똑같이 적용된다(calibrate 전이어야 observer/weight scale이 이 비트로 잡힘).
        set_first_last_bits(m.model, first_last_bits)
    if last_abits > 0:
        # 10-01: head cv2/cv3 마지막 1x1 입력 activation만 last_abits(16)로. 모든 조건 동일, calibrate 전.
        set_last_abits(m.model, last_abits)
    if hi_wbit_blocks:
        # 09-29: 혼합 정밀도(weight만 8bit로 되돌릴 블록). 모든 조건에 동일 적용, calibrate 전.
        set_block_wbits(m.model, hi_wbit_blocks, 8)
    if hi_wbit_convs:
        # 09-29: conv 단위 혼합 정밀도(채널 단위 W8과 같은 크기 비교용). 모든 조건 동일, calibrate 전.
        set_conv_wbits(m.model, hi_wbit_convs, 8)
    if hi_abit_convs:
        # 09-29: conv 단위로 입력 activation만 8bit(W4A4 상한 확인용). 모든 조건 동일, calibrate 전.
        set_conv_abits(m.model, hi_abit_convs, 8)
    if attn_quant != "none":
        # 10-01 보강 실험: attention(Linear/matmul)과 (attn_cls면) head contrastive matmul도 8bit로(quant/attn_quant.py).
        # 모든 조건에 동일 적용, gate_commute 뒤·calibrate 전.
        from quant.attn_quant import quantize_attention
        quantize_attention(m.model, attn_quant, 8)
    m.model.to(device).eval()
    if mig_alphas:
        # 09-29: 입력 채널별 scale 이전(quant/migrate.py). 양자화기 변경이라 모든 조건에 동일 적용,
        # calibrate 전(observer/weight scale이 이전된 값 기준으로 잡혀야 함). layer-wise AdaRound
        # (optimize_adaround)는 conv.weight를 FP 입력과 직접 곱해 target을 만들어서 이전과 호환 안 됨.
        # 09-29: combined/combined_m은 Stage 1이 brecq일 때만 허용(Stage 2의 s_mult는 conv당 스칼라라 이전과
        # 독립; layer-wise AdaRound Stage 1은 conv.weight를 FP 입력과 직접 곱해 호환 안 됨).
        assert mode in ("naive", "brecq", "qdrop", "brecq_vm", "qdrop_vm") or \
            (mode in ("combined", "combined_m") and combined_stage1 == "brecq"), f"migration 미지원 조건: {mode}"
        if mig_tied:
            # 09-29: 배포 제약(같은 생산 채널을 받는 소비 conv는 같은 s, residual add로 합쳐지는 생산 채널도
            # 같은 s)을 지키는 버전(quant/channel_graph.py). 아래 독립 버전은 제약 없는 상한.
            from quant.channel_graph import search_and_apply_tied
            search_and_apply_tied(m.model, calib[:mig_images], device, mig_alphas)
        else:
            from quant.migrate import search_and_apply
            search_and_apply(m.model, calib[:mig_images], device, mig_alphas)
    if hi_col_frac > 0:
        # 09-29: 채널 단위 혼합 정밀도(quant/migrate.py::select_hi_cols). 모든 조건 동일, calibrate 전,
        # migration 뒤(이전된 값 기준으로 점수). layer-wise AdaRound 대칭 모드와는 호환 안 됨.
        assert mode in ("naive", "brecq", "qdrop", "brecq_vm", "qdrop_vm"), f"hi-col 미지원 조건: {mode}"
        from quant.migrate import select_hi_cols
        select_hi_cols(m.model, calib[:mig_images], device, hi_col_frac)
    # 09-24: combined만 activation 초기 범위를 MSE 최적(0.0)과 클리핑 없는
    # min-max(1.0) 사이에서 보간할 수 있게 한다. 다른 조건은 0.0 고정이라 영향 없음.
    _fpref = fp
    if gate_commute and mode in ("brecq", "qdrop"):
        # BRECQ neck/head는 conv 단위(layer-wise)로 quant/FP conv를 트리 순서로 짝짓고(_hpairs) FP conv 출력을
        # 목표로 쓴다 -> FP 기준 모델도 같은 게이트 교환 구조여야 cv2_main/cv2_side가 올바른 짝을 갖는다(FP 등가).
        from quant.fusion_quant import gate_commute_all as _gca
        _fpref = model_cls(w); _fpref.set_classes(names); _fpref.fuse(); _gca(_fpref.model)
        _fpref.model.to(device).eval()
    calibrate(m.model, calib, device=device, act_observer=act_observer,
              range_blend=(combined_range_blend if mode in ("combined", "combined_m") else 0.0))
    if mode == "naive":
        pass
    elif mode == "adaround":
        # 09-18 (사용자 지적으로 수정): AdaRound 원 논문(Nagel et al. ICML'20)은
        # 순수 weight-rounding 방법이라 activation LSQ가 없다(claim12) -- 기본
        # False가 맞는 채택(claim4/5/13과 같은 원칙: 원 논문 충실도가 기준).
        # adaround_learn_act_scale=True일 때만 channelwise_smult=False(per-tensor)로
        # 강제(Combined와 granularity 통일, ablation 목적).
        # 09-21: activation calibration은 논문 그대로 min-max -- 다른 방법(BRECQ/QDrop/
        # naive/Combined)은 기본 act_observer가 mse로 바뀌었지만, AdaRound 논문은
        # "min/max of observed activations"를 명시하므로 여기서만 재보정해서 강제한다.
        # adaround_act_observer="mse"로 두면 재보정을 생략해 다른 방법과 같은 관측기를
        # 공유한다(교란변수 격리용 ablation, 기본값 minmax는 기존 동작과 동일).
        if adaround_act_observer != act_observer:
            calibrate(m.model, calib, device=device, act_observer=adaround_act_observer)
        convert_to_adaround(m.model, channelwise_smult=not adaround_learn_act_scale,
                            w_quant_mode="adaround")
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada,
                          verbose=False, learn_act_scale=adaround_learn_act_scale)
    elif mode == "qdrop":
        # 09-18 (이어서): QDrop(Wei et al., ICLR 2022)은 BRECQ 프레임워크를 그대로
        # 물려받아 activation LSQ를 포함한다(claim12) -- AdaRound와 반대로 기본
        # True가 맞는 채택. --no-qdrop-brecq-learn-act-scale로 끄면 이전(claim12
        # 이전) 축소 구현으로 돌아가서 "손잡이 유무" 효과 자체를 볼 수 있음(ablation).
        # optimize_brecq에 qdrop_prob를 넘겨 brecq와 재구성 단위/iters를 완전히 맞추고
        # drop 유무만 단일 변수로 비교한다(09-18 claim13, layer-wise에서 이전됨).
        convert_to_adaround(m.model, channelwise_smult=not qdrop_brecq_learn_act_scale)
        optimize_brecq(m.model, _fpref.model, calib, device, iters=recon_iters_strong,
                       qdrop_prob=qdrop_prob, verbose=False,
                       learn_act_scale=qdrop_brecq_learn_act_scale, batch=brecq_batch,
                       neck_layerwise=neck_layerwise)   # two_stage 기본 False 유지(QDrop 공식=공동 최적화)
    elif mode == "brecq":
        convert_to_adaround(m.model, channelwise_smult=not qdrop_brecq_learn_act_scale)
        optimize_brecq(m.model, _fpref.model, calib, device, iters=recon_iters_strong,
                       verbose=False, learn_act_scale=qdrop_brecq_learn_act_scale,
                       two_stage=brecq_two_stage, act_iters=brecq_act_iters,
                       batch=brecq_batch, neck_layerwise=neck_layerwise)
    elif mode in ("brecq_vm", "qdrop_vm"):
        # 09-28: vocabulary-metric 재구성. 대응 baseline(brecq/qdrop)과 인자를 **전부** 같게 넘기고
        # vocab_metric만 추가한다 -- 짝비교가 metric 단일 변수가 되도록(combined의 Stage 1이
        # neck_layerwise/batch를 안 넘겨 생겼던 교란(설계 문서 §2)을 반복하지 않는다).
        assert vocab_metric is not None, f"{mode}에는 vocab_metric이 필요합니다(--vm-vocab)"
        convert_to_adaround(m.model, channelwise_smult=not qdrop_brecq_learn_act_scale)
        if mode == "qdrop_vm":
            optimize_brecq(m.model, fp.model, calib, device, iters=recon_iters_strong,
                           qdrop_prob=qdrop_prob, verbose=False,
                           learn_act_scale=qdrop_brecq_learn_act_scale, batch=brecq_batch,
                           neck_layerwise=neck_layerwise, vocab_metric=vocab_metric)
        else:
            optimize_brecq(m.model, fp.model, calib, device, iters=recon_iters_strong,
                           verbose=False, learn_act_scale=qdrop_brecq_learn_act_scale,
                           two_stage=brecq_two_stage, act_iters=brecq_act_iters,
                           batch=brecq_batch, neck_layerwise=neck_layerwise,
                           vocab_metric=vocab_metric)
    elif mode in ("combined", "combined_m"):
        # 09-28 combined_m: "combined"와 완전히 같되 **Stage 1(BRECQ 재구성)을 기준선 BRECQ 호출과
        # 같은 neck_layerwise/batch로 맞춘다.** "combined"의 Stage 1은 optimize_brecq에 이 인자를
        # 안 넘겨 함수 기본값 neck_layerwise=True(neck을 conv 단위로 재구성)를 받는데, 기준선
        # BRECQ는 CLI 기본값(block 단위)이다 -- 의도한 차이가 아니라 호출부에서 빠진 인자다.
        # 그래서 Combined-vs-BRECQ 짝비교에 "neck 재구성 단위"가 미격리 교란으로 섞여 있었다.
        # 같은 run에서 brecq / combined / combined_m을 나란히 재면
        #   combined - combined_m = neck 재구성 단위의 효과,
        #   combined_m - brecq    = Stage 2(감독의 종류)만의 효과(LSQ 유무 포함)
        # 로 분리된다. 기존 "combined"는 건드리지 않으므로 기존 결과는 그대로 보존된다.
        # 09-18 claim14로 확정: claim13으로 1단계(optimize_adaround)가 alpha를
        # 훨씬 많이 움직이게 됐는데, 그 목적함수(순수 MSE reconstruction)는
        # margin_loss/s_mult(2단계)가 보호하는 영역(train_cols=S∪H_cal+neighbor_cols)과
        # 무관해서 그 바깥(COCO 전체 Top1_flip/lost, LVIS 1203개 class)으로 손상이
        # 샌다는 게 6-seed로 확인됐다(`PROMPTCAL_CLAIMS_2026-09-15.md` claim14).
        # combined_recon_iters=0(기본값, 1단계 생략=round-to-nearest weight)이
        # 이전(1단계 유지) 대비 COCO_AP/LVIS_AP/APr/Heval_flip/LVIS_lost 5개
        # 지표를 트레이드오프 없이 동시에 개선 -- 새 확정 설계. 이전(1단계 유지)
        # 동작을 재현하려면 combined_recon_iters=recon_iters_ada를 명시적으로
        # 넘길 것(opt-in). baseline(adaround) 조건의 recon_iters_ada는 이 값과
        # 무관 -- 영향 없음.
        # 09-18 (claim14 이어서, 진단 실험): combined_stage1로 1단계 자체를 뭘로
        # 쓸지 선택. "none"(기본, claim14 확정) = round-to-nearest. "adaround" =
        # 이전(claim14 이전) 동작, combined_recon_iters로 iters 조정. "brecq" =
        # BRECQ의 block-wise 재구성(alpha만, LSQ는 안 켬 -- activation scale은
        # 여전히 2단계 margin_loss/s_mult가 전담) -- "BRECQ+LSQ가 margin_loss 없이도
        # decision-preservation을 이기는 게 block-wise 상관 반영 때문인지, margin_loss가
        # 그 위에 추가 기여를 하는지" 분리하는 통제 실험용(제안 방법 변경 아님, 진단
        # 목적 한정 -- 헤드라인 설계는 여전히 "none").
        convert_to_adaround(m.model, channelwise_smult=channelwise_smult)
        # 09-19 (사용자 지적) 두 버그 수정: (a) combined_stage1="adaround"인데
        # combined_recon_iters가 기본값(0)이면 stage1이 조용히 생략돼 "none"과
        # 똑같은 결과가 나왔다 -- 경고 없이 진단 실험이 무의미해질 수 있어서
        # assert로 명시적 에러를 띄운다. (b) "adaround"는 combined_recon_iters,
        # "brecq"는 recon_iters_strong을 써서 서로 다른 노브였다 -- "1단계를
        # adaround로 할까 brecq로 할까"가 이 플래그의 목적인데 예산까지
        # 같이 바뀌면 단일 변수 비교가 깨진다. 이제 둘 다 combined_recon_iters
        # 하나로 통일(기존에 recon_iters_strong 기본값 2000으로 돌렸던 brecq
        # 진단을 재현하려면 --combined-recon-iters 2000을 명시할 것).
        assert mode != "combined_m" or combined_stage1 == "brecq", (
            "combined_m은 Stage 1을 기준선 BRECQ와 맞추는 조건이라 --combined-stage1 brecq에서만 의미가 있다")
        if combined_stage1 != "none":
            assert combined_recon_iters > 0, (
                f"--combined-stage1={combined_stage1}인데 --combined-recon-iters="
                f"{combined_recon_iters}입니다 -- 0이면 1단계가 조용히 생략되고 "
                f"'none'과 동일한 결과가 나옵니다. iters를 명시하세요"
                f"(예: --combined-recon-iters 1000).")
        if combined_stage1 == "adaround":
            optimize_adaround(m.model, fp.model, calib, device, iters=combined_recon_iters,
                              verbose=False)
        elif combined_stage1 == "brecq":
            _s1_kw = (dict(neck_layerwise=neck_layerwise, batch=brecq_batch)
                      if mode == "combined_m" else {})
            optimize_brecq(m.model, fp.model, calib, device, iters=combined_recon_iters,
                           verbose=False, **_s1_kw)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=True, scale_reg_weight=scale_reg_weight,
                                          exclude_from_neighbors=h_eval,
                                          cal_idx=cal_idx, cal_weight=cal_weight,
                                          identity_aware_margin=identity_aware_margin,
                                          control_mse=control_mse,
                                          neighbor_of_cal=neighbor_of_cal,
                                          aux_mse_weight=aux_mse_weight,
                                          margin_one_sided=margin_one_sided,
                                          local_recon_weight=combined_local_recon_weight,
                                          block_recon_weight=combined_block_recon_weight,
                                          random_sample=combined_random_sample,
                                          per_group_anchors=combined_per_group_anchors,
                                          region_dir_weight=combined_region_dir_weight,
                                          utility_stage2_frac=combined_utility_frac,
                                          thresh_w=combined_thresh_w, box_w=combined_box_w,
                                          learn_alpha=combined_learn_alpha,
                                          alpha_lr=combined_alpha_lr,
                                          alpha_reg_weight=combined_alpha_reg_weight,
                                          learn_alpha_bias=combined_learn_alpha_bias,
                                          alpha_bias_lr=combined_alpha_bias_lr,
                                          alpha_bias_reg_weight=combined_alpha_bias_reg_weight,
                                          alpha_bias_limit=combined_alpha_bias_limit,
                                          verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-world.pt")
    ap.add_argument("--coco-root", default="/data/taeho/coco_datasets")
    ap.add_argument("--data", default="configs/coco_local.yaml")
    ap.add_argument("--gt-ann", default=None)
    ap.add_argument("--lvis-ann",
                    default="/data/taeho/lvis_datasets/labels_dl/extracted/lvis/annotations/lvis_v1_minival.json")
    ap.add_argument("--brecq-two-stage", action=argparse.BooleanOptionalAction, default=False,
                    help="09-22: 공식 BRECQ 순차 2단계(1: activation 양자화 끄고 alpha만, 2: weight "
                         "고정 후 LSQ만, act_iters=5000 고정). 1-seed 정식 스케일 비교(runs/85_brecq_"
                         "two_stage)에서 공동 최적화보다 전 지표(COCO_AP 35.55 vs 36.10 등)가 나빴고, "
                         "메인 루프(iters=2000) 대비 act_iters=5000이 추가로 붙어 6-seed 병렬 실행"
                         "시간을 크게 늘려서(09-22 밤) 기본값을 False(공동 최적화)로 되돌림. 공식 "
                         "그대로는 --brecq-two-stage로 켤 수 있음. QDrop 모드에는 애초에 적용 안 됨"
                         "(공식이 공동 최적화라 이 플래그와 무관하게 항상 공동 최적화)")
    ap.add_argument("--brecq-act-iters", type=int, default=5000,
                    help="--brecq-two-stage의 2단계(activation scale) iters (공식 iters_a=5000)")
    ap.add_argument("--brecq-batch", type=int, default=1,
                    help="09-21: BRECQ/QDrop 재구성 스텝당 이미지 수. 공식 COCO 설정(QDrop 논문"
                         "부록 E)은 2지만, 1-seed에서 batch=1 대비 결과가 노이즈 수준으로만 달랐고"
                         "(runs/86_batch2_skiphead) 6-seed 병렬 실행 시간을 눈에 띄게 늘려서"
                         "(09-22 밤 실측, neck-layerwise와 함께 seed당 5배+ 지연) 기본값은 1로"
                         "되돌림 -- 공식 그대로는 --brecq-batch 2로 재현 가능")
    ap.add_argument("--skip-head", action=argparse.BooleanOptionalAction, default=True,
                    help="09-21 확정 기본값(True): detection head(WorldDetect)를 모든 방법에서 "
                         "양자화하지 않음 -- QDrop COCO 프로토콜('we didn't quantize the head but "
                         "applied block reconstruction to backbone and layer reconstruction to "
                         "neck')과 동일. --no-skip-head로 이전(head까지 양자화) 동작 재현 가능(ablation)")
    ap.add_argument("--act-observer", choices=["minmax", "mse"], default="mse",
                    help="09-21 확정 기본값(mse): activation scale 초기화. 공식 BRECQ/QDrop 방식"
                         "(L_2.4 grid search). AdaRound 모드는 이 플래그와 무관하게 항상 min-max로 "
                         "재보정됨(원 논문이 min-max를 명시). --act-observer minmax로 전체를 이전 "
                         "동작(모든 방법 min-max)으로 되돌릴 수 있음")
    ap.add_argument("--adaround-act-observer", choices=["minmax", "mse"], default="minmax",
                    help="AdaRound 모드 전용 activation observer. 기본 minmax(원 논문이 "
                         "'min/max of observed activations'를 명시 -- 기존 동작과 동일). "
                         "mse로 두면 다른 방법(naive/BRECQ/QDrop/Combined)과 같은 관측기를 써서 "
                         "'재구성 단위 차이'와 'activation calibration 차이'를 분리한 단일 변수 "
                         "비교가 된다(교란변수 격리 ablation용)")
    ap.add_argument("--neck-layerwise", action=argparse.BooleanOptionalAction, default=False,
                    help="09-21: BRECQ/QDrop 재구성 단위를 backbone(block-wise)/neck(layer-wise, "
                         "SPPF 이후 head 제외)으로 분리 -- QDrop COCO 프로토콜과 동일. 재구성 대상이 "
                         "17->35개로 늘어 6-seed 병렬 실행 시간이 감당 못 할 만큼 늘었는데(09-22 밤), "
                         "결과 영향은 검증한 적이 없어 기본값을 False(전부 block-wise, 이전 동작)로 "
                         "되돌림. 공식 그대로는 --neck-layerwise로 켤 수 있음 -- 결과 영향은 별도 "
                         "1-seed 진단으로 확인 예정")
    ap.add_argument("--w-bits", type=int, default=8,
                    help="09-18 사용자 지적: 이전엔 wrap_convs(m.model, 8, 8)로 하드코딩돼 "
                         "있어서 bit-width를 CLI로 조정할 방법이 없었다. W8A32/W32A8 같은 "
                         "조합으로 손상이 weight rounding 쪽인지 activation range 쪽인지 "
                         "분리하는 메커니즘 실험에 씀(naive가 FP 대비 -3.26 AP인데 weight만"
                         "건드리는 AdaRound/QDrop/BRECQ는 거의 못 고치고 activation scale만"
                         "만지는 Combined가 크게 회복한다는 정황과 직접 검증하기 위함).")
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--recon-iters-strong", type=int, default=2000)
    ap.add_argument("--qdrop-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--scale-reg-weight", type=float, default=1.0,
                    help="09-19 claim15로 재확정(기본값 10.0→1.0): s_mult(per-tensor 스칼라, "
                         "adaround.py의 AdaRoundQuantConv2d.s_mult)의 (s-1)^2 정규화 강도. "
                         "10.0은 claim14 이전(1단계가 AdaRound-refined weight였던 시절) "
                         "튜닝된 값이라, 1단계가 naive rounding으로 바뀐 뒤(claim14)엔 "
                         "과도한 정규화였다 -- QDrop/BRECQ+LSQ(공정 비교 기준, claim12) "
                         "대비 6-seed 재스윕한 결과 1.0이 COCO_AP/LVIS_AP/APr/LVIS_lost를 "
                         "동시에 개선(PROMPTCAL_CLAIMS_2026-09-15.md claim15). 0.0/0.5처럼 "
                         "너무 낮추면 LVIS_flip이 오히려 악화(원래 이 정규화가 막으려던 "
                         "현상 재현) -- 1.0 근방이 최적 구간.")
    ap.add_argument("--cal-weight", type=float, default=1.0,
                    help="확정값(PROMPTCAL_CURRENT_MODEL.md §8.10). H_cal(20개)에도 S와 동일한 "
                         "margin_loss를 직접 적용하는 가중치. 0.0=off(이전 동작).")
    ap.add_argument("--smult-per-tensor", action=argparse.BooleanOptionalAction, default=True,
                    help="09-16 §8.1 확정 설계(기본 True): Combined의 s_mult을 baseline과 동일한 "
                         "per-tensor 스칼라로 씀 -- claim4, activation quantization granularity "
                         "공정성 + 표준 INT8 엔진 배포 가능성 때문에 채택. §8.11(이전 확정값)과 "
                         "대조하려면 --no-smult-per-tensor로 per-channel 벡터로 되돌릴 수 있음.")
    ap.add_argument("--identity-aware-margin", action=argparse.BooleanOptionalAction, default=True,
                    help="09-16 §8.1 확정 설계(기본 True): margin_loss가 fp_top/q_top을 각자 "
                         "독립적으로 topk해서 class identity 없이 정렬된 값끼리만 비교(top-1/top-2가 "
                         "값을 맞바꿔도 loss=0)하던 blind spot을 막음(claim5) -- fp_idx로 sim_q를 "
                         "gather해서 identity를 고정하고, FP top-k 밖 class의 intrusion도 마지막 "
                         "열에서 같이 탐지(claim5-b, promptcal.py의 margin_loss 참고). "
                         "--no-identity-aware-margin으로 이전 동작(정렬 비교)으로 되돌릴 수 있음.")
    ap.add_argument("--adaround-learn-act-scale", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="09-18 claim12 확정(기본 False): AdaRound 원 논문(Nagel et al. "
                         "ICML'20)은 순수 weight-rounding 방법이라 activation LSQ가 "
                         "없다 -- 그 축소 없는 원 논문 충실 구현이 기본값(claim4/5/13과 "
                         "같은 원칙). True로 켜면(ablation) s_mult를 추가해 AdaRound에도 "
                         "activation scale 학습을 붙일 수 있음(자동 per-tensor 강제).")
    ap.add_argument("--qdrop-brecq-learn-act-scale", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="09-18 claim12 확정(기본 True, 09-18 이전엔 실수로 opt-in "
                         "False였음): QDrop/BRECQ 원 논문(Wei et al. ICLR'22 / Li et al. "
                         "ICLR'21)은 BRECQ 프레임워크를 통해 activation LSQ를 포함한다 -- "
                         "이 손잡이 없이 비교하면 '방법 차이'가 아니라 '구현 축소'와 "
                         "비교하는 셈이라 리뷰어 반론을 못 막는다. s_mult(Combined와 "
                         "동일 메커니즘, 각 방법 자신의 reconstruction loss로 alpha와 "
                         "공동 최적화)를 추가하고 자동으로 per-tensor 강제. "
                         "--no-qdrop-brecq-learn-act-scale로 이전(claim12 이전, 축소) "
                         "동작으로 되돌릴 수 있음(ablation 목적에 한함 -- 확정 비교표에는 "
                         "쓰지 말 것). naive/combined에는 영향 없음.")
    ap.add_argument("--combined-recon-iters", type=int, default=0,
                    help="09-18 claim14로 확정(기본값 0): Combined의 1단계"
                         "(optimize_adaround) iters를 --recon-iters-ada와 별개로 조정"
                         "(baseline인 adaround 조건에는 영향 없음). claim13 이후 1단계가 "
                         "alpha를 많이 움직이는데, 그 목적함수가 margin_loss/s_mult(2단계)가 "
                         "보호하는 영역과 무관해서 그 바깥(COCO 전체 Top1_flip/lost, LVIS)으로 "
                         "새는 손상이 커진다는 게 6-seed로 확인됨 -- 0(1단계 생략, alpha가 "
                         "초기값=round-to-nearest에 남음, optimize_promptcal_scale_neighbor가 "
                         "soft=False로 만들기 때문)이 COCO_AP/LVIS_AP/APr/Heval_flip/LVIS_lost "
                         "5개 지표를 트레이드오프 없이 동시에 개선해 새 기본값으로 채택. "
                         "이전(1단계 유지) 동작을 재현하려면 --combined-recon-iters 1000, "
                         "--combined-stage1도 adaround로 같이 줘야 함.")
    ap.add_argument("--combined-stage1", choices=["none", "adaround", "brecq"], default="none",
                    help="09-18 진단 실험(제안 방법 변경 아님, 헤드라인 설계는 계속 'none'): "
                         "Combined의 1단계로 뭘 쓸지. 'none'(기본, claim14 확정) = "
                         "round-to-nearest. 'adaround' = claim14 이전 동작(--combined-recon-iters로 "
                         "iters 조정). 'brecq' = BRECQ의 block-wise 재구성(alpha만, LSQ는 "
                         "안 켬 -- activation scale은 여전히 margin_loss/s_mult(2단계)가 전담, "
                         "iters=--recon-iters-strong). QDrop+LSQ/BRECQ+LSQ가 margin_loss 없이도 "
                         "decision-preservation에서 Combined를 이기는 게 block-wise 상관 반영 "
                         "때문인지, margin_loss가 그 위에 추가 기여를 하는지 분리하는 통제 실험용.")
    ap.add_argument("--neighbor-of-cal", action="store_true",
                    help="09-20 claim16 방향 2 (진단/실험용, 기본 False): margin_loss의 "
                         "neighbor 보호 범위를 S의 이웃뿐 아니라 H_cal의 이웃까지 넓힘. "
                         "같은 exclude_set(H_eval 포함)을 재사용하므로 held-out 불변식은 "
                         "절대 안 깨짐 -- claim6에서 S의 이웃 풀이 이미 H_cal 크기로 "
                         "포화된다고 확인됐으니, 이건 다른 각도(H_cal 자신의 이웃)에서 "
                         "보호 범위를 넓히는 것.")
    ap.add_argument("--aux-mse-weight", type=float, default=0.0,
                    help="09-20 claim16 방향 3 (진단/실험용, 기본 0.0=off): margin_loss"
                         "(sparse top-k)에 train_cols(S∪H_cal) 전체에 대한 dense "
                         "F.mse_loss를 '더해서'(대체 아님, --control-mse와 다름) 보조 "
                         "신호로 준다. BRECQ-stage1 진단(claim15)에서 margin_loss가 "
                         "BRECQ의 dense reconstruction objective보다 decision-preservation에 "
                         "못한 게 확인돼서, sparse 신호를 dense하게 보강하면 나아지는지 확인.")
    ap.add_argument("--control-mse", action="store_true",
                    help="09-17 claim(baseline엔 activation scale 학습 손잡이가 아예 "
                         "없다) 검증용 control 실험. Combined 빌드 시 margin_loss/"
                         "neighbor-hinge/cal/scale_reg를 전부 끄고 train_cols(S∪H_cal) "
                         "전체에 순수 MSE reconstruction만 적용(s_mult 메커니즘·"
                         "optimizer·iters는 동일 유지) -- promptcal.py의 "
                         "optimize_promptcal_scale_neighbor(control_mse=...) 참고. "
                         "일회성 확인용 플래그라 기본 False, --conditions combined와 "
                         "같이 쓰는 걸 권장(다른 조건은 이 플래그의 영향을 안 받음).")
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="스모크 테스트용: flip/GT/UPIR/lost 등을 계산하는 probe(COCO val2017/LVIS "
                         "minival) 이미지 수를 이만큼으로 제한. 0이면 제한 없음(실제 실행 기본값 -- "
                         "val2017 전체 5000장). 09-16 확인: COCO_AP/S_AP/H_eval_AP(measure_ap, "
                         "--data yaml의 고정 val split 사용)에는 이 옵션이 적용되지 않는다 -- AP는 "
                         "항상 --data가 가리키는 전체 val 세트로 측정됨(AP eval 자체가 병목이 아니라 "
                         "일부러 줄이지 않음, PROMPTCAL_CLAIMS_2026-09-15.md claim10 참고).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--combined-block-recon-weight", type=float, default=0.0,
                    help="09-28 공동 최적화: stage 2 목적함수에 블록 출력 재구성 손실을 추가. "
                         "기존 구조는 순차라서 stage 2(s_mult)가 stage 1(BRECQ)의 결과를 사후에 "
                         "교란한다 -- W8A8에서도 Heval_flip 악화, W4A8 0/3 · W8A6 0/7로 저비트에서 "
                         "전면화. 이 항은 BRECQ와 같은 층위(블록 출력)의 dense 신호를 직접 넣어 "
                         "교란이 아니라 협상이 되게 한다. --combined-learn-alpha와 함께 쓰면 alpha도 "
                         "(재구성+ranking) 아래에서 같이 풀린다 = 논문 4장 Semantic Objective의 형태. "
                         "블록별 상대 MSE 평균이라 스케일 무관. 0.0=꺼짐")
    ap.add_argument("--combined-local-recon-weight", type=float, default=0.0,
                    help="09-27: 논문 4장 Semantic Objective의 'Local reconstruction' 항. "
                         "cv4 입력(region feature)의 상대 제곱오차 ||x_q-x_fp||^2/||x_fp||^2. "
                         "현재 목적함수는 semantic consistency만 있어 2단계가 1단계(BRECQ) "
                         "결과에서 자유롭게 멀어진다(claim14). 대칭 margin_loss가 단측보다 "
                         "나았던 것도 그것이 암묵적 앵커라서로 보이므로, 명시적 앵커를 준다. "
                         "region_dir(방향만)과 달리 크기까지 포함. 0.0=꺼짐")
    ap.add_argument("--combined-random-sample", action="store_true",
                    help="09-27 결함 수정: Combined 2단계의 표본 추출을 it %% n(결정적 순환)에서 "
                         "무작위로. iters=1500/n=256이면 이미지 0~219는 6번, 220~255는 5번 쓰여 "
                         "calibration 이미지에 불균등 가중이 걸리고 Adam 모멘텀이 주기 n과 "
                         "상호작용한다. AdaRound/BRECQ는 09-18에 이미 고쳤는데 promptcal만 "
                         "빠져 있었다. 기본 꺼짐(기존 동작) -- 켜는 것을 권장")
    ap.add_argument("--combined-per-group-anchors", action="store_true",
                    help="09-27 결함 수정: margin 항을 각 컬럼 그룹에서 confident한 anchor에서만 "
                         "계산. 현재는 anchor를 train_cols(S∪H_cal 60개)로 선정하는데 margin은 "
                         "S(40)/H_cal(20)에서 따로 계산해서, FP가 H_cal만 확신하는 anchor에서 "
                         "S-margin을(그 반대도) 계산한다 -- 해당 그룹 top-1이 저확신 class라 "
                         "의미 없는 신호다. H_eval은 어느 쪽에도 안 들어가 held-out 불변식은 "
                         "그대로. 기본 꺼짐(기존 동작) -- 켜는 것을 권장")
    ap.add_argument("--margin-one-sided", action="store_true",
                    help="09-27: margin_loss를 단측(one-sided) hinge로. 기본 꺼짐(대칭, 기존 동작). "
                         "기존 (q_m-fp_m)^2는 margin이 FP보다 **넓어진** 경우도 똑같이 벌하는데, "
                         "넓어지는 건 flip에서 멀어지는 것이라 바람직하다. 실측: margin 항의 "
                         "48.1%%가 넓어진 쪽이고 전체 벌점의 30.4%%가 그걸 억제하는 데 쓰였다. "
                         "즉 대칭 형태는 decision loss가 아니라 margin 공간의 reconstruction "
                         "loss다 -- 이 프로젝트가 neighbor_loss에서 이미 확인한 단측 채택(41번)을 "
                         "margin_loss 본체에도 적용하는 것. relu(fp_m - q_m)^2")
    ap.add_argument("--combined-region-dir-weight", type=float, default=0.0,
                    help="09-25: vocabulary-agnostic 정규화. ContrastiveHead가 "
                         "sim_j = x_hat . w_hat_j 이므로 region embedding의 단위 방향을 "
                         "FP와 맞추면 프롬프트를 하나도 참조하지 않고 seen/unseen 모든 "
                         "vocabulary의 유사도가 함께 보존된다. 초록이 약속한 'calibration "
                         "vocabulary 과적합 억제'를 실제로 수행하는 유일한 항 -- 기존 "
                         "margin/neighbor/scale_reg는 전부 calibration vocabulary 위에서 "
                         "계산되고 neighbor_loss는 COCO-80 폐쇄 구조에서 무력화됐다(claim16). "
                         "외부 vocabulary를 안 쓰므로 공정성/누설 문제가 없다. 0.0=꺼짐")
    ap.add_argument("--combined-utility-frac", type=float, default=0.0,
                    help="09-24: 논문 §4.3 Utility-Constrained Refinement. 마지막 이 비율만큼의 "
                         "iteration에서 threshold-crossing hinge(l_thresh)와 box consistency"
                         "(l_box)를 margin/neighbor 위에 추가한다. 0.0=꺼짐(기존 동작과 "
                         "bit-identical), 0.3=마지막 30%%. 동기: 논문 motivation의 'rank "
                         "preservation alone이 AP를 보장하지 않음'에 직접 대응하는 항인데 "
                         "구현만 돼 있고(semantic_calib.utility_refinement_terms) 확정 "
                         "경로에서 호출된 적이 없다. 둘 다 sparse/저자유도라 claim18-a의 "
                         "과적합 패턴에 걸리지 않는다")
    ap.add_argument("--combined-thresh-w", type=float, default=1.0,
                    help="--combined-utility-frac의 threshold-crossing hinge 가중치")
    ap.add_argument("--combined-box-w", type=float, default=0.5,
                    help="--combined-utility-frac의 box consistency 가중치")
    ap.add_argument("--combined-range-blend", type=float, default=0.0,
                    help="09-24: Combined의 activation 초기 범위를 MSE 최적(0.0, 기존 동작)과 "
                         "클리핑 없는 min-max(1.0) 사이에서 보간. 근거: MSE observer는 "
                         "calibration 분포 기준으로 범위를 잘라내는데 그게 COCO는 개선하고 "
                         "LVIS(held-out vocabulary)는 6/6 seed 전부 악화시킨다(fidelity 문서 "
                         "§7.2) -- 재구성 최적 범위 != cross-vocabulary 최적 범위. "
                         "combined 모드에만 적용되며 다른 조건은 항상 0.0")
    ap.add_argument("--combined-learn-alpha", action="store_true",
                    help="09-24: Combined 2단계에서 alpha(weight rounding)를 s_mult와 함께 "
                         "margin 목적함수로 공동 최적화. 기본 꺼짐(alpha 동결 = 기존 확정 설계). "
                         "동기: s_mult는 conv당 스칼라 52개뿐이라 BRECQ/QDrop(alpha 수백만 개)과 "
                         "자유도 차이가 크다. claim14가 실패한 건 1단계 AdaRound가 MSE 목적함수로 "
                         "alpha를 움직여 보호 범위 밖으로 손상이 샜기 때문이므로, 같은 alpha를 "
                         "margin_loss 아래에서 직접 푸는 것은 별개의 시도다")
    ap.add_argument("--combined-alpha-lr", type=float, default=1e-2,
                    help="--combined-learn-alpha의 alpha용 Adam lr (s_mult의 --lr과 분리). "
                         "기본 1e-2. 공식 AdaRound/BRECQ는 1e-3이지만 여기선 iters=1500 예산 안에 h가 0/1로 수렴하지 못한다(실측 56%%) -- 1e-2에서 99%% 수렴하고 배포되는 hard 모델의 margin도 최저였다")
    ap.add_argument("--combined-alpha-reg-weight", type=float, default=1e-2,
                    help="--combined-learn-alpha의 rounding 정규화 가중치. margin_loss가 mean "
                         "스케일이라 reg도 reduction='mean'으로 맞춰져 있다")
    ap.add_argument("--combined-learn-alpha-bias", action="store_true",
                    help="09-28 (claim21 대응): --combined-learn-alpha의 저자유도 대안. "
                         "alpha 원소별(conv당 최대 수백만 개)이 아니라 conv당 스칼라 1개(alpha_bias, "
                         "s_mult와 같은 granularity)로 반올림을 균일하게 보정한다. 동기: W4A8 "
                         "공동최적화(--combined-learn-alpha)가 붕괴한 원인이 손상 크기가 아니라 "
                         "자유도였다(claim21) -- decision_loss의 reliable anchor는 538개뿐인데 "
                         "alpha는 수백만 개라 calib에 과적합했다. alpha_bias는 전체 모델 기준 "
                         "52~70개로 s_mult와 자릿수가 같다(538개 앵커가 이미 그 자릿수를 유의미하게 "
                         "제약한다는 게 W8A8 확정 결과). 대신 conv 전체를 균일하게만 미니 표현력은 "
                         "원소별 alpha보다 약하다(가설, 검증 필요). --combined-learn-alpha와 동시에 "
                         "켤 수도 있지만 보통 대안으로 단독 사용")
    ap.add_argument("--combined-alpha-bias-lr", type=float, default=1e-2,
                    help="--combined-learn-alpha-bias의 Adam lr")
    ap.add_argument("--combined-alpha-bias-reg-weight", type=float, default=1e-3,
                    help="--combined-learn-alpha-bias의 L2 정규화(alpha_bias를 0=BRECQ 원래 "
                         "반올림 근처로 붙잡아둠). alpha_reg_weight(rectified-sigmoid, 0/1로 "
                         "밀어냄)와 목적이 다르다")
    ap.add_argument("--combined-alpha-bias-limit", type=float, default=0.5,
                    help="--combined-learn-alpha-bias의 |alpha_bias| 상한(forward clamp). "
                         "alpha_bias는 conv 전체 반올림을 균일하게 미는 스칼라라 커지면 "
                         "불확실한 weight를 전부 같은 방향으로 반올림하는 파국이 된다 -- Adam은 "
                         "gradient 크기와 무관하게 스텝당 ~lr씩 움직이므로 1500 iter면 이론상 "
                         "|bias|가 15까지 간다. 기본 0.5 (실측: bias 0.067에서 nearest 대비 "
                         "flip 1.8%%)")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(warn_only=True) 활성화. 같은 seed "
                         "재실행에서 baseline 4개는 bit-identical이지만 Combined만 재현이 안 "
                         "되는데(09-23 실측), 그 비결정 연산을 특정/제거하기 위한 스위치. "
                         "기본 꺼짐 -- 켜면 일부 연산 구현이 바뀌어 기존 확정 수치와 달라질 수 있음")
    ap.add_argument("--torch-seed", type=int, default=None,
                    help="09-23 버그 수정: 기본값이 항상 0으로 고정돼 있어서, --seed로 S/H_cal/"
                         "H_eval 분할은 바뀌어도 재구성 루프의 배치 샘플링(torch.randint 등) "
                         "무작위성은 6-seed 내내 완전히 똑같았다 -- calibration 이미지도 --seed와 "
                         "무관하게 고정(sorted train2017[:calib])이라, COCO_AP/LVIS_AP/lost/"
                         "LVIS_flip/LVIS_lost처럼 분할과 무관한 지표는 6-seed 내내 값이 그대로였다"
                         "(cudnn.deterministic=True라 진짜로 bit-level 동일). 분할 의존 지표"
                         "(S_AP/H_eval_AP/Heval_flip)만 seed 효과를 받고 있었다. 이제 명시하지 "
                         "않으면 --seed를 그대로 따라간다(아래 args.torch_seed 처리) -- 명시하면 "
                         "분할과 학습 무작위성을 분리하는 ablation도 가능")
    ap.add_argument("--attn-quant", choices=["none", "attn", "attn_cls"], default="none",
                    help="10-01: attention Linear/matmul(attn), + head contrastive matmul(attn_cls)을 8bit로 양자화")
    ap.add_argument("--post-build", default="",
                    help="10-02: 진단 스크립트 경로. 빌드 직후(평가 전) 실행하고 평가 없이 종료(PTQ_POST_BUILD와 같음)")
    ap.add_argument("--last-abits", type=int, default=0,
                    help="10-01: >0이면 head cv2/cv3 마지막 1x1 conv의 입력 activation만 이 비트로(16 권장). "
                         "보호 진단에도 같은 값이 쓰이고 진단 캐시 이름에 _la<비트>가 붙는다. 기본 0=꺼짐")
    ap.add_argument("--first-last-bits", type=int, default=0,
                    help="09-28: >0이면 stem 첫 conv와 (head 양자화 시) cv2/cv3 마지막 1x1 conv를 이 "
                         "비트로 고정(W4A4 표준 프로토콜은 8). 모든 조건에 동일 적용. 기본 0=꺼짐(기존 동작)")
    ap.add_argument("--hi-wbit-blocks", default="",
                    help="09-29: weight를 8bit로 유지할 model.model 인덱스(쉼표 구분, 예: 12 또는 1,2,4,12). "
                         "activation 비트는 그대로. 모든 조건에 동일 적용. 기본 빈 값=꺼짐(기존 동작)")
    ap.add_argument("--mig-alphas", default="",
                    help="09-29: 입력 채널별 scale 이전(quant/migrate.py)의 a 후보(쉼표 구분, 예: "
                         "0,0.25,0.5,0.75,1). conv마다 출력 재구성 오차로 a를 고르고 '이전 없음'도 항상 후보. "
                         "모든 조건에 동일 적용. 기본 빈 값=꺼짐(기존 동작)")
    ap.add_argument("--hi-wbit-convs", default="",
                    help="09-29: weight를 8bit로 유지할 conv(model.model 기준 경로, 쉼표 구분, 예: 12.cv2,4.cv1). "
                         "모든 조건에 동일 적용. 기본 빈 값=꺼짐")
    ap.add_argument("--hi-abit-convs", default="",
                    help="09-29: 입력 activation을 8bit로 유지할 conv(model.model 기준 경로, 쉼표 구분). 기본 빈 값=꺼짐")
    ap.add_argument("--protect-budget", type=float, default=0.0,
                    help="09-29 (P0): >0이면 누수 없는 conv 단위 진단(train2017 calibration 이미지, COCO 어휘만)으로 "
                         "양자화 weight 파라미터의 이 비율 안에서 민감 conv를 골라 weight를 8bit로 보호(--hi-wbit-convs와 "
                         "합쳐짐). w-bits>=8이면 규칙상 비용 0이라 아무것도 하지 않는다. 진단 결과는 --protect-cache에 캐시")
    ap.add_argument("--protect-criterion", choices=("coco", "emb", "cls", "box"), default="coco",
                    help="보호 선택 기준(09-29 결정: coco = calibration 어휘 top-1 flip 증가량)")
    ap.add_argument("--protect-images", type=int, default=200, help="보호 진단 이미지 수(train2017 앞쪽)")
    ap.add_argument("--protect-cache", default="configs/protect_cache",
                    help="진단 결과 캐시 디렉터리(모델·이미지 수별). seed·조건과 무관하게 한 번만 계산")
    ap.add_argument("--hi-col-frac", type=float, default=0.0,
                    help="09-29: 채널 단위 혼합 정밀도 -- W4 conv마다 입력 채널 점수 max|W_:j|*max|x_j| 상위 "
                         "이 비율(최소 1개)의 weight 열을 8bit로. 모든 조건에 동일 적용. 기본 0=꺼짐(기존 동작)")
    ap.add_argument("--mig-tied", action="store_true",
                    help="09-29: --mig-alphas를 배포 제약(생산 채널 공유/residual add)을 지키는 버전으로 "
                         "(quant/channel_graph.py). 없으면 conv별 독립 s(상한)")
    ap.add_argument("--mig-images", type=int, default=8, help="migration 탐색에 쓸 calibration 이미지 수")
    ap.add_argument("--vm-vocab", default="configs/vocab_generic.txt",
                    help="09-28: brecq_vm/qdrop_vm의 metric을 정의하는 vocabulary. 'coco' | 'lvis'(oracle, "
                         "평가 vocabulary 누수 -- ablation 전용) | 'identity'(C=I, 방향 보존 재구성) | 이름 파일 경로")
    ap.add_argument("--vm-lam-mean", type=float, default=1.0,
                    help="C = Sigma + lam*mu mu^T. 1.0=비중심 2차 모멘트(순위+절대점수), 0.0=순위만")
    ap.add_argument("--vm-mix", type=float, default=0.5,
                    help="fisher 경로 손실 = (1-mix)*재구성 + mix*vocab-가중 재구성")
    ap.add_argument("--vm-samples", type=int, default=4, help="Fisher 추정용 u~N(0,C) 샘플 수(이미지당)")
    ap.add_argument("--vm-anchor-weight", choices=["conf", "uniform"], default="conf")
    ap.add_argument("--vm-conf-floor", type=float, default=0.05)
    ap.add_argument("--conditions", default="naive,adaround,qdrop,brecq,combined",
                    help="쉼표로 구분된 조건 목록(콤마 뒤 공백 없이). 09-16 추가 -- Combined 변형 "
                         "하나만 볼 때도 항상 5개 조건(특히 QDrop/BRECQ, 900~1400s대)을 다 "
                         "빌드하던 낭비를 줄이기 위함. 예: --conditions naive,combined")
    args = ap.parse_args()
    hi_wbit_blocks = tuple(int(x) for x in args.hi_wbit_blocks.split(",") if x.strip())
    mig_alphas = tuple(float(x) for x in args.mig_alphas.split(",") if x.strip())
    hi_wbit_convs = tuple(x.strip() for x in args.hi_wbit_convs.split(",") if x.strip())
    hi_abit_convs = tuple(x.strip() for x in args.hi_abit_convs.split(",") if x.strip())
    protected = ()
    if args.protect_budget > 0 and args.w_bits < 8:
        import json as _json
        from diag_w4_sensitivity import rank_convs_leakfree, select_protected
        os.makedirs(args.protect_cache, exist_ok=True)
        cache = os.path.join(args.protect_cache, f"{os.path.splitext(os.path.basename(args.model))[0]}"
                                                 f"_n{args.protect_images}_c{args.calib}_img{args.imgsz}"
                                                 f"{f'_la{args.last_abits}' if args.last_abits else ''}"
                                                 f"{f'_aq{args.attn_quant}' if args.attn_quant != 'none' else ''}.json")
        if os.path.exists(cache):
            rows = _json.load(open(cache))
            print(f"[protect] 진단 캐시 사용: {cache}")
        else:
            print(f"[protect] 누수 없는 conv 단위 진단 실행(train2017 {args.protect_images}장, COCO 어휘) -> {cache}")
            rows = rank_convs_leakfree(args.model, args.coco_root, f"cuda:{args.device}" if args.device != "cpu" else "cpu",
                                       n_eval=args.protect_images, n_calib=args.calib, imgsz=args.imgsz,
                                       first_last_bits=args.first_last_bits or 8, last_abits=args.last_abits,
                                       attn_quant=args.attn_quant)
            _json.dump(rows, open(cache, "w"), indent=1)
        protected, used, total = select_protected(rows, args.protect_budget, args.protect_criterion)
        print(f"[protect] 기준={args.protect_criterion} 예산={args.protect_budget:.1%} -> {used:.4f}/{total:.3f}M param: "
              f"{protected}")
        hi_wbit_convs = tuple(dict.fromkeys(hi_wbit_convs + tuple(protected)))
    elif args.protect_budget > 0:
        print(f"[protect] w-bits={args.w_bits} >= 8 -> 보호 단계는 규칙상 비용 0(아무것도 바꾸지 않음)")
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    gt_ann = args.gt_ann or os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    print(f"[args] {vars(args)}")
    # 09-28: --device N이 실제로 어느 물리 GPU에 올라갔는지 로그에 남긴다.
    # CUDA_DEVICE_ORDER를 PCI_BUS_ID로 고정했으므로 여기 찍히는 인덱스는
    # nvidia-smi 인덱스와 같아야 한다 -- 다르면 환경이 바뀐 것이니 멈추고 확인할 것.
    if device != "cpu":
        _p = torch.cuda.get_device_properties(int(args.device))
        print(f"[gpu] --device {args.device} -> nvidia-smi GPU {args.device} "
              f"({_p.name}, {_p.total_memory // 2**20}MiB, uuid={_p.uuid}) "
              f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}")

    if args.torch_seed is None:
        args.torch_seed = args.seed          # 09-23: 기본으로 --seed를 그대로 따라가게(버그 수정)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.deterministic:
        # 09-23: 같은 seed로 두 번 돌렸을 때 naive/AdaRound/QDrop/BRECQ는 모든 지표가
        # bit-identical인데 Combined만 재현되지 않는다(실측: COCO_AP 36.61 vs 36.64,
        # Heval_flip 4.79% vs 5.26%, lost 226 vs 239 -- runs/91 seed0 vs runs/92_env_recheck).
        # cudnn.deterministic은 conv 알고리즘만 고정할 뿐 index/scatter backward의
        # atomicAdd 계열 비결정성은 못 잡는다. 이 플래그를 켜면 결정적 구현이 있는
        # 연산은 그걸 쓰고, 없는 연산은 경고를 띄워 "어디가 비결정적인지"를 특정할 수
        # 있다. 기본 False -- 켜면 일부 연산 구현이 바뀌어 기존 확정 수치와 달라질 수
        # 있으므로 진단/재현성 확보용으로만 쓸 것.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    from lvis import LVIS
    print(f"[lvis] loading GT {args.lvis_ann}")
    lvis_gt = LVIS(args.lvis_ann)
    lvis_img_ids = set(lvis_gt.get_img_ids())

    coco = load_names("coco")
    lvis_names = load_names("lvis")
    from ultralytics import YOLOWorld

    # calib: train2017 (평가 데이터와 완전 분리)
    calib_paths = sorted(glob.glob(os.path.join(args.coco_root, "train2017", "*.jpg")))[:args.calib]
    print(f"[data] calib {len(calib_paths)}장 (train2017)")

    # probe: val2017 전체 -- COCO-80은 5000장 다 쓰고, LVIS는 그중 minival(공식 4809장)과
    # 겹치는 것만 채점
    probe_paths = sorted(glob.glob(os.path.join(args.coco_root, "val2017", "*.jpg")))
    if args.eval_cap > 0:
        probe_paths = probe_paths[:args.eval_cap]
    lvis_probe_paths, lvis_probe_ids = [], []
    for p in probe_paths:
        iid = int(os.path.basename(p).split(".")[0])
        if iid in lvis_img_ids:
            lvis_probe_paths.append(p); lvis_probe_ids.append(iid)
    print(f"[data] probe {len(probe_paths)}장(val2017 전체), 그중 LVIS minival과 겹치는 "
          f"{len(lvis_probe_paths)}장으로 LVIS 채점")

    calib = [preprocess(p, args.imgsz, device) for p in calib_paths]
    # probe(5000장)는 CPU에 미리 다 올려두지 않는다 -- 640x640x3 float32 기준
    # 장당 ~4.9MB, 5000장이면 ~24.6GB인데 이게 프로세스당 고정 비용으로 깔려서
    # 동시 실행 가능한 seed 수를 제한하는 주범이었다. 필요할 때마다
    # preprocess()로 다시 읽는다 -- 디스크 read 속도가 빠르므로 전체 실험
    # 시간(수 시간)에 비해 오버헤드가 무시할 수준.

    coco_gt_by_path = load_coco_gt_by_path(gt_ann, probe_paths)
    lvis_gt_by_path = load_lvis_gt_by_path(lvis_gt, lvis_probe_paths, lvis_probe_ids)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(80)
    S = perm[:40].tolist(); H_cal = perm[40:60].tolist(); H_eval = perm[60:80].tolist()
    H_eval_set = set(H_eval)
    S_set = set(S); H_cal_set = set(H_cal)

    print("[build] FP (COCO-80)")
    fp = build(YOLOWorld, args.model, coco, device, calib, "fp")

    conditions = [c.strip() for c in args.conditions.split(",")]

    _valid = {"naive", "adaround", "qdrop", "brecq", "combined", "combined_m", "brecq_vm", "qdrop_vm"}
    # 09-29: 조건 접미사 -- "brecq+PM"처럼 기반 조건 뒤에 양자화기 옵션을 조건별로 붙인다(한 run에서 짝비교).
    #   P = 보호(누수 없는 진단, --protect-criterion, 예산 --protect-budget 또는 기본 1.5%)
    #   R = 같은 예산 무작위 보호(random.Random(1000), runs/134 rand0과 같은 목록)
    #   H = 같은 예산 HAWQ식 보호(출력 MSE = 'cls' 기준)
    #   M = 공유 제약 scale 이전(--mig-alphas가 없으면 0,0.25,0.5,0.75,1)
    #   G = 텍스트 게이트 교환(09-30 설계 방향 1). G와 P/R/H를 함께 쓰면 C2fAttn cv2(교환으로 사라지는 conv)는
    #       보호 목록에서 빠진다(교환이 그 conv를 대신 다룸 -> 모델이 그만큼 작아짐).
    # 전역 옵션(--protect-budget, --mig-alphas 등)은 기존대로 모든 조건에 적용된다.
    _flag_ok = set("PRHMGA")

    def _split_cond(c):
        base, _, suf = c.partition("+")
        return base, set(suf)

    assert all(_split_cond(c)[0] in _valid and _split_cond(c)[1] <= _flag_ok for c in conditions), \
        f"--conditions에 알 수 없는 값: {[c for c in conditions if _split_cond(c)[0] not in _valid or not _split_cond(c)[1] <= _flag_ok]}"
    assert len(set(conditions)) == len(conditions), "--conditions에 중복이 있음"
    _all_flags = set().union(*(_split_cond(c)[1] for c in conditions))
    cond_lists = {"P": (), "R": (), "H": ()}
    if (_all_flags & {"P", "R", "H"}) and args.w_bits < 8:
        import json as _json, random as _random
        from diag_w4_sensitivity import rank_convs_leakfree, select_protected
        _budget = args.protect_budget if args.protect_budget > 0 else 0.015
        os.makedirs(args.protect_cache, exist_ok=True)
        _cache = os.path.join(args.protect_cache, f"{os.path.splitext(os.path.basename(args.model))[0]}"
                                                  f"_n{args.protect_images}_c{args.calib}_img{args.imgsz}"
                                                 f"{f'_la{args.last_abits}' if args.last_abits else ''}"
                                                 f"{f'_aq{args.attn_quant}' if args.attn_quant != 'none' else ''}.json")
        if os.path.exists(_cache):
            _rows = _json.load(open(_cache))
        else:
            print(f"[protect] 누수 없는 conv 단위 진단 실행 -> {_cache}")
            _rows = rank_convs_leakfree(args.model, args.coco_root, device, n_eval=args.protect_images,
                                        n_calib=args.calib, imgsz=args.imgsz, first_last_bits=args.first_last_bits or 8,
                                        last_abits=args.last_abits, attn_quant=args.attn_quant)
            _json.dump(_rows, open(_cache, "w"), indent=1)
        cond_lists["P"] = tuple(select_protected(_rows, _budget, args.protect_criterion)[0])
        cond_lists["H"] = tuple(select_protected(_rows, _budget, "cls")[0])
        # R: runs/134 random_sets.json(rand0)과 같은 절차 -- 진단 행 순서(groups_of 순서)로 섞고 예산 ±10%.
        _convs = [(r["name"], r["mparam"]) for r in _rows]
        _target = _budget * sum(p for _, p in _convs)
        _rng = _random.Random(1000)
        while True:
            _order = _convs[:]; _rng.shuffle(_order); _sel = []; _tot = 0.0
            for n_, p_ in _order:
                if _tot + p_ <= _target * 1.1:
                    _sel.append(n_); _tot += p_
                if _tot >= _target * 0.9:
                    break
            if _target * 0.9 <= _tot <= _target * 1.1:
                break
        cond_lists["R"] = tuple(_sel)
        for k_, v_ in cond_lists.items():
            if k_ in _all_flags:
                print(f"[cond+{k_}] 예산 {_budget:.1%} -> {list(v_)}")
    elif _all_flags & {"P", "R", "H"}:
        print(f"[cond] w-bits={args.w_bits} >= 8 -> +P/+R/+H는 규칙상 no-op")
    vocab_metric = None
    if {"brecq_vm", "qdrop_vm"} & set(conditions):
        vm_names, vm_identity = load_vocab(args.vm_vocab, coco, lvis_names)
        if not vm_identity and args.vm_vocab not in ("coco", "lvis"):
            _overlap = {n.lower() for n in vm_names} & {n.lower() for n in list(coco) + list(lvis_names)}
            if _overlap:
                print(f"[warn][vm] vocab 파일에 COCO/LVIS 이름 {len(_overlap)}개가 섞여 있음 "
                      f"(예: {sorted(_overlap)[:5]}) -- disjoint 주장을 하려면 빼야 함")
        print(f"[vm] text bank 인코딩: {args.vm_vocab} ({len(vm_names)}개, identity={vm_identity})")
        bank = encode_text_bank(YOLOWorld, args.model, vm_names, device)
        vocab_metric = VocabMetric(bank, lam_mean=args.vm_lam_mean, identity=vm_identity,
                                   anchor_weight=args.vm_anchor_weight, conf_floor=args.vm_conf_floor,
                                   n_samples=args.vm_samples, mix=args.vm_mix, seed=args.torch_seed)
        print(f"[vm] C 유효 차원(participation ratio) {vocab_metric.effective_rank():.1f}/{bank.shape[1]}")
    # 09-18: reconstruction 계열 조건 간 iteration 예산이 다르면 "방법 차이"와
    # "최적화 예산 차이"가 섞여서 A vs B 비교가 단일 변수가 아니게 된다. 기본값
    # (ada=1000, strong=2000)이 그 상태라 명시적으로 경고만 띄운다 -- 예산을 맞추려면
    # --recon-iters-ada 2000 처럼 같은 값으로 주면 된다.
    if "adaround" in conditions and ({"qdrop", "brecq"} & set(conditions)) \
            and args.recon_iters_ada != args.recon_iters_strong:
        print(f"[warn] 재구성 예산 불일치: adaround={args.recon_iters_ada} iters vs "
              f"qdrop/brecq={args.recon_iters_strong} iters. 방법 간 비교에 "
              f"'최적화 예산'이 교란변수로 섞인다(맞추려면 --recon-iters-ada "
              f"{args.recon_iters_strong}).")
    models, calib_time = {}, {}
    # 09-29: 조건마다 빌드 직전에 RNG 상태를 루프 진입 시점으로 되돌린다. 이전에는 앞 조건이 소비한 난수
    # (MSE observer 표본 추출, BRECQ 배치 샘플링, QDrop 마스크)만큼 뒤 조건의 RNG 위치가 밀려서 조건 순서가
    # 결과를 바꿨다(§5.2). 복원하면 모든 조건이 "첫 번째 조건"과 같은 상태에서 시작한다 -- 첫 조건은 이전과 동일.
    import random as _py_random
    import numpy as _np
    _rng_snapshot = (torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     _np.random.get_state(), _py_random.getstate())
    model_mib_by = {}
    for cond in conditions:
        mode, _flags0 = _split_cond(cond)
        # 10-02 +A: 이전(M)을 누수 없는 사전 검사로 켤지 정한다. M 끔/켬 두 후보를 모두 빌드(각각 같은 RNG 상태에서
        # 출발 -- 단독 조건과 bit-identical)하고, calibration에 쓰지 않은 train2017 이미지 + COCO 어휘에서 FP 대비
        # top-1 flip을 잰다. M을 켠 쪽 flip이 MCHECK_RATIO배를 넘으면 M을 끈 모델을 쓴다(runs/155 검증).
        _variants = [_flags0] if "A" not in _flags0 else [(_flags0 - {"A", "M"}), (_flags0 - {"A"}) | {"M"}]
        _built = []
        t_cond = time.perf_counter()
        for flags in _variants:
            torch.set_rng_state(_rng_snapshot[0])
            if _rng_snapshot[1] is not None:
                torch.cuda.set_rng_state_all(_rng_snapshot[1])
            _np.random.set_state(_rng_snapshot[2]); _py_random.setstate(_rng_snapshot[3])
            c_hi_wbit = tuple(dict.fromkeys(hi_wbit_convs + sum((cond_lists[f] for f in "PRH" if f in flags), ())))
            if "G" in flags:
                _attn_idx = {str(i) for i, b in enumerate(fp.model.model) if type(b).__name__ == "C2fAttn"}
                c_hi_wbit = tuple(n for n in c_hi_wbit
                                  if not (n.split(".")[0] in _attn_idx and n.split(".")[1:2] == ["cv2"]))
            c_mig_alphas = mig_alphas if mig_alphas else ((0.0, 0.25, 0.5, 0.75, 1.0) if "M" in flags else ())
            c_mig_tied = args.mig_tied or ("M" in flags)
            print(f"[build] {cond}" + (f" -> 후보 {'+'.join(sorted(flags))}" if len(_variants) > 1 else "") + (f"  (W8 보호 {len(c_hi_wbit)}개 {list(c_hi_wbit)}, 이전={'공유' if c_mig_tied else '독립'}"
                                       f"{' 켬' if c_mig_alphas else ' 끔'}, 게이트 교환={'G' in flags})" if flags else ""))
            t0 = time.perf_counter()
            _m = build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
                                 iters=args.iters, pidx=S, lr=args.lr, k=args.k,
                                 neighbor_k=args.neighbor_k, neighbor_weight=args.neighbor_weight,
                                 scale_reg_weight=args.scale_reg_weight, h_eval=H_eval,
                                 # 09-16 버그 수정(claim a): cal_weight==0일 때 cal_idx까지 None으로
                                 # 넘기면, promptcal.py의 confident-anchor 선정 기준인 train_cols가
                                 # S∪H_cal(60) 대신 S(40)로 줄어들어 anchor 풀 자체가 바뀐다. 그러면
                                 # cal_weight 0 vs 1 비교가 "H_cal margin_loss 유무"와 "anchor 풀
                                 # 크기" 두 변수를 동시에 바꾸게 돼서 단일 변수 ablation이 깨진다.
                                 # cal_idx는 항상 넘기고, ml_cal 추가 여부는 promptcal.py 내부의
                                 # `cal_weight > 0` 게이트에만 맡긴다.
                                 cal_idx=H_cal,
                                 cal_weight=args.cal_weight,
                                 recon_iters_ada=args.recon_iters_ada,
                                 recon_iters_strong=args.recon_iters_strong,
                                 qdrop_prob=args.qdrop_prob,
                                 channelwise_smult=not args.smult_per_tensor,
                                 identity_aware_margin=args.identity_aware_margin,
                                 control_mse=args.control_mse,
                                 adaround_learn_act_scale=args.adaround_learn_act_scale,
                                 qdrop_brecq_learn_act_scale=args.qdrop_brecq_learn_act_scale,
                                 combined_recon_iters=args.combined_recon_iters,
                                 combined_stage1=args.combined_stage1,
                                 w_bits=args.w_bits, a_bits=args.a_bits, act_observer=args.act_observer,
                                 brecq_two_stage=args.brecq_two_stage, brecq_act_iters=args.brecq_act_iters,
                                 brecq_batch=args.brecq_batch, skip_head=args.skip_head, neck_layerwise=args.neck_layerwise,
                                 neighbor_of_cal=args.neighbor_of_cal,
                                 aux_mse_weight=args.aux_mse_weight,
                                 adaround_act_observer=args.adaround_act_observer,
                                 combined_learn_alpha=args.combined_learn_alpha,
                                 combined_alpha_lr=args.combined_alpha_lr,
                                 combined_alpha_reg_weight=args.combined_alpha_reg_weight,
                                 combined_learn_alpha_bias=args.combined_learn_alpha_bias,
                                 combined_alpha_bias_lr=args.combined_alpha_bias_lr,
                                 combined_alpha_bias_reg_weight=args.combined_alpha_bias_reg_weight,
                                 combined_alpha_bias_limit=args.combined_alpha_bias_limit,
                                 combined_range_blend=args.combined_range_blend,
                                 combined_utility_frac=args.combined_utility_frac,
                                 combined_thresh_w=args.combined_thresh_w,
                                 combined_box_w=args.combined_box_w,
                                 combined_region_dir_weight=args.combined_region_dir_weight,
                                 margin_one_sided=args.margin_one_sided,
                                 combined_random_sample=args.combined_random_sample,
                                 combined_per_group_anchors=args.combined_per_group_anchors,
                                 combined_local_recon_weight=args.combined_local_recon_weight,
                                 combined_block_recon_weight=args.combined_block_recon_weight,
                                 first_last_bits=args.first_last_bits, vocab_metric=vocab_metric,
                                 hi_wbit_blocks=hi_wbit_blocks, mig_alphas=c_mig_alphas,
                                 mig_images=args.mig_images, hi_col_frac=args.hi_col_frac,
                                 mig_tied=c_mig_tied, hi_wbit_convs=c_hi_wbit,
                                 hi_abit_convs=hi_abit_convs, gate_commute=("G" in flags),
                                 attn_quant=args.attn_quant, last_abits=args.last_abits)
            _built.append((flags, _m))
        if len(_built) == 1:
            models[cond] = _built[0][1]
        else:
            from diag_w4_sensitivity import mcheck_flip
            t_chk = time.perf_counter()
            (_f_off, _m_off), (_f_on, _m_on) = _built
            _fl_off, _fl_on = mcheck_flip([_m_off, _m_on], fp, args.model, coco, args.coco_root, args.calib,
                                          args.imgsz, device)
            _use_m = _fl_on <= MCHECK_RATIO * _fl_off
            models[cond] = _m_on if _use_m else _m_off
            print(f"  [+A] calib 밖 COCO flip: M 끔 {_fl_off:.2f}% / M 켬 {_fl_on:.2f}% -> "
                  f"M {'켬' if _use_m else '끔'} (기준 {MCHECK_RATIO}배, 검사 {time.perf_counter() - t_chk:.1f}s)", flush=True)
            del _m_off, _m_on, _built
            torch.cuda.empty_cache()
        t0 = t_cond
        calib_time[cond] = time.perf_counter() - t0
        model_mib_by[cond] = quantized_weight_mib(models[cond].model)
        print(f"  {cond} 빌드 {calib_time[cond]:.1f}s  (이론적 크기 {model_mib_by[cond]:.2f} MiB)")
    # --conditions로 일부만 돌릴 때 "adaround"가 없을 수 있음 -- AdaRound 기반
    # 모드(adaround/qdrop/brecq/combined)는 전부 같은 conv/양자화 구조라 이론적
    # 모델 크기가 동일하므로, 돌아간 것 중 아무거나 골라도 된다(naive만 돈 경우는 naive로).
    _mib_mode = next((m for m in ["adaround", "qdrop", "brecq", "combined", "brecq_vm", "qdrop_vm"]
                      if m in models), conditions[0])
    model_mib = quantized_weight_mib(models[_mib_mode].model)
    _post = args.post_build or os.environ.get("PTQ_POST_BUILD")
    if _post:
        # 10-01: 진단용 훅 -- 빌드 직후(평가 전) 모델을 그대로 넘겨 스크립트를 실행하고 종료(주 실험 경로 영향 없음).
        exec(open(_post).read(), {**globals(), **locals()})   # 한 이름 공간: 스크립트 안 함수끼리 서로 보이게
        return

    # ---------------- COCO-80: FP sim 1회 계산 + GT 매칭 ----------------
    print(f"\n[gt] COCO-80 FP sim 계산 + anchor 매칭 ({len(probe_paths)}장)")
    h_fp = SimilarityHarness(fp.model, device=device)
    grid_specs = get_grid_specs(h_fp, preprocess(probe_paths[0], args.imgsz, "cpu"))
    fp_sims = [h_fp.run_image(preprocess(p, args.imgsz, "cpu"), i).sim
              for i, p in enumerate(probe_paths)]
    h_fp.close()
    coco_gt_targets = build_gt_targets(fp_sims, probe_paths, coco_gt_by_path, grid_specs, args.imgsz)
    n_coco_gt = sum(len(t) for t in coco_gt_targets)
    print(f"  COCO GT anchor 수={n_coco_gt}")

    results = {}
    for mode in conditions:
        h_q = SimilarityHarness(models[mode].model, device=device)
        q_sims = [h_q.run_image(preprocess(p, args.imgsz, "cpu"), i).sim
                 for i, p in enumerate(probe_paths)]
        h_q.close()
        heval_flip, n1 = group_flip(fp_sims, q_sims, H_eval)
        top1_flip, n2 = standard_flip(fp_sims, q_sims)
        coco_gt_res = gt_metrics_for_method(fp_sims, q_sims, coco_gt_targets, H_eval_set,
                                            S_set=S_set, H_cal_set=H_cal_set)
        del q_sims          # measure_ap()이 model.val() 내부에서 DataLoader worker를
        free_cpu_mem()      # fork하므로, 그 전에 이 조건의 큰 q_sims부터 비워둔다
        (coco_ap, coco_ap50), coco_pc = measure_ap(models[mode], args.data, args.imgsz, args.device)
        s_ap = subset_map(coco_pc, S); heval_ap = subset_map(coco_pc, H_eval)
        lrg = coco_gt_res["lost_rate_by_group"]; lbg = coco_gt_res["lost_by_group"]; dbg = coco_gt_res["denom_by_group"]
        print(f"  [COCO-80][{mode}] AP={coco_ap:.2f} S_AP={s_ap:.2f} H_eval_AP={heval_ap:.2f} "
              f"Heval_flip={heval_flip:.2f}% Top1_flip={top1_flip:.2f}% "
              f"GT_MRR={coco_gt_res['mrr']:.4f} GT_R@1={coco_gt_res['r1']:.4f} "
              f"lost={coco_gt_res['lost']} gained={coco_gt_res['gained']} lateral={coco_gt_res['lateral']} "
              f"corrective_rate={coco_gt_res['corrective_rate']:.2f}% UPIR={coco_gt_res['upir']:.2f}%")
        print(f"    [lost by group] S={lbg['S']}/{dbg['S']}({lrg['S']:.2f}%) "
              f"H_cal={lbg['H_cal']}/{dbg['H_cal']}({lrg['H_cal']:.2f}%) "
              f"H_eval={lbg['H_eval']}/{dbg['H_eval']}({lrg['H_eval']:.2f}%)")
        results[mode] = dict(coco_ap=coco_ap, s_ap=s_ap, heval_ap=heval_ap, heval_flip=heval_flip,
                             top1_flip=top1_flip, coco_gt=coco_gt_res, calib_time=calib_time[mode],
                             lost_rate_by_group=lrg)

    # FP32 COCO-80 AP -- 반드시 fp의 vocabulary를 LVIS로 바꾸기 전에 재야 함
    # (바꾼 뒤 COCO 80-class validator에 넣으면 confusion matrix index가 깨짐)
    (fp_coco_ap, fp_coco_ap50), _ = measure_ap(fp, args.data, args.imgsz, args.device)

    # ---------------- LVIS-1203(minival): 이식, 스트리밍(이미지 1장씩) ----------------
    print(f"\n[switch] 전부 LVIS-1203 vocab으로 이식 (minival {len(lvis_probe_paths)}장)")
    switch_vocab(fp, lvis_names, device)
    for mode in conditions:
        switch_vocab(models[mode], lvis_names, device)

    h_fp = SimilarityHarness(fp.model, device=device)
    lvis_grid_specs = get_grid_specs(h_fp, preprocess(lvis_probe_paths[0], args.imgsz, "cpu"))
    h_models = {mode: SimilarityHarness(models[mode].model, device=device) for mode in conditions}

    print(f"  [streaming] flip/GT_MRR/R@1/lost/gained 계산 중 ({len(lvis_probe_paths)}장 x {len(conditions)}조건)")
    lvis_stream_res = compute_lvis_flip_gt_streaming(h_fp, h_models, lvis_probe_paths,
                                                     lvis_gt_by_path, lvis_grid_specs, args.imgsz)
    h_fp.close()
    for h_q in h_models.values():
        h_q.close()
    n_lvis_gt = lvis_stream_res[conditions[0]]["n"]
    print(f"  LVIS GT anchor 수={n_lvis_gt}")
    for mode in conditions:
        r = lvis_stream_res[mode]
        print(f"  [LVIS-flip/GT][{mode}] Top1_flip={r['top1_flip']:.2f}%(n={r['n_flip']}) "
              f"GT_MRR={r['mrr']:.4f} GT_R@1={r['r1']:.4f} lost={r['lost']} gained={r['gained']} "
              f"lateral={r['lateral']} corrective_rate={r['corrective_rate']:.2f}%")

    for mode in conditions:
        print(f"[predict+AP] {mode} (minival {len(lvis_probe_paths)}장) ...")
        preds = predict_lvis_results(models[mode], lvis_probe_paths, lvis_probe_ids, args.imgsz, device)
        lvis_ap_res = run_lvis_eval(lvis_gt, preds, lvis_probe_ids)
        print(f"  [LVIS-AP][{mode}] AP={lvis_ap_res.get('AP',0):.4f} AP50={lvis_ap_res.get('AP50',0):.4f} "
              f"APr={lvis_ap_res.get('APr',0):.4f} APc={lvis_ap_res.get('APc',0):.4f} "
              f"APf={lvis_ap_res.get('APf',0):.4f}")
        r = lvis_stream_res[mode]
        results[mode].update(lvis_ap=lvis_ap_res.get("AP", 0.0), lvis_ap50=lvis_ap_res.get("AP50", 0.0),
                             lvis_apr=lvis_ap_res.get("APr", 0.0), lvis_apc=lvis_ap_res.get("APc", 0.0),
                             lvis_apf=lvis_ap_res.get("APf", 0.0),
                             lvis_top1_flip=r["top1_flip"],
                             lvis_gt=dict(mrr=r["mrr"], r1=r["r1"], lost=r["lost"], gained=r["gained"],
                                         lateral=r["lateral"], corrective_rate=r["corrective_rate"]))

    print("\n[predict+AP] FP32 (참고, LVIS) ...")
    preds_fp = predict_lvis_results(fp, lvis_probe_paths, lvis_probe_ids, args.imgsz, device)
    fp_lvis = run_lvis_eval(lvis_gt, preds_fp, lvis_probe_ids)

    print("\n" + "=" * 100)
    control_tag = " [CONTROL-MSE: combined는 margin_loss 대신 순수 MSE reconstruction]" if args.control_mse else ""
    lsq_bits = []
    if args.adaround_learn_act_scale:
        lsq_bits.append("AdaRound+LSQ(ablation)")
    if not args.skip_head:
        lsq_bits.append("head 양자화(QDrop COCO 프로토콜과 다름, ablation)")
    if args.brecq_batch != 2:
        lsq_bits.append(f"BRECQ/QDrop batch={args.brecq_batch}(공식 COCO=2와 다름, ablation)")
    if not args.brecq_two_stage:
        lsq_bits.append("BRECQ 공동 최적화(공식 순차 2단계와 다름, ablation)")
    if args.act_observer != "mse":
        lsq_bits.append(f"act-observer={args.act_observer}(공식 mse와 다름, ablation)")
    if not args.neck_layerwise:
        lsq_bits.append("neck도 block-wise(공식 layer-wise와 다름, ablation)")
    if not args.qdrop_brecq_learn_act_scale:
        lsq_bits.append("QDrop/BRECQ LSQ 꺼짐(claim12 이전 축소구현, ablation)")
    if "combined_m" in conditions:
        lsq_bits.append("combined_m=Stage 1을 기준선 BRECQ와 같은 neck_layerwise/batch로 맞춘 대조(09-28)")
    if args.combined_stage1 != "none":
        lsq_bits.append(f"Combined 1단계={args.combined_stage1}(진단 실험, 헤드라인 아님)")
    if args.neighbor_of_cal:
        lsq_bits.append("neighbor_of_cal(claim16 방향2, 진단 실험)")
    if args.aux_mse_weight > 0:
        lsq_bits.append(f"aux_mse_weight={args.aux_mse_weight}(claim16 방향3, 진단 실험)")
    if args.first_last_bits > 0:
        lsq_bits.append(f"첫/마지막 레이어 {args.first_last_bits}bit")
    if args.last_abits > 0:
        lsq_bits.append(f"head 마지막 conv 입력 A{args.last_abits}")
    if hi_wbit_blocks:
        lsq_bits.append(f"W8 유지 블록 {list(hi_wbit_blocks)}")
    if protected:
        lsq_bits.append(f"보호(자동, {args.protect_criterion}, 예산 {args.protect_budget:.1%})")
    if hi_wbit_convs:
        lsq_bits.append(f"W8 유지 conv {list(hi_wbit_convs)}")
    if hi_abit_convs:
        lsq_bits.append(f"A8 유지 conv {list(hi_abit_convs)}")
    if args.hi_col_frac > 0:
        lsq_bits.append(f"W8 입력 채널 상위 {args.hi_col_frac:.1%}")
    if mig_alphas:
        lsq_bits.append(f"scale 이전{'(공유 제약)' if args.mig_tied else '(독립, 상한)'} "
                        f"a∈{list(mig_alphas)} ({args.mig_images}장)")
    if vocab_metric is not None:
        lsq_bits.append(f"vm: vocab={args.vm_vocab} lam={args.vm_lam_mean} mix={args.vm_mix} "
                        f"K={args.vm_samples} w={args.vm_anchor_weight}")
    lsq_tag = f" [{', '.join(lsq_bits)}]" if lsq_bits else ""
    print(f" 공식 데이터 설정(calib=train2017 {len(calib_paths)}장, LVIS=공식 minival) -- seed {args.seed}{control_tag}{lsq_tag}")
    if args.eval_cap > 0:
        print(f" 주의: --eval-cap={args.eval_cap}은 COCO_AP/S_AP/H_eval_AP(measure_ap, --data yaml의 "
              f"고정 val split 사용)에는 적용 안 됨 -- 항상 전체 val 세트로 측정됨. "
              f"LVIS_AP/APr/APc/APf는 probe_paths에서 파생돼 --eval-cap이 적용됨(위 LVIS 채점 "
              f"{len(lvis_probe_paths)}장 참고). flip/GT/UPIR/lost(아래 두 번째 표)도 --eval-cap 적용됨.")
    print("=" * 100)
    print(f"{'':>10} | {'COCO AP':>8} | {'S_AP':>7} | {'H_eval_AP':>9} | {'LVIS AP':>8} | {'APr':>6} | {'APc':>6} | {'APf':>6}")
    print(f"{'FP32':>10} | {fp_coco_ap:>8.2f} | {'-':>7} | {'-':>9} | {fp_lvis.get('AP',0):>8.4f} | "
          f"{fp_lvis.get('APr',0):>6.4f} | {fp_lvis.get('APc',0):>6.4f} | {fp_lvis.get('APf',0):>6.4f}")
    for mode in conditions:
        r = results[mode]
        print(f"{mode:>10} | {r['coco_ap']:>8.2f} | {r['s_ap']:>7.2f} | {r['heval_ap']:>9.2f} | "
              f"{r['lvis_ap']:>8.4f} | {r['lvis_apr']:>6.4f} | {r['lvis_apc']:>6.4f} | {r['lvis_apf']:>6.4f}")

    print("\n" + "=" * 100)
    print(" flip / GT / UPIR / 비용")
    print("=" * 100)
    print(f"{'':>10} | {'Heval_flip':>10} | {'Top1_flip':>9} | {'UPIR':>6} | {'lost':>5} | "
          f"{'CorrRate':>8} | {'LVIS_flip':>9} | {'LVIS_lost':>9} | {'L_CorrR':>8} | {'calib(s)':>9}")
    for mode in conditions:
        r = results[mode]
        print(f"{mode:>10} | {r['heval_flip']:>9.2f}% | {r['top1_flip']:>8.2f}% | "
              f"{r['coco_gt']['upir']:>5.2f}% | {r['coco_gt']['lost']:>5} | "
              f"{r['coco_gt']['corrective_rate']:>7.2f}% | "
              f"{r['lvis_top1_flip']:>8.2f}% | {r['lvis_gt']['lost']:>9} | "
              f"{r['lvis_gt']['corrective_rate']:>7.2f}% | {r['calib_time']:>9.1f}")

    print("\n" + "=" * 100)
    print(" lost 그룹별 분해 (S/H_cal/H_eval) -- \"lost가 H_eval에 몰리는가\" 가설 확인용")
    print("=" * 100)
    print(f"{'':>10} | {'lost_rate_S':>11} | {'lost_rate_H_cal':>15} | {'lost_rate_H_eval':>16}")
    for mode in conditions:
        lrg = results[mode]["lost_rate_by_group"]
        print(f"{mode:>10} | {lrg['S']:>10.2f}% | {lrg['H_cal']:>14.2f}% | {lrg['H_eval']:>15.2f}%")

    print(f"\n이론적 모델 크기: {model_mib:.2f} MiB")
    if len(set(round(v, 4) for v in model_mib_by.values())) > 1:
        print("조건별 이론적 크기: " + ", ".join(f"{k} {v:.2f} MiB" for k, v in model_mib_by.items()))


if __name__ == "__main__":
    main()
