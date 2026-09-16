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
import cv2, numpy as np, torch

if not hasattr(np, "float"):
    np.float = float

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import SimilarityHarness
from quant.quant_model import wrap_convs, calibrate
from quant.adaround import convert_to_adaround, optimize_adaround, AdaRoundQuantConv2d, free_cpu_mem
from quant.fake_quant import QuantConv2d
from quant.brecq import optimize_brecq
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


def predict_lvis_results(model, img_paths, img_ids, imgsz, device, conf=0.001, max_det=300):
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
    return results


def run_lvis_eval(lvis_gt, results, img_ids):
    from lvis import LVISEval, LVISResults
    if not results:
        return dict(AP=0.0, AP50=0.0)
    lvis_dt = LVISResults(lvis_gt, results, max_dets=300)
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
    total = 0
    for m in model_module.modules():
        if isinstance(m, (AdaRoundQuantConv2d, QuantConv2d)):
            total += m.conv.weight.numel()
    return total / (1024 * 1024)


def build(model_cls, w, names, device, calib, mode, fp=None, iters=1500, pidx=None,
          lr=1e-2, k=5, neighbor_k=5, neighbor_weight=1.0, scale_reg_weight=10.0,
          h_eval=None, cal_idx=None, cal_weight=1.0,
          recon_iters_ada=1000, recon_iters_strong=2000, qdrop_prob=0.5,
          channelwise_smult=False, identity_aware_margin=True):
    m = model_cls(w)
    m.set_classes(names)
    if mode == "fp":
        m.fuse(); m.model.to(device).eval()
        return m
    m.fuse()
    wrap_convs(m.model, 8, 8)
    m.model.to(device).eval()
    calibrate(m.model, calib, device=device)
    if mode == "naive":
        pass
    elif mode == "adaround":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
    elif mode == "qdrop":
        convert_to_adaround(m.model)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_strong,
                          qdrop_prob=qdrop_prob, verbose=False)
    elif mode == "brecq":
        convert_to_adaround(m.model)
        optimize_brecq(m.model, fp.model, calib, device, iters=recon_iters_strong, verbose=False)
    elif mode == "combined":
        convert_to_adaround(m.model, channelwise_smult=channelwise_smult)
        optimize_adaround(m.model, fp.model, calib, device, iters=recon_iters_ada, verbose=False)
        optimize_promptcal_scale_neighbor(m.model, fp.model, calib, device, pidx, iters=iters,
                                          lr=lr, k=k, neighbor_k=neighbor_k,
                                          neighbor_weight=neighbor_weight,
                                          asymmetric=True, scale_reg_weight=scale_reg_weight,
                                          exclude_from_neighbors=h_eval,
                                          cal_idx=cal_idx, cal_weight=cal_weight,
                                          identity_aware_margin=identity_aware_margin,
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
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--recon-iters-ada", type=int, default=1000)
    ap.add_argument("--recon-iters-strong", type=int, default=2000)
    ap.add_argument("--qdrop-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--neighbor-weight", type=float, default=1.0)
    ap.add_argument("--scale-reg-weight", type=float, default=10.0,
                    help="확정값(PROMPTCAL_CURRENT_MODEL.md §5.5/§8.2). s_mult(per-channel "
                         "벡터, adaround.py의 AdaRoundQuantConv2d.s_mult)의 (s-1)^2 정규화 강도.")
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
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--conditions", default="naive,adaround,qdrop,brecq,combined",
                    help="쉼표로 구분된 조건 목록(콤마 뒤 공백 없이). 09-16 추가 -- Combined 변형 "
                         "하나만 볼 때도 항상 5개 조건(특히 QDrop/BRECQ, 900~1400s대)을 다 "
                         "빌드하던 낭비를 줄이기 위함. 예: --conditions naive,combined")
    args = ap.parse_args()
    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    gt_ann = args.gt_ann or os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    print(f"[args] {vars(args)}")

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    _valid = {"naive", "adaround", "qdrop", "brecq", "combined"}
    assert all(c in _valid for c in conditions), f"--conditions에 알 수 없는 값: {set(conditions) - _valid}"
    models, calib_time = {}, {}
    for mode in conditions:
        print(f"[build] {mode}")
        t0 = time.perf_counter()
        models[mode] = build(YOLOWorld, args.model, coco, device, calib, mode, fp=fp,
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
                             identity_aware_margin=args.identity_aware_margin)
        calib_time[mode] = time.perf_counter() - t0
        print(f"  {mode} 빌드 {calib_time[mode]:.1f}s")
    # --conditions로 일부만 돌릴 때 "adaround"가 없을 수 있음 -- AdaRound 기반
    # 모드(adaround/qdrop/brecq/combined)는 전부 같은 conv/양자화 구조라 이론적
    # 모델 크기가 동일하므로, 돌아간 것 중 아무거나 골라도 된다(naive만 돈 경우는 naive로).
    _mib_mode = next((m for m in ["adaround", "qdrop", "brecq", "combined"] if m in models), conditions[0])
    model_mib = quantized_weight_mib(models[_mib_mode].model)

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
    print(f" 공식 데이터 설정(calib=train2017 {len(calib_paths)}장, LVIS=공식 minival) -- seed {args.seed}")
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


if __name__ == "__main__":
    main()
