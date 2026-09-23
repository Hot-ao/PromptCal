"""
BRECQ (D-2): block-wise reconstruction.

AdaRound가 conv 하나씩 출력을 맞추는 것(layer-wise)과 달리, BRECQ는 block(YOLOv8의 C2f 등
아키텍처 단위)을 통째로 재구성한다. block의 FP 입력을 흘려 block 출력을 FP와 맞추도록
block 안의 모든 반올림(alpha)을 동시에 최적화 → layer 간 상관을 반영.

우리 구현:
  - 재구성 단위 = DetectionModel.model(Sequential)의 top-level 블록.
  - 비head 블록(backbone/neck): block 단위 joint 재구성(BRECQ). block 내부로 grad가 흐르도록
    activation STE 사용.
  - head(WorldDetect): 출력이 복잡(decode)하므로 내부 conv(cv2/cv3)를 conv 단위로 재구성
    (=AdaRound). cv4(ContrastiveHead)는 애초에 양자화 안 함.

qdrop_prob > 0이면 이 함수가 곧 QDrop(Wei et al., ICLR 2022)이다 -- QDrop 원 논문은
BRECQ의 block-wise 재구성 위에 activation 양자화 drop을 얹은 것이기 때문이다.

핵심 예측(우리 논문): block 재구성으로 seen decision을 더 지킬 수는 있어도, held-out
decision은 여전히 못 지킨다 = reconstruction 계열 공통 한계.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .adaround import (AdaRoundQuantConv2d, list_adaround_convs, h_alpha, free_cpu_mem,
                       lp_rec_loss, temp_decay,
                       DEFAULT_LR, DEFAULT_REG_WEIGHT, DEFAULT_WARMUP,
                       DEFAULT_ACT_LR_BRECQ, DEFAULT_ACT_LR_QDROP)


def _backbone_end_idx(blocks):
    """SPPF(YOLOv8 backbone 마지막 블록)의 인덱스를 backbone/neck 경계로 쓴다.
    QDrop 공식 COCO 프로토콜(부록 E): backbone은 block-wise, neck(FPN)은 layer-wise
    재구성. 못 찾으면 None(경계 없음 = 전부 block-wise, 이전 동작과 동일)."""
    for i, b in enumerate(blocks):
        if type(b).__name__ == "SPPF":
            return i
    return None


def _hpairs(qm, fm):
    """quant/fp 트리 나란히 순회 → (AdaRoundConv, fp Conv2d) 짝."""
    for (qn, qc), (fn, fc) in zip(qm.named_children(), fm.named_children()):
        if isinstance(qc, AdaRoundQuantConv2d):
            yield qc, fc
        else:
            yield from _hpairs(qc, fc)


def _to_cpu(x):
    """block 입력을 fp16 CPU로 내림. Concat/Detect류처럼 입력이 list/tuple로
    중첩된 경우까지 재귀 처리 -- 예전엔 비텐서를 `else x`로 그대로 뒀는데,
    그러면 list 안의 CUDA 텐서가 calib 이미지 수만큼 GPU에 그대로 살아남아
    (fp32) OOM이 난다."""
    if torch.is_tensor(x):
        return x.detach().half().cpu()
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu(v) for v in x)
    return x


def _to_dev(x, device):
    """_to_cpu의 역 -- 텐서만 device/float으로 복원, 구조는 보존."""
    if torch.is_tensor(x):
        return x.to(device).float()
    if isinstance(x, list):
        return [_to_dev(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_dev(v, device) for v in x)
    return x


def _collate(items):
    """batch=1 항목들을 배치로 합친다(텐서는 dim0 cat, list/tuple은 재귀, 그 외는 첫 항목)."""
    first = items[0]
    if torch.is_tensor(first):
        return torch.cat(items, dim=0)
    if isinstance(first, list):
        return [_collate([it[k] for it in items]) for k in range(len(first))]
    if isinstance(first, tuple):
        return tuple(_collate([it[k] for it in items]) for k in range(len(first)))
    return first


# 09-23: calibration forward 배치화를 시도했으나(CAPTURE_BATCH+_split1), batch=1 대비
# batch=32 forward가 cuDNN 알고리즘 선택 차이로 레이어를 지날수록 활성화값이 눈에 띄게
# 갈라짐을 확인(직접 버퍼 대조, 레이어 25 근처 최대 절대오차 0.51 -- 신호 크기와 맞먹음)
# -- 재구성 목표값 자체가 바뀌어 최종 지표가 유의미하게 달라졌다. 리스크가 커서 되돌림
# (이미지 1장씩 순차 forward, 기존 동작). 편향 없이 노이즈 수준인지 검증 전엔 재도입 안 함.


def _mix(q, f, prob):
    """QDrop의 input_prob: block 입력을 원소별 확률 prob로만 양자화 경로 값(q)으로,
    나머지는 FP 값(f)으로 쓴다. 구조(list/tuple)는 그대로 유지."""
    if torch.is_tensor(q) and torch.is_tensor(f) and q.shape == f.shape:
        return torch.where(torch.rand_like(q) < prob, q, f)
    if isinstance(q, list) and isinstance(f, list):
        return [_mix(a, b, prob) for a, b in zip(q, f)]
    if isinstance(q, tuple) and isinstance(f, tuple):
        return tuple(_mix(a, b, prob) for a, b in zip(q, f))
    return q


def _brecq_act_stage(targets, quant_module, fp_module, calib_list, device,
                     iters, act_lr, p, verbose, batch=1):
    """공식 BRECQ 2단계: weight는 hard 고정, activation scale(LSQActQuant.delta, 09-21부터
    Combined의 s_mult와 독립)만 block/conv 단위로 순차 재구성. 입력은 앞 단위의 학습이
    반영된 quant_module 현재 상태에서 캡처."""
    for ti, (label, qm, fm, convs, is_block) in enumerate(targets):
        out_buf, in_buf = [], []

        def fp_hook(m, inp, out):
            if torch.is_tensor(out):
                out_buf.append(out.detach().half().cpu())
        hh = fm.register_forward_hook(fp_hook)
        with torch.no_grad():
            for t in calib_list:
                fp_module(t.to(device))
        hh.remove()

        def q_hook(m, inp, out):
            in_buf.append(_to_cpu(inp))
        hh = qm.register_forward_hook(q_hook)
        with torch.no_grad():
            for t in calib_list:
                quant_module(t.to(device))
        hh.remove()

        n = min(len(in_buf), len(out_buf))
        if n == 0:
            continue
        for c in convs:
            c.soft = False
            c.use_lsq = True
        opt_a = torch.optim.Adam([c.lsq_act.delta for c in convs], lr=act_lr)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=iters, eta_min=0.0)
        for it in range(iters):
            js = torch.randint(0, n, (batch,)).tolist()
            args = _to_dev(_collate([in_buf[j] for j in js]), device)
            tgt = torch.cat([out_buf[j] for j in js], dim=0).to(device).float()
            opt_a.zero_grad()
            loss = lp_rec_loss(qm(*args), tgt, p=p)
            loss.backward()
            opt_a.step()
            sched_a.step()
        if verbose:
            print(f"  [act-stage {ti+1}/{len(targets)}] {label} done")
        del in_buf, out_buf
        free_cpu_mem()


def optimize_brecq(quant_module, fp_module, calib_tensors, device,
                   iters=2000, lr=DEFAULT_LR, reg_weight=DEFAULT_REG_WEIGHT, verbose=True,
                   learn_act_scale=False, act_lr=None, warmup=DEFAULT_WARMUP,
                   qdrop_prob=0.0, grad_clip=None,
                   two_stage=False, act_iters=5000, act_p=2.4, batch=1,
                   neck_layerwise=True):
    """
    09-06 수정: 순차/누적 오차 반영. 이전 버전은 block의 입력과 출력(target)을
    둘 다 fp_module에서만 캡처해서, 앞선 block들의 실제 양자화 오차가 뒤쪽 block
    재구성에 전혀 반영되지 않았다(모든 block이 "앞은 전부 완벽한 FP"라고 가정).
    이번 버전은 입력을 quant_module 자체의 현재 상태(이미 처리된 앞쪽 block은
    hardened, 아직 처리 안 된 뒤쪽은 soft/init)로 다시 캡처해서 누적 오차를
    반영한다. target(출력) 기준은 그대로 FP다(asymmetric reconstruction,
    optimize_adaround와 동일 원리).

    09-18 공식 구현 정합성 수정(adaround.py와 동일 취지):
      - 재구성 손실을 lp_rec_loss(BRECQ lp_loss p=2 정규화)로 교체. 기존
        .pow(2).mean()은 공식 대비 C_out배 작아서 rounding 정규화가 압도했다.
      - lr 1e-2 → 1e-3, reg_weight 1e-3 → 0.01(공식 기본값), warmup 20% 도입.
      - 표본을 it % n(결정적 순환)에서 무작위 추출로 교체.
      - clip_grad_norm_(1.0) 기본 해제. 공식 구현엔 없는 처리인데, reg가 수백만
        원소 sum이라 global norm이 쉽게 1을 넘어 재구성 gradient까지 같이
        축소시킨다. 필요하면 grad_clip=1.0으로 되살릴 수 있게 인자로 남김.
      - learn_act_scale의 s_mult를 alpha와 분리된 Adam(act_lr)+CosineAnnealingLR로.

    qdrop_prob (09-18): QDrop(Wei et al., ICLR 2022). 원 논문은 BRECQ의 block-wise
    재구성 위에서 동작하므로 여기 구현하는 것이 맞다(이전엔 optimize_adaround의
    layer-wise 경로에 붙어 있었다 = QDrop의 핵심 설정과 다름). 공식 구현과 동일하게
    두 곳에 drop을 건다:
      (a) block 내부 모든 activation quantizer -- AdaRoundQuantConv2d.qdrop_prob
          (공식 UniformAffineQuantizer.prob과 동일, "양자화값을 유지할 확률").
      (b) block 입력 -- 양자화 경로 입력과 FP 입력을 원소별로 섞음
          (공식 block_recon.py의 input_prob).
    최적화가 끝나면 전부 0으로 되돌려 추론에는 drop이 남지 않는다.

    neck_layerwise (09-21): QDrop 공식 COCO 프로토콜(논문 부록 E, "we didn't
    quantize the head but applied block reconstruction to backbone and layer
    reconstruction to neck like BRECQ")과 동일하게, backbone(SPPF까지)은
    block-wise joint, neck(그 이후, head 제외)은 head와 같은 conv 단위(layer-wise)로
    재구성한다. False면 이전 동작(전부 block-wise) 유지.

    learn_act_scale (09-17, adaround.py의 optimize_adaround와 동일 취지):
    원 BRECQ 논문은 alpha(rounding)와 activation step size(LSQ)를 block
    reconstruction loss 하나로 공동 최적화한다. True면 AdaRoundQuantConv2d의
    lsq_act(LSQActQuant, 절대 delta 파라미터)를 켜서 alpha와 같이 최적화 -- STE
    처리는 forward()의 lsq_act 경로가 대신하므로 ste=True를 따로 켤 필요 없음
    (use_lsq가 우선). 09-21: 이전엔 Combined의 s_mult(배율 파라미터화)를 그대로
    재사용했는데(코드 공유 -- baseline이 제안 방법의 메커니즘을 쓰는 것처럼 읽힐
    위험 지적됨), 이제 완전히 독립된 LSQActQuant + 공식 lr(act_lr, 위 참고)을
    쓴다 -- s_mult/Combined와 파라미터도 코드 경로도 공유하지 않는다. 기본 False
    (기존 동작과 완전히 동일).

    two_stage (09-21): 공식 BRECQ(main_imagenet.py)는 순차 2단계다 -- (1) activation
    양자화를 끈 채(act_quant=False) alpha만 재구성, (2) weight를 hard로 고정하고
    activation 양자화를 켠 뒤 activation step(LSQ)만 iters_a=5000, cosine, L_p(p=2.4)
    손실로 별도 학습. True면 이 절차를 따른다(learn_act_scale은 2단계를 켜는 스위치로
    쓰이며 alpha 루프에서는 꺼진다). QDrop(qdrop_prob>0)은 공식이 공동 최적화라 해당 없음.
    """
    if act_lr is None:
        # 09-21: baseline LSQ가 이제 절대 delta 파라미터(LSQActQuant)라 공식 lr을 그대로
        # 쓴다 -- BRECQ 공식(main_imagenet.py --lr) 4e-4, QDrop 공식(논문 부록 E) 4e-5.
        act_lr = DEFAULT_ACT_LR_QDROP if qdrop_prob > 0 else DEFAULT_ACT_LR_BRECQ
    if two_stage:
        assert qdrop_prob == 0.0, "two_stage는 BRECQ 전용(QDrop 공식은 alpha와 LSQ를 공동 최적화)"
    stage2 = two_stage and learn_act_scale
    if two_stage:
        learn_act_scale = False
        for c in list_adaround_convs(quant_module):
            c.act_quant_enabled = False
    q_seq = quant_module.model      # DetectionModel.model = Sequential(blocks)
    fp_seq = fp_module.model
    q_blocks = list(q_seq)
    fp_blocks = list(fp_seq)
    n_blocks = len(q_blocks)
    head_idx = n_blocks - 1
    calib_list = list(calib_tensors)

    backbone_end = _backbone_end_idx(fp_blocks) if neck_layerwise else None

    # 재구성 대상 구성: (label, q_module, fp_module, convs, is_block)
    targets = []
    for i, (qb, fb) in enumerate(zip(q_blocks, fp_blocks)):
        convs = list_adaround_convs(qb)
        if not convs:
            continue                                            # skip_head 등으로 미양자화면 자동 제외
        is_neck = backbone_end is not None and i > backbone_end and i != head_idx
        if i == head_idx or is_neck:
            tag = "head" if i == head_idx else f"neck{i}"
            for hi, (qc, fc) in enumerate(_hpairs(qb, fb)):     # head/neck: conv 단위(layer-wise)
                targets.append((f"{tag}.conv{hi}", qc, fc, [qc], False))
        else:
            targets.append((f"block{i}", qb, fb, convs, True))   # backbone: block 단위 joint

    if verbose:
        nb = sum(1 for t in targets if t[4])
        nc = sum(1 for t in targets if not t[4])
        tag = f", QDrop prob={qdrop_prob}" if qdrop_prob > 0 else ""
        print(f"[brecq] 재구성 대상: block단위 {nb}개(joint), head conv단위 {nc}개 (누적 오차 반영{tag})")

    n_skipped = 0
    flips = []

    for ti, (label, qm, fm, convs, is_block) in enumerate(targets):
        # (1) target(FP 출력) 캐시 -- 이상적 목표, 순수 FP forward에서만 캡처.
        #     QDrop의 input_prob을 쓸 땐 같은 pass에서 FP 입력도 같이 캐시(추가 forward 불필요).
        out_buf = []
        fp_in_buf = [] if qdrop_prob > 0 else None

        def fp_hook(m, inp, out):
            if torch.is_tensor(out):
                out_buf.append(out.detach().half().cpu())
                if fp_in_buf is not None:
                    fp_in_buf.append(_to_cpu(inp))
        hh_fp = fm.register_forward_hook(fp_hook)
        with torch.no_grad():
            for t in calib_list:
                fp_module(t.to(device))
        hh_fp.remove()

        # (2) 이 block/conv로 실제 흘러들어오는 입력 -- quant_module 현재 상태로
        #     forward해서 캡처(누적 오차 반영). C2fAttn 등 입력이 여러 개인
        #     경우 tuple 전체를 저장(텍스트 guide 등 양자화 안 하는 입력은 그대로).
        in_buf = []

        def q_hook(m, inp, out):
            in_buf.append(_to_cpu(inp))
        hh_q = qm.register_forward_hook(q_hook)
        with torch.no_grad():
            for t in calib_list:
                quant_module(t.to(device))
        hh_q.remove()

        n = min(len(in_buf), len(out_buf))
        if n == 0:
            for c in convs:
                c.soft = False
            n_skipped += 1
            if verbose:
                print(f"  [{ti+1}/{len(targets)}] {label} 입력/출력 미포착 스킵 (hard 유지)")
            del in_buf, out_buf, fp_in_buf
            free_cpu_mem()
            continue

        for c in convs:
            c.soft = True
            c.ste = True                       # block 내부로 grad 흐르게
            c.qdrop_prob = qdrop_prob          # QDrop (a): block 내부 quantizer drop
            if learn_act_scale:
                c.use_lsq = True               # forward()가 이 분기를 ste보다 우선함
        params = [c.alpha for c in convs]
        opt = torch.optim.Adam(params, lr=lr)
        opt_a = sched_a = None
        if learn_act_scale:
            opt_a = torch.optim.Adam([c.lsq_act.delta for c in convs], lr=act_lr)
            sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=iters, eta_min=0.0)
        for it in range(iters):
            js = torch.randint(0, n, (batch,)).tolist()
            # 입력 복원: 텐서는 device로, 비텐서(있으면)는 그대로
            args = _to_dev(_collate([in_buf[j] for j in js]), device)
            if qdrop_prob > 0:                 # QDrop (b): block 입력 drop(input_prob)
                args = _mix(args, _to_dev(_collate([fp_in_buf[j] for j in js]), device), qdrop_prob)
            tgt = torch.cat([out_buf[j] for j in js], dim=0).to(device).float()
            opt.zero_grad()
            if opt_a is not None:
                opt_a.zero_grad()
            out = qm(*args)                    # block/conv를 quant weight로 통과 (다중 입력 지원)
            loss = lp_rec_loss(out, tgt)
            if it >= warmup * iters:           # 공식 warmup: 앞 구간은 round_loss=0
                beta = temp_decay(it, iters, warmup)
                loss = loss + reg_weight * sum(c.reg_loss(beta) for c in convs)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            if opt_a is not None:
                opt_a.step()
                sched_a.step()

        for c in convs:
            c.soft = False
            c.ste = False
            c.qdrop_prob = 0.0                 # 추론에는 drop이 남으면 안 됨
        fl = sum(c.flip_rate() for c in convs) / len(convs)
        flips.append(fl)
        if verbose:
            hconv = sum(float(((h_alpha(c.alpha) < 0.05) |
                               (h_alpha(c.alpha) > 0.95)).float().mean())
                        for c in convs) / len(convs) * 100
            print(f"  [{ti+1}/{len(targets)}] {label} ({len(convs)} conv) "
                  f"done, h→0/1 {hconv:.0f}%, nearest 대비 flip {fl:.3f}%")
        del in_buf, out_buf, fp_in_buf
        free_cpu_mem()

    if two_stage:
        for c in list_adaround_convs(quant_module):
            c.act_quant_enabled = True
    if stage2:
        _brecq_act_stage(targets, quant_module, fp_module, calib_list, device,
                         act_iters, act_lr, act_p, verbose, batch)

    if n_skipped:
        # verbose=False(run_comparison 기본)여도 조용히 넘어가면 안 되는 정보.
        print(f"[brecq][warn] {n_skipped}/{len(targets)} 대상이 입력/출력 미포착으로 "
              f"학습 없이 hard 확정됨(블록 출력이 텐서가 아닌 경우 등)")
    if flips:
        mean_flip = sum(flips) / len(flips)
        name = "qdrop" if qdrop_prob > 0 else "brecq"
        print(f"[{name}] 전체 재구성 완료 (hard 반올림). nearest 대비 평균 flip "
              f"{mean_flip:.3f}% (max {max(flips):.3f}%)")
