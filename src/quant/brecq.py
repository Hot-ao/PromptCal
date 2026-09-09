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

핵심 예측(우리 논문): block 재구성으로 seen decision을 더 지킬 수는 있어도, held-out
decision은 여전히 못 지킨다 = reconstruction 계열 공통 한계.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .adaround import AdaRoundQuantConv2d, list_adaround_convs, h_alpha, free_cpu_mem


def _hpairs(qm, fm):
    """quant/fp 트리 나란히 순회 → (AdaRoundConv, fp Conv2d) 짝."""
    for (qn, qc), (fn, fc) in zip(qm.named_children(), fm.named_children()):
        if isinstance(qc, AdaRoundQuantConv2d):
            yield qc, fc
        else:
            yield from _hpairs(qc, fc)


def optimize_brecq(quant_module, fp_module, calib_tensors, device,
                   iters=2000, lr=1e-2, reg_weight=1e-3, verbose=True):
    """
    09-06 수정: 순차/누적 오차 반영. 이전 버전은 block의 입력과 출력(target)을
    둘 다 fp_module에서만 캡처해서, 앞선 block들의 실제 양자화 오차가 뒤쪽 block
    재구성에 전혀 반영되지 않았다(모든 block이 "앞은 전부 완벽한 FP"라고 가정).
    이번 버전은 입력을 quant_module 자체의 현재 상태(이미 처리된 앞쪽 block은
    hardened, 아직 처리 안 된 뒤쪽은 soft/init)로 다시 캡처해서 누적 오차를
    반영한다. target(출력) 기준은 그대로 FP다(asymmetric reconstruction,
    optimize_adaround와 동일 원리).
    """
    q_seq = quant_module.model      # DetectionModel.model = Sequential(blocks)
    fp_seq = fp_module.model
    q_blocks = list(q_seq)
    fp_blocks = list(fp_seq)
    n_blocks = len(q_blocks)
    head_idx = n_blocks - 1
    calib_list = list(calib_tensors)

    # 재구성 대상 구성: (label, q_module, fp_module, convs, is_block)
    targets = []
    for i, (qb, fb) in enumerate(zip(q_blocks, fp_blocks)):
        convs = list_adaround_convs(qb)
        if not convs:
            continue
        if i == head_idx:
            for qc, fc in _hpairs(qb, fb):     # head: conv 단위
                targets.append((f"head.conv", qc, fc, [qc], False))
        else:
            targets.append((f"block{i}", qb, fb, convs, True))   # block 단위 joint

    if verbose:
        nb = sum(1 for t in targets if t[4])
        nc = sum(1 for t in targets if not t[4])
        print(f"[brecq] 재구성 대상: block단위 {nb}개(joint), head conv단위 {nc}개 (누적 오차 반영)")

    for ti, (label, qm, fm, convs, is_block) in enumerate(targets):
        # (1) target(FP 출력) 캐시 -- 이상적 목표, 순수 FP forward에서만 캡처.
        out_buf = []

        def fp_hook(m, inp, out):
            if torch.is_tensor(out):
                out_buf.append(out.detach().half().cpu())
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
            saved = tuple(
                x.detach().half().cpu() if torch.is_tensor(x) else x
                for x in inp
            )
            in_buf.append(saved)
        hh_q = qm.register_forward_hook(q_hook)
        with torch.no_grad():
            for t in calib_list:
                quant_module(t.to(device))
        hh_q.remove()

        n = min(len(in_buf), len(out_buf))
        if n == 0:
            for c in convs:
                c.soft = False
            if verbose:
                print(f"  [{ti+1}/{len(targets)}] {label} 입력 미포착 스킵")
            continue

        for c in convs:
            c.soft = True
            c.ste = True                       # block 내부로 grad 흐르게
        alphas = [c.alpha for c in convs]
        opt = torch.optim.Adam(alphas, lr=lr)
        for it in range(iters):
            j = it % n
            # 입력 복원: 텐서는 device로, 비텐서(있으면)는 그대로
            args = tuple(
                x.to(device).float() if torch.is_tensor(x) else x
                for x in in_buf[j]
            )
            tgt = out_buf[j].to(device).float()
            opt.zero_grad()
            out = qm(*args)                    # block/conv를 quant weight로 통과 (다중 입력 지원)
            beta = max(2.0, 20.0 * (1 - it / iters))
            reg = sum(c.reg_loss(beta) for c in convs)
            loss = (out - tgt).pow(2).mean() + reg_weight * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(alphas, max_norm=1.0)
            opt.step()

        for c in convs:
            c.soft = False
            c.ste = False
        if verbose:
            hconv = sum(float(((h_alpha(c.alpha) < 0.05) |
                               (h_alpha(c.alpha) > 0.95)).float().mean())
                        for c in convs) / len(convs) * 100
            print(f"  [{ti+1}/{len(targets)}] {label} ({len(convs)} conv) "
                  f"done, h→0/1 {hconv:.0f}%")
        del in_buf, out_buf
        free_cpu_mem()

    if verbose:
        print("[brecq] 전체 재구성 완료 (hard 반올림 모드)")
