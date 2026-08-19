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

from .adaround import AdaRoundQuantConv2d, list_adaround_convs, h_alpha


def _hpairs(qm, fm):
    """quant/fp 트리 나란히 순회 → (AdaRoundConv, fp Conv2d) 짝."""
    for (qn, qc), (fn, fc) in zip(qm.named_children(), fm.named_children()):
        if isinstance(qc, AdaRoundQuantConv2d):
            yield qc, fc
        else:
            yield from _hpairs(qc, fc)


def optimize_brecq(quant_module, fp_module, calib_tensors, device,
                   iters=2000, lr=1e-2, reg_weight=1e-3, verbose=True):
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
        print(f"[brecq] 재구성 대상: block단위 {nb}개(joint), head conv단위 {nc}개")

    for ti, (label, qm, fm, convs, is_block) in enumerate(targets):
        # FP (입력들, 출력) 캐시. C2fAttn 등은 입력이 2개(vision + text guide)이므로
        # 첫 입력만이 아니라 입력 tuple 전체를 저장. 텍스트 guide는 양자화 안 하므로 FP값 유지.
        in_buf, out_buf = [], []

        def hook(m, inp, out):
            if torch.is_tensor(out):
                saved = tuple(
                    x.detach().half().cpu() if torch.is_tensor(x) else x
                    for x in inp
                )
                in_buf.append(saved)
                out_buf.append(out.detach().half().cpu())
        hh = fm.register_forward_hook(hook)
        with torch.no_grad():
            for t in calib_list:
                fp_module(t.to(device))
        hh.remove()

        if not in_buf:
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
            j = it % len(in_buf)
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

    if verbose:
        print("[brecq] 전체 재구성 완료 (hard 반올림 모드)")
