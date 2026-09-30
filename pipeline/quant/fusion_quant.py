"""
텍스트 융합 지점 전용 양자화 (09-30, 설계 방향 1·2의 프로토타입).

방향 1 — 게이트 교환(gate commutation), C2fAttn 전용
  C2fAttn:  out = act( cv2( cat(y0, y1, y2, a) ) ),   a_h = p_h * aw_h   (p = attn.proj_conv(y2), aw = 텍스트 게이트)
  cv2가 1×1이고 aw는 위치별(head별) 스칼라라서 FP로는
      out = act( W_main · cat(y0, y1, y2) + b  +  Σ_h aw_h ⊙ (W_h · p_h) )
  와 같다. 텍스트 분기를 전용 1×1 conv(head마다)로 분리하고 게이트를 conv **뒤**에 곱한다.
  -> 텍스트 분기가 다른 분기와 weight/activation scale을 공유하지 않고(runs/139: 공유 scale 오염이 12.cv2 손상의 원인),
     양자화되는 입력이 게이트 전의 p가 된다. 배포: 표준 1×1 conv + 위치별 곱 + 덧셈, 추가 비트 없음.
  `gate_commute(c2fattn)`은 FP 수준의 구조 변환이다(nn.Conv2d로 분리) -> 이후 wrap_convs가 새 conv를 감싼다.

방향 2 — 분기 인식 concat 양자화(branch-aware)
  concat을 입력으로 받는 conv의 입력 채널을 생산자(branch)별 구간으로 나누고, 구간마다 activation scale과
  (출력 채널별) weight scale을 따로 둔다. 배포: 생산 conv들은 원래 각자 scale로 requant하므로 concat 뒤 공통
  scale로 맞추지 않고, 소비 conv를 구간별 conv의 합으로 계산하면 된다(추가 비트 없음).
  `apply_branch_quant(qconv, groups)` — QuantConv2d 하나에 적용. `branch_groups_from_trace(...)` — 채널 계보
  추적(channel_graph.trace_channel_producers)에서 생산자 집합이 바뀌는 지점으로 구간을 자동 결정.

현재는 naive(학습 없는) 경로 전용 프로토타입이다. BRECQ(AdaRoundQuantConv2d) 연동은 효과 확인 후.
"""
from __future__ import annotations
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import ActObserver, QuantConv2d, mse_weight_scale_asym_channelwise


# ----------------------------------------------------------------------------- 방향 1
def _split_conv(conv: nn.Conv2d, cols: slice, bias: bool) -> nn.Conv2d:
    w = conv.weight.detach()[:, cols].clone()
    c = nn.Conv2d(w.shape[1], w.shape[0], 1, bias=bias).to(w.device, w.dtype)
    c.weight.data.copy_(w)
    if bias and conv.bias is not None:
        c.bias.data.copy_(conv.bias.detach())
    elif bias:
        c.bias.data.zero_()
    return c


def gate_commute(block: nn.Module) -> nn.Module:
    """C2fAttn 인스턴스를 제자리에서 게이트 교환 형태로 바꾼다(fuse() 뒤, wrap_convs 전에 호출).
    cv2.conv(1×1)를 main(앞 2+n 분기)과 head별 side conv로 쪼개고 forward를 교체한다."""
    assert type(block).__name__ == "C2fAttn", type(block).__name__
    cv2 = block.cv2
    conv = cv2.conv
    assert conv.kernel_size == (1, 1) and conv.groups == 1
    attn = block.attn
    c, nh = block.c, attn.nh
    n_main = conv.in_channels - c                        # y0, y1, (m 출력들)
    hc_in = c // nh                                      # head당 proj 출력 채널(= view(bs, nh, -1)의 -1)
    block.cv2_main = _split_conv(conv, slice(0, n_main), bias=True)
    block.cv2_side = nn.ModuleList(
        _split_conv(conv, slice(n_main + h * hc_in, n_main + (h + 1) * hc_in), bias=False) for h in range(nh))
    block._gc_act = cv2.act
    block._gc_hc_in = hc_in
    del block.cv2                                        # 원래 conv는 더 이상 쓰지 않음(wrap_convs 대상에서 제외)

    def forward(self, x, guide):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        a = self.attn
        z = y[-1]
        bs, _, h, w = z.shape
        g = a.gl(guide).view(bs, guide.shape[1], a.nh, a.hc)
        emb = (a.ec(z) if a.ec is not None else z).view(bs, a.nh, a.hc, h, w)
        aw = torch.einsum("bmchw,bnmc->bmhwn", emb, g).max(dim=-1)[0]
        aw = (aw / (a.hc ** 0.5) + a.bias[None, :, None, None]).sigmoid() * a.scale   # [bs, nh, h, w]
        p = a.proj_conv(z)                                                           # 게이트 전
        out = self.cv2_main(torch.cat(y, 1))
        for hh, sc in enumerate(self.cv2_side):
            ph = p[:, hh * self._gc_hc_in:(hh + 1) * self._gc_hc_in]
            out = out + aw[:, hh:hh + 1] * sc(ph)
        return self._gc_act(out)

    block.forward = types.MethodType(forward, block)
    return block


def gate_commute_all(det_model: nn.Module) -> list:
    done = []
    for i, m in enumerate(det_model.model):
        if type(m).__name__ == "C2fAttn":
            gate_commute(m)
            done.append(i)
    return done


# ----------------------------------------------------------------------------- 방향 2
class GroupActObserver(nn.Module):
    """입력 채널 구간마다 독립 ActObserver. ActObserver와 같은 인터페이스(calibrate()가 그대로 부른다)."""
    def __init__(self, bits: int, groups, method: str = "mse"):
        super().__init__()
        self.groups = [tuple(g) for g in groups]
        self.obs = nn.ModuleList(ActObserver(bits, method) for _ in self.groups)

    @property
    def method(self):
        return self.obs[0].method

    @method.setter
    def method(self, v):
        for o in self.obs:
            o.method = v

    @property
    def bits(self):
        return self.obs[0].bits

    @bits.setter
    def bits(self, v):
        for o in self.obs:
            o.bits = v

    @property
    def ready(self):
        return all(o.ready for o in self.obs)

    @torch.no_grad()
    def observe(self, x):
        for (s, e), o in zip(self.groups, self.obs):
            o.observe(x[:, s:e])

    @torch.no_grad()
    def freeze(self, range_blend: float = 0.0):
        for o in self.obs:
            o.freeze(range_blend=range_blend)

    @torch.no_grad()
    def quantize(self, x):
        return torch.cat([o.quantize(x[:, s:e]) for (s, e), o in zip(self.groups, self.obs)], 1)


def apply_branch_quant(qc: QuantConv2d, groups, act: bool = True, weight: bool = True):
    """calibrate() 전에 호출. groups: [(start, end), ...]가 입력 채널 전체를 덮어야 한다."""
    groups = [tuple(g) for g in groups]
    assert groups[0][0] == 0 and groups[-1][1] == qc.conv.in_channels
    assert all(a[1] == b[0] for a, b in zip(groups, groups[1:]))
    assert not qc._w_quant_ready and not qc.a_obs.ready, "calibrate() 전에 적용해야 함"
    if act:
        qc.a_obs = GroupActObserver(qc.a_obs.bits, groups, qc.a_obs.method).to(qc.conv.weight.device)
    if weight:
        def freeze_weight_quant(self):
            w = self.conv.weight.detach()
            sc = torch.empty(w.shape[0], w.shape[1], 1, 1, device=w.device, dtype=w.dtype)
            zp = torch.empty_like(sc)
            for s, e in groups:
                d, z = mse_weight_scale_asym_channelwise(w[:, s:e].contiguous(), self.w_bits)
                sc[:, s:e], zp[:, s:e] = d, z
            self.w_scale, self.w_zero_point = sc, zp
            self.w_qmax = torch.full((1, w.shape[1], 1, 1), 2 ** self.w_bits - 1, device=w.device, dtype=w.dtype)
            self._w_quant_ready = True
        qc.freeze_weight_quant = types.MethodType(freeze_weight_quant, qc)
    qc._branch_groups = groups
    return qc


@torch.no_grad()
def branch_groups_from_trace(deps, n_channels):
    """channel_graph.trace_channel_producers의 deps[ci](입력 채널별 생산 채널 집합)로부터, 생산자 집합이
    바뀌는 지점에서 끊은 연속 구간 목록을 만든다. 계보가 없는 채널은 앞 구간에 붙인다."""
    keys = [frozenset(p for p, _ in s) for s in deps]
    groups, start = [], 0
    for c in range(1, n_channels):
        if keys[c] and keys[c] != keys[c - 1] and keys[c - 1]:
            groups.append((start, c)); start = c
    groups.append((start, n_channels))
    return groups
