"""
attention/contrastive 연산 양자화 (10-01 보강 실험: "attention·DFL을 FP로 둬도 되는가").

기본 프로토콜은 conv만 양자화하고 텍스트 융합 attention(Linear/matmul)과 head의 contrastive matmul은 FP다.
이 모듈은 그 연산들을 8bit로 양자화해 "FP로 둔 것이 결론(GPM − BRECQ)에 영향을 주는가"를 잰다(QATMA 조건과 정합).

양자화 규칙(배포 관점):
  - 실행 중 값(이미지에서 온 피연산자, Linear 입력, softmax/sigmoid 출력)
      -> per-tensor 비대칭 8bit ActObserver(calibrate()가 conv와 같은 방식으로 관측·freeze).
  - 텍스트에서만 오는 상수 피연산자(어휘가 고정된 배포에서 미리 계산해 두는 값: C2fAttn guide projection,
    ImagePoolingAttn query, contrastive head의 정규화된 텍스트)
      -> weight처럼 행(클래스)별 비대칭 MSE 8bit(데이터 불필요, 입력이 바뀔 때만 다시 계산해 캐시).
  - 실행 중 입력을 받는 Linear(ImagePoolingAttn key/value/proj, ImagePoolingAttn 뒤의 C2fAttn guide Linear)
      -> weight 출력 채널별 MSE 8bit + 입력 A8.
  - LayerNorm, softmax, sigmoid, max, residual add, logit scale/bias는 FP(원소별 연산, 통상 LUT/FP).
scope: "attn"     = C2fAttn 게이트 경로 + ImagePoolingAttn
       "attn_cls" = 위 + head contrastive matmul(영역 임베딩 × 텍스트)
v1(yolov8s-world)에서 ImagePoolingAttn(16) 뒤의 C2fAttn(19, 22)은 이미지로 갱신된 텍스트를 받으므로 상수가 아니다.
fuse() 뒤, gate_commute_all 뒤, calibrate() 전에 호출한다.
"""
from __future__ import annotations
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import ActObserver, mse_weight_scale_asym_channelwise


class QPoint(nn.Module):
    """실행 중 값 양자화 지점. QuantConv2d와 같은 calibrating/quantized/a_obs 인터페이스."""
    _is_qpoint = True

    def __init__(self, bits: int = 8):
        super().__init__()
        self.a_obs = ActObserver(bits)
        self.calibrating = False
        self.quantized = False

    def forward(self, x):
        if self.calibrating:
            self.a_obs.observe(x)
            return x
        if self.quantized and self.a_obs.ready:
            return self.a_obs.quantize_ste(x)          # STE: BRECQ 재구성 중 gradient가 앞 블록 conv로 흐르게
        return x


@torch.no_grad()
def _rowwise_q(t: torch.Tensor, bits: int = 8) -> torch.Tensor:
    shp = t.shape
    w = t.reshape(-1, shp[-1]).float()
    d, z = mse_weight_scale_asym_channelwise(w, bits)
    n = 2 ** bits - 1
    return ((torch.clamp(torch.round(w / d) + z, 0, n) - z) * d).reshape(shp).to(t.dtype)


class ConstQ(nn.Module):
    """텍스트 상수 피연산자 양자화(행별 MSE, 캐시). quantized일 때만 적용(calibrate 중엔 통과)."""
    _is_qpoint = True

    def __init__(self, bits: int = 8):
        super().__init__()
        self.bits = bits
        self.calibrating = False
        self.quantized = False
        self.a_obs = None
        self._key = None
        self._val = None

    def forward(self, t):
        if not self.quantized:
            return t
        if self._key is None or self._key.shape != t.shape or not torch.equal(self._key, t):
            self._key = t.detach().clone()
            self._val = _rowwise_q(self._key, self.bits)
        return self._val


class QuantLinear(nn.Module):
    """실행 중 입력을 받는 Linear: weight 출력 채널별 MSE 8bit + 입력 A8."""
    _is_qpoint = True

    def __init__(self, lin: nn.Linear, bits: int = 8):
        super().__init__()
        self.lin = lin
        self.bits = bits
        self.a_obs = ActObserver(bits)
        self.calibrating = False
        self.quantized = False
        self._wq = None

    def forward(self, x):
        if self.calibrating:
            self.a_obs.observe(x)
            return self.lin(x)
        if not self.quantized:
            return self.lin(x)
        if self._wq is None:
            self._wq = _rowwise_q(self.lin.weight.detach(), self.bits)
        xq = self.a_obs.quantize_ste(x) if self.a_obs.ready else x
        return F.linear(xq, self._wq, self.lin.bias)


# ----------------------------------------------------------------------------- 게이트(C2fAttn 공용)
def attn_gate(a: nn.Module, z: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """MaxSigmoidAttnBlock의 게이트 aw[bs, nh, h, w]. 양자화 지점(a._aq)이 있으면 적용."""
    q = getattr(a, "_aq", None)
    bs, _, h, w = z.shape
    if q is None:
        g = a.gl(guide)
    elif q["const"]:
        g = q["g"](a.gl(guide))                       # 상수: projection 결과를 weight처럼 양자화
    else:
        g = q["g"](q["gl"](guide))                    # 실행 중: QuantLinear 뒤 출력도 matmul 피연산자로 A8
    g = g.view(bs, guide.shape[1], a.nh, a.hc)
    emb = a.ec(z) if a.ec is not None else z
    if q is not None:
        emb = q["emb"](emb)
    emb = emb.view(bs, a.nh, a.hc, h, w)
    aw = torch.einsum("bmchw,bnmc->bmhwn", emb, g).max(dim=-1)[0]
    aw = (aw / (a.hc ** 0.5) + a.bias[None, :, None, None]).sigmoid() * a.scale
    if q is not None:
        aw = q["aw"](aw)
    return aw


def _msab_forward(self, x, guide):
    bs, _, h, w = x.shape
    aw = attn_gate(self, x, guide)
    x = self.proj_conv(x).view(bs, self.nh, -1, h, w)
    return (x * aw.unsqueeze(2)).view(bs, -1, h, w)


def _ipa_forward(self, x, text):
    q_ = self._aq
    bs = x[0].shape[0]
    np_ = self.k ** 2
    x = [pool(proj(x)).view(bs, -1, np_) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
    x = torch.cat(x, dim=-1).transpose(1, 2)
    q = q_["q"](self.query(text))
    k = q_["k"](self.key(x))
    v = q_["v"](self.value(x))
    q = q.reshape(bs, -1, self.nh, self.hc)
    k = k.reshape(bs, -1, self.nh, self.hc)
    v = v.reshape(bs, -1, self.nh, self.hc)
    aw = F.softmax(torch.einsum("bnmc,bkmc->bmnk", q, k) / (self.hc ** 0.5), dim=-1)
    aw = q_["aw"](aw)
    x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
    x = self.proj(x.reshape(bs, -1, self.ec))
    return x * self.scale + text


def _cls_forward(self, x, w):
    if hasattr(self, "norm"):                         # BNContrastiveHead
        x = self.norm(x)
    else:                                             # ContrastiveHead
        x = F.normalize(x, dim=1, p=2)
    w = F.normalize(w, dim=-1, p=2)
    x = self._aq["x"](x)
    w = self._aq["w"](w)
    x = torch.einsum("bchw,bkc->bkhw", x, w)
    return x * self.logit_scale.exp() + self.bias


def _register(mod, d: dict):
    mod._aq_mods = nn.ModuleDict({k: v for k, v in d.items() if isinstance(v, nn.Module)})
    mod._aq = dict(d)


def quantize_attention(det_model: nn.Module, scope: str = "attn", bits: int = 8) -> list:
    """det_model: DetectionModel(WorldModel). 반환: 적용한 지점 설명 목록."""
    assert scope in ("attn", "attn_cls"), scope
    seq = det_model.model
    ipa_idx = [i for i, m in enumerate(seq) if type(m).__name__ == "ImagePoolingAttn"]
    first_ipa = min(ipa_idx) if ipa_idx else len(seq)
    done = []
    for i, m in enumerate(seq):
        name = type(m).__name__
        if name == "C2fAttn":
            a = m.attn
            const = i < first_ipa
            d = {"const": const, "g": ConstQ(bits) if const else QPoint(bits),
                 "emb": QPoint(bits), "aw": QPoint(bits)}
            if not const:
                d["gl"] = QuantLinear(a.gl, bits)
            _register(a, d)
            if not hasattr(m, "cv2_main"):            # 게이트 교환이 아니면 원래 attention forward를 교체
                a.forward = types.MethodType(_msab_forward, a)
            done.append(f"{i}.C2fAttn({'const' if const else 'runtime'} text)")
        elif name == "ImagePoolingAttn":
            # 같은 v1 구조에서 ImagePoolingAttn의 텍스트 입력은 원래 텍스트(상수)
            m.key[1] = QuantLinear(m.key[1], bits)
            m.value[1] = QuantLinear(m.value[1], bits)
            m.proj = QuantLinear(m.proj, bits)
            _register(m, {"q": ConstQ(bits), "k": QPoint(bits), "v": QPoint(bits), "aw": QPoint(bits)})
            m.forward = types.MethodType(_ipa_forward, m)
            done.append(f"{i}.ImagePoolingAttn")
    if scope == "attn_cls":
        head = seq[-1]
        for li, c in enumerate(head.cv4):
            _register(c, {"x": QPoint(bits), "w": ConstQ(bits)})
            c.forward = types.MethodType(_cls_forward, c)
            done.append(f"head.cv4.{li}")
    return done
