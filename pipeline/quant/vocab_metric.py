"""
Vocabulary-metric reconstruction (09-28, 저비트·head 포함 설계).

동기: ContrastiveHead는 sim_j = tau_l * <x_hat, w_hat_j> + b_l 이다(x = cv3 출력 =
cv4 입력). 프롬프트 j의 유사도 오차를 1차 근사하면 Delta_j ~= tau_l <Delta x_hat, w_hat_j>
이고, 프롬프트 분포 V에 대한 기대 제곱오차는 닫힌 형태의 2차형식이 된다:

    E_{w~V}[Delta_j^2] = tau_l^2 * Delta x_hat^T C_V Delta x_hat,   C_V = E_{w~V}[w w^T]

즉 "임의 vocabulary에서의 기대 유사도 오차"를 특정 프롬프트 집합에 맞추지 않고
dense한 재구성 손실로 쓸 수 있다. top-k margin(claim21/22)처럼 sparse한 앵커 신호로
rounding을 움직이는 게 아니라, BRECQ와 같은 재구성(FP에 고정된) 손실의 **metric만**
바꾸는 것이다 -- 옵티마이저·자유도·비용은 BRECQ와 같다.

두 경로:
  (1) exact  : head cv3의 마지막 1x1 conv(출력 = 임베딩 그 자체)는 위 2차형식을
               그대로 쓴다(선형화 없이 normalize(q) - normalize(f)로 계산).
  (2) fisher : 그 앞 블록/conv는 BRECQ의 FIM 가중 재구성과 같은 방식으로 metric을
               끌어올린다. anchor마다 u_a ~ N(0, C)를 뽑아
                   s = sum_a sqrt(w_a) * tau_l * <x_hat_a, u_a>
               를 해당 target 출력 z로 역전파하면 E[(ds/dz)^2] = sum_a w_a diag(J_a^T C J_a)
               (anchor 간 교차항은 기댓값 0) -- 즉 끌어올린 metric의 대각이다.
               유사도 경로와 무관한 target(head cv2 = box 회귀)은 gradient가 None이라
               자동으로 순수 재구성 손실을 유지한다.

C 구성: C = Sigma_V(centered) + lam_mean * mu mu^T. lam_mean=1이면 비중심 2차 모멘트
E[w w^T]와 같다. Sigma는 프롬프트 간 순위(상대 logit), mu 방향은 모든 프롬프트 logit을
같이 움직이는 성분(절대 점수 = threshold/AP)을 담당한다. 평균 고유값이 1이 되도록
정규화한다(항등 metric과 같은 척도).

anchor 가중치 w_a: FP region이 bank의 어떤 프롬프트에든 확신하는 정도
max_j sigmoid(sim_fp[a, j]) + floor. 배경 anchor가 수천 개라 균등 가중이면 신호를
지배한다. floor는 배경이 threshold를 넘어 새 오검출이 되는 경우를 완전히 무시하지
않기 위함.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


class VocabMetric:
    def __init__(self, text_feats: torch.Tensor, lam_mean: float = 1.0, identity: bool = False,
                 anchor_weight: str = "conf", conf_floor: float = 0.05, n_samples: int = 4,
                 mix: float = 0.5, seed: int = 0):
        """text_feats: [N, D] FP CLIP 텍스트 임베딩(metric을 정의하는 vocabulary bank).
        identity=True면 C=I(방향 보존 재구성, bank는 anchor 가중치에만 쓰임).
        mix: fisher 경로에서 (1-mix)*재구성 + mix*vocab-가중 재구성. exact 경로는 mix와 무관하게
        순수 vocab 손실(cv3 마지막 conv 출력은 정규화 후 유사도에만 쓰이므로)."""
        assert anchor_weight in ("conf", "uniform")
        assert 0.0 <= mix <= 1.0
        T = F.normalize(text_feats.detach().float(), dim=-1)
        N, D = T.shape
        if identity:
            C = torch.eye(D, device=T.device)
        else:
            mu = T.mean(0)
            Tc = T - mu
            C = Tc.T @ Tc / N + lam_mean * torch.outer(mu, mu)
        C = C / torch.trace(C) * D                          # 평균 고유값 1
        evals, evecs = torch.linalg.eigh(C)
        self.L = evecs * evals.clamp(min=0).sqrt()          # C = L L^T
        self.C = C
        self.bank = T
        self.identity = identity
        self.anchor_weight = anchor_weight
        self.conf_floor = conf_floor
        self.n_samples = n_samples
        self.mix = mix
        self.seed = seed
        self._gen = None
        self.levels = None                                  # [(tau_l, b_l)] -- bind_head()에서 채움

    # ------------------------------------------------------------------ setup
    def to(self, device):
        self.C, self.L, self.bank = self.C.to(device), self.L.to(device), self.bank.to(device)
        return self

    def bind_head(self, fp_head):
        """FP head의 cv4(ContrastiveHead) 레벨별 (tau, bias)를 고정값으로 읽는다."""
        levels = []
        for sub in fp_head.cv4:
            name = type(sub).__name__
            if name != "ContrastiveHead":
                # BNContrastiveHead(v2 가중치)는 L2 정규화 대신 BN이라 위 metric 유도가 성립하지 않는다.
                raise NotImplementedError(f"vocab metric은 ContrastiveHead 전용입니다(현재 {name})")
            levels.append((float(sub.logit_scale.detach().exp()), float(sub.bias.detach())))
        self.levels = levels
        return self

    def generator(self, device):
        """전용 RNG. 전역 torch RNG를 소비하지 않아야 BRECQ 배치 샘플링 스트림과 다른 조건의
        RNG 위치가 이 기능 때문에 밀리지 않는다."""
        if self._gen is None or self._gen.device != torch.device(device):
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.seed)
        return self._gen

    def effective_rank(self) -> float:
        """participation ratio (tr C)^2 / tr(C^2). D면 등방, 작을수록 metric이 소수 방향에 집중."""
        return float(torch.trace(self.C) ** 2 / torch.trace(self.C @ self.C))

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def flat(x: torch.Tensor) -> torch.Tensor:
        """[1, D, H, W] -> [HW, D]"""
        B, D, H, W = x.shape
        return x.reshape(B, D, H * W)[0].transpose(0, 1)

    @torch.no_grad()
    def anchor_weights(self, x_level: torch.Tensor, level: int, chunk: int = 2048) -> torch.Tensor:
        """x_level: FP cv4 입력 [1, D, H, W]. 반환 [HW]."""
        xf = F.normalize(self.flat(x_level).float(), dim=-1)
        if self.anchor_weight == "uniform":
            return torch.ones(xf.shape[0], device=xf.device)
        tau, b = self.levels[level]
        out = []
        for i in range(0, xf.shape[0], chunk):
            sim = tau * (xf[i:i + chunk] @ self.bank.T) + b
            out.append(torch.sigmoid(sim.max(dim=1).values))
        return torch.cat(out) + self.conf_floor

    def sample_objective(self, x_levels, weights, gen) -> torch.Tensor:
        """s = sum_l sum_a sqrt(w_a) tau_l <x_hat_a, u_a>,  u_a ~ N(0, C).
        x_levels: 레벨별 FP cv4 입력(그래프 유지), weights: 레벨별 [HW]."""
        s = 0.0
        for lvl, (x, w) in enumerate(zip(x_levels, weights)):
            xf = F.normalize(self.flat(x).float(), dim=-1)
            eps = torch.randn(xf.shape, device=xf.device, generator=gen)
            u = eps @ self.L.T
            tau = self.levels[lvl][0]
            s = s + (w.sqrt().unsqueeze(1) * tau * xf * u).sum()
        return s

    def exact_loss(self, q: torch.Tensor, f: torch.Tensor, w: torch.Tensor, level: int) -> torch.Tensor:
        """q/f: [B, D, H, W] (quant/FP의 cv3 마지막 conv 출력), w: [B, HW].
        anchor 가중 평균 tau^2 * v^T C v,  v = normalize(q) - normalize(f)."""
        B, D, H, W = q.shape
        qf = F.normalize(q.reshape(B, D, H * W).transpose(1, 2), dim=-1)
        ff = F.normalize(f.reshape(B, D, H * W).transpose(1, 2), dim=-1)
        v = qf - ff                                          # [B, HW, D]
        e = ((v @ self.C) * v).sum(-1)                       # [B, HW]
        tau = self.levels[level][0]
        return tau ** 2 * (e * w).sum() / w.sum().clamp(min=1e-8)


def load_vocab(spec: str, coco_names, lvis_names):
    """--vm-vocab 해석. 반환: (이름 목록, identity 여부)."""
    if spec == "coco":
        return list(coco_names), False
    if spec == "lvis":
        return list(lvis_names), False                      # oracle 상한(평가 vocabulary 누수) -- ablation 전용
    if spec == "identity":
        return list(coco_names), True                       # C=I, bank는 anchor 가중치에만
    with open(spec) as fh:
        names = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    return names, False


@torch.no_grad()
def encode_text_bank(model_cls, weights, names, device, batch=256):
    """FP YOLO-World의 CLIP 텍스트 인코더로 bank 임베딩 [N, D]를 만든다. 평가 모델과 분리된
    별도 인스턴스를 써서 fp/양자화 모델의 txt_feats(COCO-80)를 건드리지 않는다."""
    m = model_cls(weights)
    feats = []
    for i in range(0, len(names), batch):
        m.set_classes(names[i:i + batch])
        feats.append(m.model.txt_feats[0].detach().float().cpu())
    del m
    return torch.cat(feats).to(device)
