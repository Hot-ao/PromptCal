"""
격리 검증 (YOLOWorld/COCO 없이 torch만으로):
  T1. LearnableActScale.quantize_train 이 scale(log_m)에 nonzero grad를 흘리는가?
      (v1 방향 C의 'scale detach로 grad 끊김' 버그가 재발하지 않는지)
  T2. 그 grad가 LSQ residual 형태인가? (수치적으로 유한차분과 일치)
  T3. decision_loss 가 held-out fallback argmax를 겨냥하는가?
      (pseudo-label과 다른 argmax를 가진 quant를 CE가 줄이도록 grad를 주는가)
  T4. round_ste forward=round, backward=identity 확인.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from src.quant.fake_quant import ActObserver
from src.quant.promptcal_v2 import LearnableActScale, decision_loss, _round_ste

torch.manual_seed(0)
ok = True

# --- 준비: min/max로 freeze된 ActObserver 하나 ---
obs = ActObserver(bits=8)
x = torch.randn(4, 8, 5, 5) * 3.0 + 1.0
obs.observe(x); obs.freeze()
las = LearnableActScale(obs)

# ============ T1: scale에 grad가 흐르는가 ============
las.log_m.grad = None
xin = x.clone().requires_grad_(False)
xq = las.quantize_train(xin)
loss = (xq - (x + 0.5)).pow(2).mean()      # 임의의 하류 손실
loss.backward()
g = las.log_m.grad
print(f"[T1] log_m.grad = {None if g is None else g.item():.6e}")
t1 = (g is not None) and (abs(g.item()) > 1e-9)
print(f"[T1] scale grad nonzero: {'PASS' if t1 else 'FAIL'}")
ok &= t1

# ============ T2: autograd ∂x_hat/∂s 가 LSQ 닫힌형과 일치 ============
# (계단함수 유한차분이 아니라, LSQ 미분 공식 자체와의 일치가 올바른 검증.)
#   in-range: round(u)-u,  high-clamp: qmax-zp,  low-clamp: -zp   (u=x/s+zp)
# loss=sum(x_hat) 로 두면 upstream=1 이라 ∂loss/∂s = Σ closed_form.
las.log_m.grad = None
s = las.s_eff.detach(); zp = obs.zero_point; qmax = las.qmax
u = x / s + zp
closed = torch.where((u > 0) & (u < qmax), torch.round(u) - u,
          torch.where(u >= qmax, torch.full_like(u, float(qmax)) - zp, -zp))
grad_s_closed = closed.sum()                       # ∂Σx_hat/∂s
grad_logm_closed = (grad_s_closed * s).item()      # ∂/∂log_m = ∂/∂s · s

xq = las.quantize_train(x); (xq.sum()).backward()
ana = las.log_m.grad.item()
rel = abs(ana - grad_logm_closed) / (abs(grad_logm_closed) + 1e-8)
print(f"[T2] autograd={ana:.5f} closed-form={grad_logm_closed:.5f} rel_err={rel:.4f}")
t2 = rel < 1e-3
print(f"[T2] autograd == LSQ closed-form gradient: {'PASS' if t2 else 'FAIL'}")
ok &= t2

# ============ T3: decision_loss가 held-out fallback argmax를 겨냥 ============
# 시나리오: P=6 프롬프트. removed(H_cal)=[4,5], keep(S)=[0,1,2,3].
# anchor 하나가 FP에서 top-1=4(removed, confident). keep 안에서 FP fallback top-1 = col 2.
# quant는 keep 안에서 col 0을 top-1으로(=틀림). CE가 col 2 쪽으로 밀어야 한다.
A, P = 3, 6
sim_fp = torch.full((A, P), -5.0)
sim_fp[0, 4] = 6.0      # FP top-1 = removed col 4 (confident: sigmoid(6)~0.998)
sim_fp[0, 2] = 2.0      # keep 안 fallback 1위 = col 2
sim_fp[0, 0] = 1.0      # keep 안 2위 = col 0
# 나머지 anchor는 confident 아님(무시되도록 낮게)
sim_q = sim_fp.clone().detach()
sim_q[0, 0] = 3.0       # quant는 keep에서 col 0을 1위로(=fallback 뒤집힘)
sim_q[0, 2] = 2.0
sim_q.requires_grad_(True)

removed = torch.tensor([4, 5]); keep = torch.tensor([0, 1, 2, 3])
dl = decision_loss(sim_q, sim_fp, removed, keep, conf_thres=0.25, temp=1.0)
dl.backward()
# col 2(정답 fallback) logit을 올리는 방향이면 grad<0 (loss 감소 방향), col 0은 grad>0
g2 = sim_q.grad[0, 2].item(); g0 = sim_q.grad[0, 0].item()
print(f"[T3] decision_loss={dl.item():.4f}  d/d(col2)={g2:.4f}  d/d(col0)={g0:.4f}")
t3 = (dl.item() > 0) and (g2 < 0) and (g0 > 0)
print(f"[T3] CE pushes quant toward FP fallback top-1: {'PASS' if t3 else 'FAIL'}")
ok &= t3

# T3b: quant가 이미 FP와 같은 fallback을 고르면 loss가 작아야 함
sim_q_good = sim_fp.clone().detach().requires_grad_(True)
dl_good = decision_loss(sim_q_good, sim_fp, removed, keep, conf_thres=0.25, temp=1.0)
print(f"[T3b] loss(agree)={dl_good.item():.4f} < loss(flip)={dl.item():.4f}: "
      f"{'PASS' if dl_good.item() < dl.item() else 'FAIL'}")
ok &= (dl_good.item() < dl.item())

# ============ T4: round_ste ============
z = torch.tensor([1.2, 1.8, -0.4], requires_grad=True)
r = _round_ste(z); r.sum().backward()
f_ok = torch.allclose(r.detach(), torch.round(z.detach()))
b_ok = torch.allclose(z.grad, torch.ones_like(z))
print(f"[T4] round_ste forward==round:{f_ok}  backward==identity:{b_ok}: "
      f"{'PASS' if (f_ok and b_ok) else 'FAIL'}")
ok &= (f_ok and b_ok)

# ============ T5: 학습된 scale이 '추론'에 반영되는가 (회귀 가드) ============
# (7.04==7.04 버그: 추론 분기가 학습된 s_eff 대신 naive scale을 써서 결과가 naive와 동일했음)
import torch.nn as nn
from src.quant.quant_model import wrap_convs
import src.quant.promptcal_v2 as v2
from src.quant.fake_quant import QuantConv2d as _QC

mdl = nn.Sequential(nn.Conv2d(3,4,3,padding=1), nn.ReLU(), nn.Conv2d(4,4,1))
wrap_convs(mdl,8,8)
for mod in mdl.modules():
    if isinstance(mod, _QC):
        mod.a_obs.scale=torch.tensor(0.05); mod.a_obs.zero_point=torch.tensor(4.0)
        mod.a_obs.ready=True; mod.quantized=True
v2.convert_to_learnable_scale(mdl)
xin5 = torch.randn(1,3,8,8)
y0 = mdl(xin5).detach().clone()                       # log_m=0 → naive와 같아야
convs = v2.list_scale_convs(mdl)
with torch.no_grad():
    for i,c in enumerate(convs): c.act_scale.log_m.copy_(torch.tensor(0.3 if i==0 else -0.2))
y1 = mdl(xin5).detach().clone()                       # 학습된 scale → 달라야
moved = (y1 - y0).abs().max().item()
with torch.no_grad():
    for c in convs: c.act_scale.log_m.zero_()
yb = mdl(xin5).detach().clone()
back = (yb - y0).abs().max().item()
t5 = (moved > 1e-5) and (back < 1e-6)
print(f"[T5] learned-scale affects inference (Δ={moved:.4f}), log_m=0==naive (Δ={back:.1e}): "
      f"{'PASS' if t5 else 'FAIL'}")
ok &= t5

print("\n" + "="*50)
print("ALL PASS" if ok else "SOME FAILED")
print("="*50)
sys.exit(0 if ok else 1)
