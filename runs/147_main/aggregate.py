"""runs/147 + 이전 단독 run(seed 0/1)을 모아 조건별 평균±표준편차와 BRECQ 대비 짝비교 k/n을 낸다."""
import re, os, glob, statistics as st, sys
R = '/home/taeho/promptcal-ptq/runs'
def parse(path):
    out = {}
    if not os.path.exists(path): return out
    txt = open(path, errors='ignore').read()
    if '이론적 모델 크기' not in txt: return out
    for blk, cols in (('COCO AP |', ['coco','s_ap','heval_ap','lvis','apr','apc','apf']),
                      ('Heval_flip |', ['heval_flip','top1','upir','lost','corr','lvis_flip','lvis_lost','lcorr','t'])):
        i = txt.index(blk); lines = txt[i:].split('\n')[1:]
        for ln in lines:
            if not ln.strip() or ln.startswith('='): break
            parts = [p.strip() for p in ln.split('|')]
            name = parts[0]
            if name == 'FP32': continue
            vals = [float(p.rstrip('%')) if p not in ('-','') else None for p in parts[1:]]
            out.setdefault(name, {}).update(dict(zip(cols, vals)))
    return out
data = {}  # (setting, cond) -> {seed: metrics}
def add(setting, seed, path, rename=None):
    for c, m in parse(path).items():
        c2 = (rename or {}).get(c, c)
        data.setdefault((setting, c2), {})[seed] = m
# 이전 단독 run (첫 조건 = RNG 위치 동일)
for s, f in ((0, '123_lowbit_vm/e2_w4a8_seed0.log'), (1, '123_lowbit_vm/e2_w4a8_seed1.log')):
    p = parse(f'{R}/{f}'); data.setdefault(('W4A8','brecq'), {})[s] = p['brecq']
for s in (0,1):
    data.setdefault(('W4A8','brecq+M'),{})[s] = parse(f'{R}/132_mig_tied/tied_seed{s}.log')['brecq']
    data.setdefault(('W4A8','brecq+PM'),{})[s] = parse(f'{R}/141_convw8_mig/seed{s}.log')['brecq']
    data.setdefault(('W4A8','brecq+R'),{})[s] = parse(f'{R}/134_leakfree/rand0_seed{s}.log')['brecq']
    data.setdefault(('W8A8','brecq+M'),{})[s] = parse(f'{R}/135_w8a8_mig/tied_seed{s}.log')['brecq']
data.setdefault(('W4A8','brecq+P'),{})[0] = parse(f'{R}/134_leakfree/sel_coco_seed0.log')['brecq']
data.setdefault(('W4A8','brecq+P'),{})[1] = parse(f'{R}/140_promptcal_stack/w4a8_convw8_seed1.log')['brecq']
data.setdefault(('W8A8','brecq'),{})[0] = parse(f'{R}/125_vm_w8a8/lam0_seed0.log')['brecq']
data.setdefault(('W8A8','brecq'),{})[1] = parse(f'{R}/127_vm_w8a8_seed1/identity_seed1.log')['brecq']
# runs/147
for f in glob.glob(f'{R}/147_main/*-s[0-9].log'):
    tag = os.path.basename(f)[:-4]; _, setting, s = tag.split('-'); s = int(s[1:])
    for c, m in parse(f).items():
        data.setdefault((setting, c), {}).setdefault(s, m)   # 이전 run 값이 있으면 유지
keys = ['coco','lvis','apr','lvis_flip','lvis_lost','heval_flip']
lower_better = {'lvis_flip','lvis_lost','heval_flip'}
for setting in ('W8A8','W4A8','W4A6','W4A5','W4A4'):
    conds = [c for (s_, c) in data if s_ == setting]
    if not conds: continue
    order = ['naive','brecq','brecq+M','brecq+P','brecq+PM','brecq+R','brecq+H','adaround','qdrop','qdrop+PM','combined']
    conds = sorted(conds, key=lambda c: order.index(c) if c in order else 99)
    print(f"\n=== {setting} ===  (평균 ± 표준편차, [n seed]; 괄호 = brecq 대비 개선 seed 수)")
    print(f"{'조건':>10} | n | " + " | ".join(f"{k:>16}" for k in keys))
    base = data.get((setting,'brecq'), {})
    for c in conds:
        d = data[(setting,c)]; seeds = sorted(d)
        cells = []
        for k in keys:
            v = [d[s][k] for s in seeds if d[s].get(k) is not None]
            if not v: cells.append(f"{'-':>16}"); continue
            mu = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0
            fmt = f"{mu:.4f}±{sd:.4f}" if k in ('lvis','apr') else (f"{mu:.0f}±{sd:.0f}" if k=='lvis_lost' else f"{mu:.2f}±{sd:.2f}")
            if c != 'brecq' and base:
                common = [s for s in seeds if s in base and base[s].get(k) is not None]
                win = sum((d[s][k] < base[s][k]) if k in lower_better else (d[s][k] > base[s][k]) for s in common)
                fmt += f"({win}/{len(common)})"
            cells.append(f"{fmt:>16}")
        print(f"{c:>10} | {len(seeds)} | " + " | ".join(cells))
