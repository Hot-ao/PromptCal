"""표 2 (W4A5 ablation)·표 3 (W4A8 선택 기준 대조, QDrop plug-in) 마크다운. 사용: cd runs && python 158_final/make_table23.py
절대값: 6 seed 평균 ± 표준편차. 짝 Δ: 기준 대비 같은 seed 차이 평균 ± 95% CI. 표마다 주 지표(LVIS AP, LVIS_flip) Holm 보정.
W4A5는 seed 0에서 BRECQ·QDrop이 붕괴해 BRECQ/QDrop 기준 비교는 seed 0 제외 값을 함께 낸다."""
import math, statistics as st
from scipy import stats
exec(open('158_final/agg.py').read().split("K = [")[0])
K = [('coco','COCO AP',1),('lvis','LVIS AP',100),('apr','APr',100),('lvis_flip','LVIS_flip (%)',1),('lvis_lost','LVIS_lost',1)]
def paired(st_, a, b, k, sc, excl=()):
    A, B = D[(st_, a)], D[(st_, b)]
    ss = [s for s in range(6) if s in A and s in B and s not in excl]
    x = [A[s][k]*sc - B[s][k]*sc for s in ss]; n = len(x); m = st.mean(x); sd = st.stdev(x); se = sd/math.sqrt(n)
    return m, stats.t.ppf(0.975, n-1)*se, (2*stats.t.sf(abs(m/se), n-1) if sd > 0 else 1.0)
def holm(comps):
    fam = []
    for st_, a, b, ex in comps:
        for k, sc in (('lvis',100),('lvis_flip',1)): fam.append(((st_, a, b, k), paired(st_, a, b, k, sc, ex)[2]))
    fam.sort(key=lambda z: z[1]); M = len(fam); out = {}; run = 0
    for i, (key, p) in enumerate(fam): run = max(run, min(1, (M-i)*p)); out[key] = run
    return out, M
def absrow(st_, c, name):
    d = D[(st_, c)]; ss = sorted(d)[:6]; assert len(ss) == 6, (st_, c, ss)
    cells = [(f"{st.mean([d[s][k]*sc for s in ss]):.0f} ± {st.stdev([d[s][k]*sc for s in ss]):.0f}" if k == 'lvis_lost'
              else f"{st.mean([d[s][k]*sc for s in ss]):.2f} ± {st.stdev([d[s][k]*sc for s in ss]):.2f}") for k, _, sc in K]
    return f"| {name} | " + " | ".join(cells)
def dcell(st_, a, b, H, ex=()):
    r = []
    for k, sc in (('lvis',100),('lvis_flip',1)):
        m, ci, _ = paired(st_, a, b, k, sc, ex); r.append(f"{m:+.2f} ± {ci:.2f} ({H[(st_, a, b, k)]:.2g})")
    return r
hdr = lambda extra: "| 방법 | " + " | ".join(k[1] for k in K) + extra
# ---------------- 표 2
abl = [('brecq','BRECQ'),('brecq+G','+G'),('brecq+P','+P'),('brecq+M','+M'),('brecq+GP','+GP'),('brecq+GM','+GM'),('brecq+PM','+PM'),('brecq+GPM','+GPM')]
comps = [('W4A5', c, 'brecq', (0,)) for c, _ in abl[1:]] + [('W4A5','brecq+GPM','brecq+PM',()), ('W4A5','brecq+GM','brecq+G',()), ('W4A5','brecq+PM','brecq+P',()), ('W4A5','brecq+GPM','brecq+GP',())]
H2, M2 = holm(comps)
print("**표 2 — W4A5 ablation (BRECQ 위, 6 seed)**\n")
print(hdr(" | Δ LVIS AP vs BRECQ, seed0 제외 (p_Holm) | Δ LVIS_flip vs BRECQ, seed0 제외 (p_Holm) |")); print("|" + "---|" * (len(K) + 3))
for c, n in abl:
    print(absrow('W4A5', c, n) + (" |  |  |" if c == 'brecq' else " | " + " | ".join(dcell('W4A5', c, 'brecq', H2, (0,))) + " |"))
print("\n**표 2b — 요소 추가의 짝 효과 (W4A5, 6 seed, 붕괴 seed 영향 없음)**\n")
print("| 비교 | 뜻 | Δ LVIS AP (p_Holm) | Δ LVIS_flip (p_Holm) |\n|---|---|---|---|")
for a, b, why in (('brecq+GPM','brecq+PM','G 추가 (PM 위)'), ('brecq+GM','brecq+G','M 추가 (G 위)'), ('brecq+PM','brecq+P','M 추가 (P 위)'), ('brecq+GPM','brecq+GP','M 추가 (GP 위)')):
    print(f"| {a.split('+')[1]} − {b.split('+')[1]} | {why} | " + " | ".join(dcell('W4A5', a, b, H2)) + " |")
print(f"\n_Holm family(표 2): {M2}개._")
d = D[('W4A5','brecq')]; print("_W4A5 seed 0 COCO AP: " + ", ".join(f"{n} {D[('W4A5',c)][0]['coco']:.2f}" for c, n in abl) + "._")
# ---------------- 표 3
print("\n**표 3a — 보호 대상 선택 기준 (W4A8, 같은 예산 1.5%, M 없음, 6 seed)**\n")
sel = [('brecq','BRECQ'),('brecq+R','+R (무작위)'),('brecq+H','+H (HAWQ식 출력 MSE)'),('brecq+P','+P (누수 없는 판정 진단)')]
comps3 = [('W4A8', c, 'brecq', ()) for c, _ in sel[1:]] + [('W4A8','brecq+P','brecq+R',()), ('W4A8','brecq+P','brecq+H',())]
for st_, b in (('W8A8','qdrop'),('W4A8','qdrop'),('W4A6','qdrop'),('W4A5','qdrop')):
    comps3.append((st_, 'qdrop+M' if st_ == 'W8A8' else 'qdrop+PM', 'qdrop', (0,) if st_ == 'W4A5' else ()))
H3, M3 = holm(comps3)
print(hdr(" | Δ LVIS AP vs BRECQ (p_Holm) | Δ LVIS_flip vs BRECQ (p_Holm) |")); print("|" + "---|" * (len(K) + 3))
for c, n in sel:
    print(absrow('W4A8', c, n) + (" |  |  |" if c == 'brecq' else " | " + " | ".join(dcell('W4A8', c, 'brecq', H3)) + " |"))
print("\n| 비교 | Δ LVIS AP (p_Holm) | Δ LVIS_flip (p_Holm) |\n|---|---|---|")
for a, b in (('brecq+P','brecq+R'), ('brecq+P','brecq+H')):
    print(f"| {a.split('+')[1]} − {b.split('+')[1]} | " + " | ".join(dcell('W4A8', a, b, H3)) + " |")
print("\n**표 3b — QDrop 위의 plug-in (6 seed)**\n")
print("| 설정 | 방법 | COCO AP | LVIS AP | LVIS_flip (%) | Δ LVIS AP vs QDrop (p_Holm) | Δ LVIS_flip vs QDrop (p_Holm) |\n|---|---|---|---|---|---|---|")
for st_ in ('W8A8','W4A8','W4A6','W4A5'):
    ours = 'qdrop+M' if st_ == 'W8A8' else 'qdrop+PM'; ex = (0,) if st_ == 'W4A5' else ()
    for c in ('qdrop', ours):
        d = D[(st_, c)]; ss = sorted(d)[:6]
        mm = lambda k, sc: st.mean([d[s][k]*sc for s in ss])
        tail = " |  |  |" if c == 'qdrop' else " | " + " | ".join(dcell(st_, c, 'qdrop', H3, ex)) + (" (seed0 제외)" if ex else "") + " |"
        print(f"| {st_} | {'QDrop' if c == 'qdrop' else '**QDrop + ' + c.split('+')[1] + '**'} | {mm('coco',1):.2f} | {mm('lvis',100):.2f} | {mm('lvis_flip',1):.2f}" + tail)
print(f"\n_Holm family(표 3): {M3}개._")
