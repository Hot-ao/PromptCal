"""표 1 (확정 프로토콜, s-World, 6 seed) 마크다운 생성. 사용: cd runs && python 158_final/make_table1.py
절대값: 6 seed 평균 ± 표준편차. 짝 Δ: BRECQ 대비 같은 seed 차이의 평균 ± 95% CI(t). 주 지표(LVIS AP, LVIS_flip)는
표 1 비교 전체를 한 family로 Holm 보정. W4A5는 seed 0에서 BRECQ·QDrop이 붕괴해 seed 0 제외 짝 Δ를 함께 낸다."""
import math, statistics as st
from scipy import stats
exec(open('158_final/agg.py').read().split("K = [")[0])
ROWS = {'W8A8': ['naive','brecq','qdrop','adaround','brecq+M','brecq+GPM'],
        'W4A8': ['naive','brecq','qdrop','adaround','combined','brecq+GPM','brecq+PM'],
        'W4A6': ['naive','brecq','qdrop','brecq+GPM','brecq+PM'],
        'W4A5': ['naive','brecq','qdrop','brecq+GPM','brecq+PM']}
NAME = {'naive':'naive (RTN)','brecq':'BRECQ','qdrop':'QDrop','adaround':'AdaRound','combined':'PromptCal (Combined)',
        'brecq+M':'M 단독','brecq+PM':'PM (변형, +1.3%)','brecq+GPM':'**우리 (GPM)**'}
K = [('coco','COCO AP',1),('lvis','LVIS AP',100),('apr','APr',100),('lvis_flip','LVIS_flip (%)',1),('lvis_lost','LVIS_lost',1)]
def paired(st_, c, k, sc, excl=()):
    a, b = D[(st_, c)], D[(st_, 'brecq')]
    ss = [s for s in range(6) if s in a and s in b and s not in excl]
    x = [a[s][k]*sc - b[s][k]*sc for s in ss]; n = len(x); m = st.mean(x); sd = st.stdev(x)
    se = sd/math.sqrt(n); return m, stats.t.ppf(0.975, n-1)*se, (2*stats.t.sf(abs(m/se), n-1) if sd > 0 else 1.0), n
fam = []
for st_, rows in ROWS.items():
    for c in rows[2:] if rows[1] == 'brecq' else []:
        for k, sc in (('lvis',100),('lvis_flip',1)):
            fam.append(((st_, c, k), paired(st_, c, k, sc)[2]))
fam.sort(key=lambda z: z[1]); M = len(fam); holm = {}; run = 0
for i, (key, p) in enumerate(fam):
    run = max(run, min(1, (M-i)*p)); holm[key] = run
out = []
for st_, rows in ROWS.items():
    out.append(f"\n**{st_}**\n")
    out.append("| 방법 | " + " | ".join(k[1] for k in K) + " | Δ LVIS AP (p_Holm) | Δ LVIS_flip (p_Holm) |")
    out.append("|" + "---|" * (len(K) + 3))
    for c in rows:
        d = D[(st_, c)]; ss = sorted(d)[:6]; assert len(ss) == 6, (st_, c, ss)
        cells = []
        for k, _, sc in K:
            v = [d[s][k]*sc for s in ss]
            cells.append(f"{st.mean(v):.0f} ± {st.stdev(v):.0f}" if k == 'lvis_lost' else f"{st.mean(v):.2f} ± {st.stdev(v):.2f}")
        if c in ('naive', 'brecq'):
            dl = df = ''
        else:
            m1, c1, _, _ = paired(st_, c, 'lvis', 100); m2, c2, _, _ = paired(st_, c, 'lvis_flip', 1)
            dl = f"{m1:+.2f} ± {c1:.2f} ({holm[(st_, c, 'lvis')]:.2g})"; df = f"{m2:+.2f} ± {c2:.2f} ({holm[(st_, c, 'lvis_flip')]:.2g})"
            if st_ == 'W4A5':
                m1, c1, _, _ = paired(st_, c, 'lvis', 100, (0,)); m2, c2, _, _ = paired(st_, c, 'lvis_flip', 1, (0,))
                dl += f"<br>seed0 제외: {m1:+.2f} ± {c1:.2f}"; df += f"<br>seed0 제외: {m2:+.2f} ± {c2:.2f}"
        out.append(f"| {NAME[c]} | " + " | ".join(cells) + f" | {dl} | {df} |")
print("\n".join(out))
print(f"\n_Holm 보정 family: 표 1의 BRECQ 대비 비교 {M//2}개 × 주 지표 2개 = {M}개._")
b = D[('W4A5','brecq')]; q = D[('W4A5','qdrop')]
print(f"_W4A5 seed별 COCO AP — BRECQ: {', '.join(f'{b[s]['coco']:.2f}' for s in range(6))}; QDrop: {', '.join(f'{q[s]['coco']:.2f}' for s in range(6))}._")
