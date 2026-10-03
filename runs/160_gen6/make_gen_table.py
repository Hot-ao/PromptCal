"""표 4 — 일반화 (v2, m; 확정 프로토콜, 6 seed). 사용: cd runs && python 160_gen6/make_gen_table.py
W4: runs/154_protocol(seed 0,1) + runs/160_gen6(seed 2~5). W8A8: runs/160_gen6만(154의 v2 W8A8은 L40S에서 돌려 제외).
짝 Δ는 BRECQ 대비 같은 seed 차이의 평균 ± 95% CI(t), 주 지표 Holm 보정(이 표 전체)."""
import glob, os, re, math, statistics as st
from scipy import stats
exec(open('147_main/aggregate.py').read().split("data = {}")[0])
D = {}
files = [f for f in glob.glob('154_protocol/[mv]*-W4A*-s[0-9].log') if 'old' not in f] + glob.glob('160_gen6/*-s[0-9].log')
for f in files:
    b = os.path.basename(f)[:-4]; mdl = b.split('-')[0]; stg = re.search(r'W\dA\d', b).group(0); s = int(b.split('-s')[-1])
    for c, m in parse(f).items(): D.setdefault((mdl, stg, c), {}).setdefault(s, m)
K = [('coco','COCO AP',1),('lvis','LVIS AP',100),('apr','APr',100),('lvis_flip','LVIS_flip (%)',1),('lvis_lost','LVIS_lost',1)]
ROWS = {'v2': ['naive','brecq','brecq+GPM','brecq+PM'], 'm': ['naive','brecq','brecq+GP']}
NAME = {'naive':'naive (RTN)','brecq':'BRECQ','brecq+GPM':'**우리 (GPM)**','brecq+PM':'PM (변형)','brecq+GP':'**우리 (GPM → M은 검사에서 꺼짐 = GP)**'}
MOD = {'v2':'YOLOv8s-WorldV2','m':'YOLOv8m-World'}
def paired(mdl, stg, c, k, sc):
    A, B = D[(mdl, stg, c)], D[(mdl, stg, 'brecq')]
    ss = [s for s in range(6) if s in A and s in B]
    x = [A[s][k]*sc - B[s][k]*sc for s in ss]; n = len(x); m = st.mean(x); sd = st.stdev(x); se = sd/math.sqrt(n)
    return m, stats.t.ppf(0.975, n-1)*se, (2*stats.t.sf(abs(m/se), n-1) if sd > 0 else 1.0), n
fam = []
for mdl, rows in ROWS.items():
    for stg in ('W8A8','W4A8','W4A5'):
        for c in rows[2:]:
            if (mdl, stg, c) not in D: continue
            for k, sc in (('lvis',100),('lvis_flip',1)): fam.append(((mdl, stg, c, k), paired(mdl, stg, c, k, sc)[2]))
fam.sort(key=lambda z: z[1]); M = len(fam); H = {}; run = 0
for i, (key, p) in enumerate(fam): run = max(run, min(1, (M-i)*p)); H[key] = run
for mdl, rows in ROWS.items():
    for stg in ('W8A8','W4A8','W4A5'):
        if (mdl, stg, 'brecq') not in D: continue
        print(f"\n**{MOD[mdl]} · {stg}**\n")
        print("| 방법 | seed | " + " | ".join(k[1] for k in K) + " | Δ LVIS AP (p_Holm) | Δ LVIS_flip (p_Holm) |"); print("|" + "---|" * (len(K) + 4))
        for c in rows:
            if (mdl, stg, c) not in D: continue
            d = D[(mdl, stg, c)]; ss = sorted(d)
            cells = [(f"{st.mean([d[s][k]*sc for s in ss]):.0f} ± {st.stdev([d[s][k]*sc for s in ss]):.0f}" if k == 'lvis_lost'
                      else f"{st.mean([d[s][k]*sc for s in ss]):.2f} ± {st.stdev([d[s][k]*sc for s in ss]):.2f}") for k, _, sc in K]
            if c in ('naive','brecq'): dd = ' |  |'
            else:
                r = []
                for k, sc in (('lvis',100),('lvis_flip',1)):
                    m_, ci, _, n = paired(mdl, stg, c, k, sc); r.append(f"{m_:+.2f} ± {ci:.2f} ({H[(mdl, stg, c, k)]:.2g})")
                dd = ' | ' + ' | '.join(r) + ' |'
            cocos = [d[s]['coco'] for s in ss]; med = st.median(cocos); col = [s for s in ss if d[s]['coco'] < med - 5]
            print(f"| {NAME[c]}{' ⚠ seed ' + ','.join(map(str,col)) if col else ''} | {len(ss)} | " + " | ".join(cells) + dd)
print(f"\n_Holm family(표 4): {M}개._")
