"""확정 프로토콜 s 결과: runs/158_final + runs/154_protocol(s seed 0·1 재사용). 조건별 seed 값과 BRECQ 대비 짝 Δ."""
import glob, os, math, statistics as st, sys
exec(open('147_main/aggregate.py').read().split("data = {}")[0])
D = {}
for f in glob.glob('158_final/T*-s[0-9].log') + glob.glob('154_protocol/s-W4A*-all-s[0-9].log'):
    b = os.path.basename(f)[:-4]; s = int(b.split('-s')[-1])
    st_ = 'W' + b.split('-W')[1].split('-')[0][:3] if '154_' in f else 'W' + b.split('-W')[1][:3]
    for c, m in parse(f).items(): D.setdefault((st_, c), {}).setdefault(s, m)
K = [('coco','COCO',1),('lvis','LVIS',100),('apr','APr',100),('lvis_flip','LVIS_flip',1),('lvis_lost','lost',1)]
ORDER = ['naive','brecq','qdrop','adaround','combined','brecq+M','brecq+PM','brecq+GPM','brecq+G','brecq+P','brecq+GP','brecq+GM','brecq+R','brecq+H','qdrop+M','qdrop+PM']
from scipy import stats
for st_ in ('W8A8','W4A8','W4A6','W4A5'):
    conds = sorted({c for (s_, c) in D if s_ == st_}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    if not conds: continue
    print(f"\n### {st_}\n\n| 조건 | seed | " + " | ".join(k[1] for k in K) + " | Δ LVIS vs BRECQ (짝) | Δ flip |")
    print("|" + "---|" * (len(K) + 4))
    b = D.get((st_, 'brecq'), {})
    for c in conds:
        d = D[(st_, c)]; ss = sorted(d)
        cells = []
        for k in K:
            v = [d[s][k[0]] * k[2] for s in ss if d[s].get(k[0]) is not None]
            cells.append((f"{st.mean(v):.0f}" if k[0]=='lvis_lost' else f"{st.mean(v):.2f}") if v else '-')
        dl = ''; df = ''
        if c != 'brecq':
            com = [s for s in ss if s in b]
            if com:
                x = [d[s]['lvis']*100 - b[s]['lvis']*100 for s in com]; y = [d[s]['lvis_flip'] - b[s]['lvis_flip'] for s in com]
                ci = lambda z: f" ± {stats.t.ppf(0.975,len(z)-1)*st.stdev(z)/math.sqrt(len(z)):.2f}" if len(z) > 1 else ''
                dl = f"{st.mean(x):+.2f}{ci(x)}"; df = f"{st.mean(y):+.2f}{ci(y)}"
        print(f"| {c} | {','.join(map(str,ss))} | " + " | ".join(cells) + f" | {dl} | {df} |")
