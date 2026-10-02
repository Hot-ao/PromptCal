"""attn_cls 6 seed 짝 Δ ± 95% CI (그리고 같은 seed의 attention FP 기존 결과와 나란히)."""
import math, statistics as st
from scipy import stats
exec(open('153_attnq/agg.py').read().split("seeds = sorted")[0])
K = [('coco','COCO',1),('lvis','LVIS AP',100),('apr','APr',100),('lvis_flip','LVIS_flip',1),('lvis_lost','LVIS_lost',1),('heval_flip','Heval_flip',1)]
def pair(lv, st_, a, b, k, excl=()):
    da, db = Q.get((lv, st_, a), {}), Q.get((lv, st_, b), {})
    ss = [s for s in range(6) if s in da and s in db and s not in excl]
    d = [da[s][k[0]] * k[2] - db[s][k[0]] * k[2] for s in ss]
    n = len(d); m = st.mean(d)
    ci = stats.t.ppf(0.975, n-1) * st.stdev(d) / math.sqrt(n) if n > 1 else float('nan')
    p = float(2 * stats.t.sf(abs(m / (st.stdev(d) / math.sqrt(n))), n-1)) if n > 1 and st.stdev(d) > 0 else float('nan')
    return n, m, ci, p
def absm(lv, st_, c, k, excl=()):
    d = Q.get((lv, st_, c), {}); v = [d[s][k[0]] * k[2] for s in range(6) if s in d and s not in excl]
    return st.mean(v), st.stdev(v), len(v)
for st_ in ('W4A8', 'W4A5'):
    for excl, tag in (((), '6 seed 전부'), ((0,), 'seed0 제외(BRECQ 붕괴)')) if st_ == 'W4A5' else (((), '6 seed 전부'),):
        print(f"\n#### {st_} — {tag}\n")
        print("| 수준 | 조건 | n | " + " | ".join(k[1] for k in K) + " |"); print("|" + "---|" * (len(K) + 3))
        for lv in ('none', 'attn_cls'):
            for c in ('brecq', 'brecq+PM', 'brecq+GPM'):
                cells = []
                for k in K:
                    m, sd, n = absm(lv, st_, c, k, excl); cells.append(f"{m:.0f} ± {sd:.0f}" if k[0]=='lvis_lost' else f"{m:.2f} ± {sd:.2f}")
                print(f"| {lv} | {c} | {n} | " + " | ".join(cells) + " |")
        print("\n| 수준 | 비교 | n | " + " | ".join(k[1] for k in K) + " |"); print("|" + "---|" * (len(K) + 3))
        for lv in ('none', 'attn_cls'):
            for a, b in (('brecq+GPM','brecq'),('brecq+PM','brecq'),('brecq+GPM','brecq+PM')):
                cells = []
                for k in K:
                    n, m, ci, p = pair(lv, st_, a, b, k, excl)
                    f = (lambda x: f"{x:+.0f}") if k[0]=='lvis_lost' else (lambda x: f"{x:+.2f}")
                    cells.append(f"{f(m)} ± {abs(ci):.2f}" + (f" (p {p:.2g})" if k[0] in ('lvis','lvis_flip') else ""))
                print(f"| {lv} | {a} − {b} | {n} | " + " | ".join(cells) + " |")
        # 같은 조건의 수준 간 차이(attn_cls − none)
        print("\n| 조건 | attn_cls − none: LVIS AP | LVIS_flip |"); print("|---|---|---|")
        for c in ('brecq', 'brecq+PM', 'brecq+GPM'):
            r = []
            for k in (K[1], K[3]):
                da, db = Q[('attn_cls', st_, c)], Q[('none', st_, c)]
                ss = [s for s in range(6) if s in da and s in db and s not in excl]
                d = [da[s][k[0]]*k[2] - db[s][k[0]]*k[2] for s in ss]
                r.append(f"{st.mean(d):+.2f} ± {stats.t.ppf(0.975,len(d)-1)*st.stdev(d)/math.sqrt(len(d)):.2f} (n={len(d)})")
            print(f"| {c} | " + " | ".join(r) + " |")
