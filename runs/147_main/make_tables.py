"""runs/147 + 이전 단독 run을 모아 결과 표(markdown)를 만든다 (10-01 개정: 짝 Δ ± 95% CI).
사용: cd runs && python3 147_main/make_tables.py > 표.md

표 A (설정별 절대값): 6 seed가 모두 있는 조건만, 같은 6 seed 평균 ± 표준편차.
    seed가 모자란 조건은 표 아래에 seed 집합을 밝히고 **같은 seed 집합의 기준 평균**과 함께 따로 적는다.
표 B (짝 비교): 비교마다 공통 seed에서 Δ = (조건 − 기준)의 평균 ± 95% CI(t 분포, n−1 자유도).
    지표 방향: flip/lost는 낮을수록 좋음(Δ<0이 개선). LVIS AP, APr는 ×100(AP점) 단위로 표시.
주 지표(LVIS AP, LVIS_flip)는 모든 표 B 비교를 한 family로 보고 짝 t-검정 p값에 Holm 보정을 한다.
나머지 지표는 보조 지표(보정 없음, 서술용).
seed 간 H_eval_AP 편차가 큰 것은 seed마다 held-out 클래스 분할이 다르기 때문이다(예: W4A8 BRECQ seed0 30.7, seed5 38.2)."""
import statistics as st, glob, os, math
from scipy import stats
exec(open('147_main/aggregate.py').read().split("keys = ")[0])
for f in glob.glob('147_main/R-*-s[0-9].log'):
    _, setting, s = os.path.basename(f)[:-4].split('-'); s = int(s[1:]); p = parse(f)
    if 'adaround' in p:
        data.setdefault((setting, 'adaround(MSE)'), {})[s] = p['adaround']

# W4A5 QDrop seed 2~5(runs/151_gen/Q-*)
for f in glob.glob('151_gen/Q-W4A5-s[0-9].log'):
    s_ = int(os.path.basename(f)[:-4].split('-s')[1])
    for c_, m_ in parse(f).items():
        data.setdefault(('W4A5', c_), {}).setdefault(s_, m_)
# W8A8 짝비교 baseline 재실행(runs/151_gen, --adaround-act-observer mse)
for f in glob.glob('151_gen/B-W8A8-s[0-9].log'):
    s_ = int(os.path.basename(f)[:-4].split('-s')[1]); p_ = parse(f)
    for c_, m_ in p_.items():
        data.setdefault(('W8A8', 'adaround(MSE)' if c_ == 'adaround' else c_), {}).setdefault(s_, m_)

AP100 = {'lvis', 'apr', 'apc', 'apf'}
KEYS = [('coco', 'COCO AP'), ('heval_ap', 'H_eval_AP'), ('lvis', 'LVIS AP'), ('apr', 'APr'),
        ('lvis_flip', 'LVIS_flip'), ('lvis_lost', 'LVIS_lost'), ('heval_flip', 'Heval_flip'),
        ('top1', 'Top1_flip'), ('lost', 'lost')]
PRIMARY = ['lvis', 'lvis_flip']
SIZE = {'W8A8': lambda c: '12.08'}


def sz(setting, c):
    if setting == 'W8A8':
        return '12.08'
    suf = c.split('+')[1] if '+' in c else ''
    if 'R' in suf: return '6.24'
    if 'H' in suf: return '6.23'
    if 'G' in suf and 'P' in suf: return '6.16'
    if 'P' in suf: return '6.22'
    return '6.14'


def val(d, s, k):
    v = d[s].get(k)
    return None if v is None else (v * 100 if k in AP100 else v)


def fmt(x, k):
    return f"{x:.0f}" if k in ('lvis_lost', 'lost') else f"{x:.2f}"


CONDS = {
    'W8A8': ['naive', 'brecq', 'brecq+M', 'adaround(MSE)', 'qdrop', 'qdrop+M'],
    'W4A8': ['naive', 'brecq', 'adaround(MSE)', 'qdrop', 'combined', 'brecq+M', 'brecq+P', 'brecq+PM', 'brecq+R', 'brecq+H',
             'brecq+G', 'brecq+GM', 'brecq+GPM', 'qdrop+PM'],
    'W4A6': ['naive', 'brecq', 'qdrop', 'combined', 'brecq+M', 'brecq+P', 'brecq+PM', 'brecq+G', 'brecq+GM', 'brecq+GPM', 'qdrop+PM'],
    'W4A5': ['naive', 'brecq', 'qdrop', 'brecq+M', 'brecq+P', 'brecq+PM', 'brecq+G', 'brecq+GM', 'brecq+GPM', 'qdrop+PM'],
}
COMPS = {
    'W8A8': [('brecq+M', 'brecq'), ('qdrop', 'brecq'), ('adaround(MSE)', 'brecq'), ('qdrop+M', 'qdrop')],
    'W4A8': [('brecq+M', 'brecq'), ('brecq+P', 'brecq'), ('brecq+PM', 'brecq'), ('brecq+R', 'brecq'), ('brecq+H', 'brecq'),
             ('brecq+G', 'brecq'), ('brecq+GM', 'brecq'), ('brecq+GPM', 'brecq'), ('brecq+GPM', 'brecq+PM'),
             ('brecq+PM', 'brecq+H'), ('qdrop', 'brecq'), ('combined', 'brecq'), ('adaround(MSE)', 'brecq'), ('qdrop+PM', 'qdrop')],
    'W4A6': [('brecq+M', 'brecq'), ('brecq+P', 'brecq'), ('brecq+PM', 'brecq'), ('brecq+G', 'brecq'), ('brecq+GM', 'brecq'),
             ('brecq+GPM', 'brecq'), ('brecq+GPM', 'brecq+PM'), ('qdrop', 'brecq'), ('combined', 'brecq'), ('qdrop+PM', 'qdrop')],
    'W4A5': [('brecq+M', 'brecq'), ('brecq+P', 'brecq'), ('brecq+PM', 'brecq'), ('brecq+G', 'brecq'), ('brecq+GM', 'brecq'),
             ('brecq+GPM', 'brecq'), ('brecq+GPM', 'brecq+PM'), ('qdrop', 'brecq'), ('qdrop+PM', 'qdrop')],
}


def paired(setting, a, b, k):
    da, db = data.get((setting, a), {}), data.get((setting, b), {})
    ss = [s for s in sorted(set(da) & set(db)) if val(da, s, k) is not None and val(db, s, k) is not None]
    diffs = [val(da, s, k) - val(db, s, k) for s in ss]
    n = len(diffs)
    if n == 0:
        return None
    m = st.mean(diffs)
    if n > 1:
        sd = st.stdev(diffs); se = sd / math.sqrt(n)
        ci = stats.t.ppf(0.975, n - 1) * se
        p = 1.0 if sd == 0 else float(2 * stats.t.sf(abs(m / se), n - 1))
    else:
        ci, p = float('nan'), float('nan')
    return dict(n=n, seeds=ss, m=m, ci=ci, p=p)


# ---- Holm 보정(주 지표 family)
family = []
for setting, comps in COMPS.items():
    for a, b in comps:
        for k in PRIMARY:
            r = paired(setting, a, b, k)
            if r and r['n'] > 1:
                family.append(((setting, a, b, k), r['p']))
family.sort(key=lambda x: x[1])
M = len(family); holm = {}; run_max = 0.0
for i, (key, p) in enumerate(family):
    adj = min(1.0, (M - i) * p); run_max = max(run_max, adj); holm[key] = run_max

print(f"\n_주 지표(LVIS AP, LVIS_flip) Holm 보정 family 크기: {M}개 비교. p는 짝 t-검정(양측)._\n")
for setting in CONDS:
    conds = [c for c in CONDS[setting] if (setting, c) in data]
    full = [c for c in conds if len(data[(setting, c)]) >= 6]
    part = [c for c in conds if c not in full]
    print(f"\n#### {setting}\n")
    print("**A. 절대값 (6 seed 평균 ± 표준편차)**\n")
    print("| 조건 | 크기(MiB) | " + " | ".join(n for _, n in KEYS) + " |")
    print("|" + "---|" * (len(KEYS) + 2))
    for c in full:
        d = data[(setting, c)]; ss = sorted(d)[:6]
        cells = []
        for k, _ in KEYS:
            v = [val(d, s, k) for s in ss if val(d, s, k) is not None]
            cells.append(f"{fmt(st.mean(v), k)} ± {fmt(st.stdev(v), k) if k not in ('lvis_lost','lost') else f'{st.stdev(v):.0f}'}" if len(v) > 1 else fmt(v[0], k))
        print(f"| {c} | {sz(setting, c)} | " + " | ".join(cells) + " |")
    if part:
        print("\n_seed가 모자란 조건 (같은 seed 집합의 BRECQ 평균과 나란히):_\n")
        print("| 조건 (seed) | " + " | ".join(n for _, n in KEYS) + " |")
        print("|" + "---|" * (len(KEYS) + 1))
        for c in part:
            d = data[(setting, c)]; ss = sorted(d); base = data.get((setting, 'brecq'), {})
            cells = []
            for k, _ in KEYS:
                v = [val(d, s, k) for s in ss if val(d, s, k) is not None]
                bv = [val(base, s, k) for s in ss if s in base and val(base, s, k) is not None]
                cells.append(f"{fmt(st.mean(v), k)} (BRECQ {fmt(st.mean(bv), k)})" if v and bv else "-")
            print(f"| {c} ({','.join(map(str, ss))}) | " + " | ".join(cells) + " |")
    print("\n**B. 짝 비교: Δ = 조건 − 기준, 공통 seed 평균 ± 95% CI** (LVIS AP·APr는 AP점; flip/lost는 음수가 개선)\n")
    print("| 비교 | n | " + " | ".join(n for _, n in KEYS) + " |")
    print("|" + "---|" * (len(KEYS) + 2))
    for a, b in COMPS[setting]:
        if (setting, a) not in data or (setting, b) not in data:
            continue
        cells = []; n = None
        for k, _ in KEYS:
            r = paired(setting, a, b, k)
            if r is None:
                cells.append('-'); continue
            n = r['n']
            if math.isnan(r['ci']):
                ci = ''
            else:
                ci = f" ± {r['ci']:.0f}" if k in ('lvis_lost', 'lost') else f" ± {r['ci']:.2f}"
            t = f"{r['m']:+.0f}{ci}" if k in ('lvis_lost', 'lost') else f"{r['m']:+.2f}{ci}"
            if k in PRIMARY and (setting, a, b, k) in holm:
                t += f" (p_Holm {holm[(setting, a, b, k)]:.3g})"
            cells.append(t)
        print(f"| {a} − {b} | {n} | " + " | ".join(cells) + " |")
