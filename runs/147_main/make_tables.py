"""runs/147 + 이전 단독 run을 모아 설정별 전체 지표 표(markdown)를 만든다.
사용: cd runs && python3 147_main/make_tables.py > 표.md
괄호 = 비교 대상(기본 brecq, qdrop+PM은 qdrop)보다 좋아진 seed 수 / 공통 seed 수."""
import statistics as st, glob, os
exec(open('147_main/aggregate.py').read().split("keys = ")[0])
for f in glob.glob('147_main/R-*-s[0-9].log'):
    _, setting, s = os.path.basename(f)[:-4].split('-'); s = int(s[1:]); p = parse(f)
    if 'adaround' in p:
        data.setdefault((setting, 'adaround(MSE)'), {})[s] = p['adaround']
keys = [('coco', 'COCO AP', 2), ('heval_ap', 'H_eval_AP', 2), ('lvis', 'LVIS AP', 4), ('apr', 'APr', 4), ('apc', 'APc', 4),
        ('apf', 'APf', 4), ('lvis_flip', 'LVIS_flip', 2), ('lvis_lost', 'LVIS_lost', 0), ('heval_flip', 'Heval_flip', 2),
        ('top1', 'Top1_flip', 2), ('lost', 'lost', 0), ('lcorr', 'L_CorrR', 2)]
low = {'lvis_flip', 'lvis_lost', 'heval_flip', 'top1', 'lost'}


def sz(setting, c):
    if setting == 'W8A8':
        return '12.08'
    suf = c.split('+')[1] if '+' in c else ''
    if 'R' in suf: return '6.24'
    if 'H' in suf: return '6.23'
    if 'G' in suf and 'P' in suf: return '6.16'
    if 'P' in suf: return '6.22'
    return '6.14'


rows = {
    'W8A8': [('naive', None), ('brecq', None), ('brecq+M', 'brecq')],
    'W4A8': [('naive', None), ('brecq', None), ('adaround(MSE)', 'brecq'), ('qdrop', 'brecq'), ('combined', 'brecq'),
             ('brecq+M', 'brecq'), ('brecq+P', 'brecq'), ('brecq+PM', 'brecq'), ('brecq+R', 'brecq'), ('brecq+H', 'brecq'),
             ('brecq+G', 'brecq'), ('brecq+GM', 'brecq'), ('brecq+GPM', 'brecq'), ('brecq+GPM', 'brecq+PM'),
             ('qdrop+PM', 'qdrop')],
    'W4A6': [('naive', None), ('brecq', None), ('qdrop', 'brecq'), ('combined', 'brecq'), ('brecq+M', 'brecq'),
             ('brecq+P', 'brecq'), ('brecq+PM', 'brecq'), ('brecq+G', 'brecq'), ('brecq+GM', 'brecq'),
             ('brecq+GPM', 'brecq'), ('brecq+GPM', 'brecq+PM'), ('qdrop+PM', 'qdrop')],
    'W4A5': [('naive', None), ('brecq', None), ('qdrop', 'brecq'), ('brecq+M', 'brecq'), ('brecq+P', 'brecq'),
             ('brecq+PM', 'brecq'), ('brecq+G', 'brecq'), ('brecq+GM', 'brecq'), ('brecq+GPM', 'brecq'),
             ('brecq+GPM', 'brecq+PM'), ('qdrop+PM', 'qdrop')],
}
for setting, rr in rows.items():
    print(f"\n#### {setting}\n")
    print("| 조건 | 크기(MiB) | n | " + " | ".join(k[1] for k in keys) + " |")
    print("|" + "---|" * (len(keys) + 3))
    for c, ref in rr:
        if (setting, c) not in data:
            continue
        d = data[(setting, c)]; ss = sorted(d); r = data.get((setting, ref), {}) if ref else {}
        cells = []
        for k, _, p in keys:
            v = [d[s][k] for s in ss if d[s].get(k) is not None]
            t = f"{st.mean(v):.{p}f}"
            if ref:
                com = [s for s in ss if s in r]
                t += f" ({sum((d[s][k] < r[s][k]) if k in low else (d[s][k] > r[s][k]) for s in com)}/{len(com)})"
            cells.append(t)
        name = c + (f" (vs {ref})" if ref and ref != 'brecq' else '')
        print(f"| {name} | {sz(setting, c)} | {len(ss)} | " + " | ".join(cells) + " |")
