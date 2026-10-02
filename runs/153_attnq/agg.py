"""runs/153 attention 양자화 보강: 수준(none=runs/147 기존, attn, attn_cls)별 조건 값과 짝 차이."""
import glob, os
exec(open('147_main/aggregate.py').read().split("keys = ")[0])
base = data                                   # (setting, cond) -> {seed: m}  (attention FP)
Q = {}                                         # (level, setting, cond) -> {seed: m}
for (st_, c), d in base.items():
    for s, m in d.items(): Q.setdefault(('none', st_, c), {})[s] = m
for f in glob.glob('153_attnq/[QP]-*-s[0-9].log'):
    _, setting, lv, s = os.path.basename(f)[:-4].split('-'); s = int(s[1:])
    lv = {'attn': 'attn', 'cls': 'attn_cls'}[lv]
    for c, m in parse(f).items(): Q.setdefault((lv, setting, c), {})[s] = m
K = [('coco', 'COCO', 1), ('lvis', 'LVIS', 100), ('apr', 'APr', 100), ('lvis_flip', 'LVIS_flip', 1), ('lvis_lost', 'LVIS_lost', 1), ('heval_flip', 'Heval_flip', 1)]
def g(lv, st_, c, s, k):
    m = Q.get((lv, st_, c), {}).get(s); return None if not m or m.get(k[0]) is None else m[k[0]] * k[2]
seeds = sorted({s for (lv, _, _), d in Q.items() if lv != 'none' for s in d})
for st_ in ('W4A8', 'W4A5'):
    for s in seeds:
        print(f"\n### {st_} seed{s}")
        print("| 수준 | 조건 | " + " | ".join(k[1] for k in K) + " |"); print("|" + "---|" * (len(K) + 2))
        for lv in ('none', 'attn', 'attn_cls'):
            for c in ('naive', 'brecq', 'brecq+PM', 'brecq+GPM'):
                v = [g(lv, st_, c, s, k) for k in K]
                if all(x is None for x in v) or (lv == 'none' and c == 'naive'): continue
                print(f"| {lv} | {c} | " + " | ".join('-' if x is None else (f'{x:.0f}' if k[0] == 'lvis_lost' else f'{x:.2f}') for x, k in zip(v, K)) + " |")
        print("\n| 수준 | 비교 | " + " | ".join(k[1] for k in K) + " |"); print("|" + "---|" * (len(K) + 2))
        for lv in ('none', 'attn', 'attn_cls'):
            for a, b in (('brecq+GPM', 'brecq'), ('brecq+PM', 'brecq'), ('brecq+GPM', 'brecq+PM')):
                v = [(g(lv, st_, a, s, k), g(lv, st_, b, s, k)) for k in K]
                if any(x is None or y is None for x, y in v): continue
                print(f"| {lv} | {a} − {b} | " + " | ".join(f'{x - y:+.0f}' if k[0] == 'lvis_lost' else f'{x - y:+.2f}' for (x, y), k in zip(v, K)) + " |")
