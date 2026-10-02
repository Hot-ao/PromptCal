"""runs/154 (A16 + attn_cls) 결과 + 비교용 runs/151 m(이전 프로토콜) 결과."""
import glob, os, sys
exec(open('147_main/aggregate.py').read().split("data = {}")[0])
R = {}   # (model, setting, cond) -> {seed: m}
for f in glob.glob('154_protocol/*-s[0-9].log'):
    parts = os.path.basename(f)[:-4].split('-'); mdl, st_, s = parts[0], parts[1], int(parts[-1][1:])
    for c, m in parse(f).items(): R.setdefault((mdl, st_, c), {})[s] = m
OLD = {}
for f in glob.glob('151_gen/M*-W4A*-s[0-9].log'):
    st_ = os.path.basename(f).split('-')[1]; s = int(f[-5])
    for c, m in parse(f).items(): OLD.setdefault((st_, c), {})[s] = m
K = [('coco','COCO',1),('lvis','LVIS',100),('apr','APr',100),('lvis_flip','LVIS_flip',1),('lvis_lost','lost',1),('heval_flip','Heval_flip',1)]
mdl = sys.argv[1] if len(sys.argv) > 1 else 'm'
for st_ in ('W8A8', 'W4A8', 'W4A5'):
    for s in (0, 1):
        rows = [(c, R[(mdl, st_, c)][s]) for c in ('naive','brecq','brecq+M','brecq+P','brecq+G','brecq+GP','brecq+PM','brecq+GPM') if s in R.get((mdl, st_, c), {})]
        if not rows: continue
        print(f"\n### {mdl} {st_} seed{s} (A16 + attn_cls)")
        print("| 조건 | " + " | ".join(k[1] for k in K) + " | Δ LVIS vs brecq |"); print("|" + "---|" * (len(K) + 2))
        b = dict(rows).get('brecq')
        for c, m in rows:
            d = f"{(m['lvis']-b['lvis'])*100:+.2f}" if b and c not in ('naive','brecq') else ''
            print(f"| {c} | " + " | ".join(f"{m[k[0]]*k[2]:.0f}" if k[0]=='lvis_lost' else f"{m[k[0]]*k[2]:.2f}" for k in K) + f" | {d} |")
        if mdl == 'm':
            old = [(c, OLD[(st_, c)][s]) for c in ('naive','brecq','brecq+P','brecq+M','brecq+PM','brecq+GPM') if s in OLD.get((st_, c), {})]
            if old:
                print(f"\n_이전 프로토콜(runs/151, A8·attention FP) {st_} seed{s}:_ " + ", ".join(f"{c} COCO {m['coco']:.2f} / LVIS {m['lvis']*100:.2f}" for c, m in old))
