"""크기-성능 Pareto 그림 (s-World, 확정 프로토콜). 사용: .venv/bin/python figures/make_pareto.py
데이터: runs/158_final, 154_protocol, 159_gpm_all (runs/158_final/agg.py의 D). W4A5는 붕괴한 seed 0을 모든 조건에서 뺀다.
크기: 로그의 conv weight 크기 + 8bit attention Linear 0.44 MiB(모든 조건 공통). x축은 BRECQ 대비 추가 크기(%)."""
import os, math, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'runs'))
exec(open('158_final/agg.py').read().split("K = [")[0])
CONV_MIB = {'brecq': 6.14, 'qdrop': 6.14, 'brecq+G': 6.14, 'brecq+M': 6.14, 'brecq+GM': 6.14, 'brecq+P': 6.22, 'brecq+PM': 6.22,
            'brecq+GP': 6.16, 'brecq+GPM': 6.16, 'brecq+R': 6.24, 'brecq+H': 6.23}
ATTN = 0.44
LABEL = {'brecq': 'BRECQ', 'qdrop': 'QDrop', 'brecq+G': 'G', 'brecq+P': 'P', 'brecq+M': 'M', 'brecq+GP': 'GP', 'brecq+GM': 'GM',
         'brecq+PM': 'PM', 'brecq+GPM': 'GPM (ours)', 'brecq+R': 'R (random)', 'brecq+H': 'H (HAWQ-like)'}
STYLE = {'brecq': ('k', 'o'), 'qdrop': ('0.5', 'o'), 'brecq+G': ('#7a5cff', 's'), 'brecq+P': ('#2f9e44', 's'), 'brecq+M': ('#e8590c', 's'),
         'brecq+GP': ('#5f3dc4', 'D'), 'brecq+GM': ('#9c36b5', 'D'), 'brecq+PM': ('#1c7ed6', 'D'), 'brecq+GPM': ('#d6336c', '*'),
         'brecq+R': ('#adb5bd', 'x'), 'brecq+H': ('#0ca678', 'x')}
SETS = ['W4A8', 'W4A6', 'W4A5']
def pts(st_, k, sc):
    out = {}
    for c in CONV_MIB:
        d = D.get((st_, c));
        if not d: continue
        ss = [s for s in sorted(d) if not (st_ == 'W4A5' and s == 0)]
        v = [d[s][k] * sc for s in ss if d[s].get(k) is not None]
        if len(v) < 2: continue
        ci = stats.t.ppf(0.975, len(v) - 1) * st.stdev(v) / math.sqrt(len(v))
        out[c] = (100 * (CONV_MIB[c] - CONV_MIB['brecq']) / (CONV_MIB['brecq'] + ATTN), st.mean(v), ci, len(v))
    return out
def frontier(p, higher):
    xs = sorted(p.items(), key=lambda kv: (kv[1][0], -kv[1][1] if higher else kv[1][1]))
    best, line = None, []
    for c, (x, y, _, _) in xs:
        if best is None or (y > best if higher else y < best):
            best = y; line.append((x, y))
    return line
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
for j, st_ in enumerate(SETS):
    for i, (k, sc, name, higher) in enumerate((('lvis', 100, 'LVIS AP', True), ('lvis_flip', 1, 'LVIS decision flip (%)', False))):
        ax = axes[i, j]; p = pts(st_, k, sc)
        fl = frontier(p, higher)
        if len(fl) > 1:
            fx, fy = zip(*fl); fx = list(fx) + [1.6]; fy = list(fy) + [fy[-1]]    # 오른쪽 끝까지 이어 계단을 분명히 보이게
            ax.step(fx, fy, where='post', color='#d6336c', alpha=0.35, lw=2.5, zorder=1)
        ax.set_xlim(-0.1, 1.6)
        for c, (x, y, ci, n) in p.items():
            col, mk = STYLE[c]
            ax.errorbar(x, y, yerr=ci, fmt=mk, color=col, ms=13 if mk == '*' else 7, capsize=3, zorder=3,
                        mec='k' if mk == '*' else col, label=LABEL[c])
            ax.annotate(LABEL[c].split(' ')[0], (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8, color=col)
        ax.grid(alpha=0.25)
        if i == 0: ax.set_title(f'{st_}' + ('  (seeds 1-5)' if st_ == 'W4A5' else '  (6 seeds)'), fontsize=11)
        if j == 0: ax.set_ylabel(name)
        if i == 1: ax.set_xlabel('extra model size vs BRECQ (%)')
h, l = [], []
for ax in axes.flat:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, loc='lower center', ncol=11, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('YOLO-World-S, confirmed protocol (all convs + attention/contrastive 8-bit, head-last input A16). '
             'Error bars: 95% CI over seeds. Shaded step: Pareto frontier.', fontsize=10)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = os.path.join('..', 'figures', 'pareto_size_vs_accuracy')
fig.savefig(out + '.png', dpi=160); fig.savefig(out + '.pdf')
for st_ in SETS:
    p = pts(st_, 'lvis', 100); q = pts(st_, 'lvis_flip', 1)
    print(st_, ' | '.join(f"{LABEL[c].split(' ')[0]} +{p[c][0]:.1f}% LVIS {p[c][1]:.2f} flip {q[c][1]:.2f}" for c in sorted(p, key=lambda c: p[c][0])))
print('저장:', out + '.png/.pdf')
