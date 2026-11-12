"""Manuscript Figures 2-3 from <results_dir> (v2 models, share-based labels). Reads saved results only.
Usage: python scripts/make_figures.py <results_dir> <out_dir> <results_dir>/v2/anchor_mediation/full.csv
"""
import sys, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = sys.argv[1]; OUT = sys.argv[2]; MED = sys.argv[3]
BLUE, ORANGE, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#a8a6a0', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
ann = pd.read_csv(f'{R}/glgc/annotated_all_snps.csv')
def gene(r):
    for c in ['overlapping_genes', 'upstream_gene', 'downstream_gene']:
        v = r.get(c)
        if isinstance(v, str) and v and not v.startswith('ENSG'):
            import re as _re
            return _re.sub(r'\(\d+bp\)', '', v.split(',')[0]).strip()
    return ''
rsid = dict(zip(ann.feature, ann.rsid)); gn = {r.feature: gene(r) for _, r in ann.iterrows()}
mnames = pd.read_csv(MED, usecols=['metabolite', 'CHEMICAL_NAME']).drop_duplicates('metabolite')
mname = dict(zip(mnames.metabolite.astype(str), mnames.CHEMICAL_NAME.fillna('')))
def label(f):
    f = str(f)
    if f in mname: return mname[f]
    r = rsid.get(f); g = gn.get(f, '')
    return f"{r if isinstance(r, str) else f}" + (f" ({g})" if g else '')

# per-split importances
G, J = [], []
for d in sorted(glob.glob(f'{R}/v2/splits/seed_*')):
    G.append(pd.read_csv(d + '/imp_genomics.csv').set_index('feature').mean_abs_shap)
    J.append(pd.read_csv(d + '/imp_joint.csv').set_index('feature').mean_abs_shap)
Gm = pd.concat(G, axis=1); Jm = pd.concat(J, axis=1)
jtype = pd.read_csv(sorted(glob.glob(f'{R}/v2/splits/seed_*'))[0] + '/imp_joint.csv').set_index('feature').type

# ---------- Figure 2 ----------
fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 2.2], hspace=0.28, wspace=1.15)
axA = fig.add_subplot(gs[0, :])
nc = json.load(open(f'{R}/nested_cv_decision.json'))
ks = sorted(int(k) for k in nc['inner_mean']); m = np.array([nc['inner_mean'][str(k)] for k in ks]); sd = np.array([nc['inner_sd'][str(k)] for k in ks])
axA.fill_between(ks, m - sd, m + sd, color=BLUE, alpha=0.15, linewidth=0)
axA.plot(ks, m, color=BLUE, lw=2, marker='o', ms=6, mfc='white', mec=BLUE, mew=2)
for k, v in zip(ks, m): axA.annotate(f'{v:.3f}', (k, v), textcoords='offset points', xytext=(0, 8), ha='center', color=INK2, fontsize=7)
axA.set_xticks(ks); axA.set_xticklabels([f'{k:,}' for k in ks]); axA.set_xlabel('Number of pre-selected SNPs (k)')
axA.set_ylabel('Inner-CV R²'); axA.set_ylim(0, max(m + sd) * 1.25); axA.grid(axis='y', color='#e6e5e1', lw=0.5); axA.set_axisbelow(True)
axA.text(-0.08, 1.05, 'A', transform=axA.transAxes, fontweight='bold', fontsize=10)

def bars(ax, M, top, colors, letter, xlabel):
    sel = M.loc[top]
    mean = sel.fillna(0).mean(axis=1); sdv = sel.fillna(0).std(axis=1)
    y = np.arange(len(top))[::-1]
    ax.barh(y, mean, xerr=sdv, color=colors, height=0.72, error_kw=dict(ecolor=INK2, lw=0.6, capsize=0))
    ax.set_yticks(y); ax.set_yticklabels([label(f) for f in top], fontsize=6.6)
    ax.set_xlabel(xlabel); ax.grid(axis='x', color='#e6e5e1', lw=0.5); ax.set_axisbelow(True)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight='bold', fontsize=10, ha='right')
gtop = list(Gm.fillna(0).mean(axis=1).sort_values(ascending=False).index[:20])
axB = fig.add_subplot(gs[1, 0]); bars(axB, Gm, gtop, [BLUE] * 20, 'B', 'Mean |SHAP|, genomics-only model')
jtop = list(Jm.fillna(0).mean(axis=1).sort_values(ascending=False).index[:20])
cols = [ORANGE if jtype.get(f) != 'metabolite' else BLUE for f in jtop]
axC = fig.add_subplot(gs[1, 1]); bars(axC, Jm, jtop, cols, 'C', 'Mean |SHAP|, joint model')
from matplotlib.patches import Patch
axC.legend(handles=[Patch(color=BLUE, label='Metabolite'), Patch(color=ORANGE, label='SNP')], frameon=False, loc='lower right', fontsize=7)
fig.savefig(f'{OUT}/figure2.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/figure2.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------- Figure 3 ----------
t = pd.read_csv(f'{R}/v6/v2/snp_table.csv'); a = t[(t.glgc_label == 'anchor') & t.eligible].copy()
# mean genomics importance over the splits in which the SNP was selected
a['imp_g'] = [Gm.loc[f].dropna().mean() if f in Gm.index else np.nan for f in a.feature]
a['imp_g'] = a.imp_g.clip(lower=1e-5)
a['ratio'] = (a.mean_S / (1 - a.mean_S)).clip(lower=0.005)
print('imp_g min', a.imp_g.min(), 'n ratio clipped', (a.mean_S / (1 - a.mean_S) < 0.005).sum(), 'n imp_g<=0', (a.imp_g<=0).sum())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1.7, 1], wspace=0.5))
colmap = {'losing': BLUE, 'gaining': ORANGE, 'none': GRAY}
for lab, z in [('none', 1), ('losing', 2), ('gaining', 3)]:
    s = a[a.v6_label == lab]
    ax1.scatter(s.imp_g, s.ratio, s=22, c=colmap[lab], edgecolors='white', linewidths=0.8, zorder=z,
                label={'none': 'Not classified', 'losing': 'Losing', 'gaining': 'Gaining'}[lab] + f' (n = {len(s)})')
ax1.set_xscale('log'); ax1.set_yscale('log')
for yv in (0.5, 1.0): ax1.axhline(yv, color=INK2, lw=0.7, ls='--', zorder=0)
import matplotlib.transforms as mtrans
tr = mtrans.blended_transform_factory(ax1.transAxes, ax1.transData)
ax1.text(0.01, 0.5, 'losing (0.5)', transform=tr, ha='left', va='top', fontsize=6.3, color=INK2)
ax1.text(0.01, 1.0, 'gaining (1)', transform=tr, ha='left', va='bottom', fontsize=6.3, color=INK2)
key = {'rs174592': 'FADS2', 'rs28601761': 'TRIB1', 'rs72658867': 'LDLR', 'rs597076': 'DOCK7', 'rs12446515': 'CETP',
       'rs7499892': 'CETP', 'rs71033586': 'LPA', 'rs7412': 'APOE', 'rs429358': 'APOE', 'rs4646272': 'SLC22A1'}
off = {'rs174592': (40, -30), 'rs28601761': (10, -45), 'rs597076': (-80, -25), 'rs72658867': (12, 0),
       'rs71033586': (-70, 18), 'rs4646272': (8, -3)}
arrow = dict(arrowstyle='-', color=INK2, lw=0.5, shrinkA=0, shrinkB=3)
pos = {r.rsid: (r.imp_g, r.ratio) for _, r in a.iterrows()}
for rs, o in off.items():
    if rs in pos:
        ax1.annotate(f"{key[rs]} {rs}", pos[rs], textcoords='offset points', xytext=o, fontsize=6.2, color=INK, arrowprops=arrow)
# one label for the two CETP-region SNPs
cx, cy = 0.011, 3.6
for rs in ('rs12446515', 'rs7499892'):
    ax1.annotate('', pos[rs], xytext=(cx, cy), textcoords='data', arrowprops=arrow)
ax1.text(cx, cy, 'CETP rs12446515, rs7499892', fontsize=6.2, color=INK, va='bottom')
ex, ey = 0.012, 2.0
for rs in ('rs429358', 'rs7412'):
    ax1.annotate('', pos[rs], xytext=(ex, ey), textcoords='data', arrowprops=arrow)
ax1.text(ex, ey, 'APOE rs429358, rs7412', fontsize=6.2, color=INK, va='bottom')
ax1.set_xlabel('Mean |SHAP|, genomics-only model (log)'); ax1.set_ylabel('Importance retention, joint / genomics (log)')
ax1.legend(frameon=False, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, handletextpad=0.2, columnspacing=1.0)
ax1.text(-0.12, 1.03, 'A', transform=ax1.transAxes, fontweight='bold', fontsize=10)
order = ['losing', 'none', 'gaining']; rng = np.random.default_rng(1)
for i, lab in enumerate(order):
    v = a[a.v6_label == lab].v3_prop_mediated.dropna().values
    ax2.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=16, c=colmap[lab], edgecolors='white', linewidths=0.6, zorder=2)
    ax2.hlines(np.median(v), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=3)
ax2.set_xticks(range(3)); ax2.set_xticklabels(['Losing', 'Not\nclassified', 'Gaining'])
ax2.set_ylabel('Proportion mediated (metabolome score)'); ax2.grid(axis='y', color='#e6e5e1', lw=0.5); ax2.set_axisbelow(True)
ax2.text(-0.22, 1.03, 'B', transform=ax2.transAxes, fontweight='bold', fontsize=10)
fig.savefig(f'{OUT}/figure3.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/figure3.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('genomics top20:', [label(f) for f in gtop]); print('joint top20:', [label(f) for f in jtop])
print('anchors plotted', len(a), a.v6_label.value_counts().to_dict(), 'imp_g NaN', a.imp_g.isna().sum())
