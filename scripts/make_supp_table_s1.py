"""Supplementary Table S1: importance dynamics, total effect and mediation for every eligible SNP
(79 established GLGC SNPs + 53 suggestive SNPs; v6 rule, v2 models).

Usage: python make_supp_table_s1.py <results_dir> <output csv>
"""
import sys
import numpy as np
import pandas as pd

RES, OUT = sys.argv[1:3]

st = pd.read_csv(f'{RES}/v6/v2/snp_table.csv')
st = st[st.eligible == True].copy()

cols = ['snp', 'metabolite', 'c_total', 'se_c_total', 'p_c_total', 'c_direct', 'indirect', 'sobel_p',
        'prop_mediated', 'bonferroni_sig', 'CHEMICAL_NAME', 'SUB_PATHWAY', 'glgc_pvalue_neg_log10_GC']
med = []
for path in (f'{RES}/v2/anchor_mediation/full.csv', f'{RES}/v6/candidate_mediation/full.csv'):
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
        med.append(chunk[chunk.snp.isin(st.feature)])
med = pd.concat(med).drop_duplicates(['snp', 'metabolite'])

# Definitions as in the manuscript: consistent = IE and c' same sign; suppressor = opposite signs and |c'| > |c|
med['consistent'] = np.sign(med.indirect) == np.sign(med.c_direct)
med['suppressor'] = (np.sign(med.indirect) != np.sign(med.c_direct)) & (med.c_direct.abs() > med.c_total.abs())

rows = []
for snp, g in med.groupby('snp'):
    top = g[g.consistent].sort_values('sobel_p').head(1)
    t = top.iloc[0] if len(top) else None
    rows.append({
        'feature': snp,
        'glgc_neg_log10_p': g.glgc_pvalue_neg_log10_GC.iloc[0],
        'c_total_mgdl': g.c_total.iloc[0], 'se_c_total': g.se_c_total.iloc[0], 'p_c_total': g.p_c_total.iloc[0],
        'n_metabolites_tested': len(g),
        'top_consistent_mediator': t.CHEMICAL_NAME if t is not None else '',
        'top_mediator_id': t.metabolite if t is not None else '',
        'top_mediator_sub_pathway': t.SUB_PATHWAY if t is not None else '',
        'top_indirect_mgdl': t.indirect if t is not None else np.nan,
        'top_prop_mediated': t.prop_mediated if t is not None else np.nan,
        'top_sobel_p': t.sobel_p if t is not None else np.nan,
        'n_bonf_consistent': int((g.bonferroni_sig & g.consistent).sum()),
        'n_bonf_suppressor': int((g.bonferroni_sig & g.suppressor).sum()),
        'n_bonf_all': int(g.bonferroni_sig.sum()),
    })
m = pd.DataFrame(rows)
assert len(m) == len(st), (len(m), len(st))

out = st.merge(m, on='feature', how='left')
out['glgc_category'] = out.set.map({'primary': 'established (p < 5e-8)', 'secondary': 'suggestive (5e-8 <= p < 0.05)'})
out['z_c_total'] = out.c_total_mgdl.abs() / out.se_c_total
out['retention'] = out.ratio_mean_S
out['class'] = out.v6_label

# order: established first; losing, gaining, unclassified; losing/none by retention ascending, gaining descending
out['_p'] = out.set.map({'primary': 0, 'secondary': 1})
out['_c'] = out['class'].map({'losing': 0, 'gaining': 1, 'none': 2})
out['_r'] = np.where(out['class'] == 'gaining', -out.retention, out.retention)
out = out.sort_values(['_p', '_c', '_r'])

final = out[['glgc_category', 'rsid', 'feature', 'gene', 'effect_allele', 'other_allele', 'eaf_sample', 'maf_sample',
             'imputation_rsq', 'glgc_neg_log10_p', 'n_selected', 'mean_S', 'retention', 'stability_losing',
             'stability_gaining', 'class', 'c_total_mgdl', 'se_c_total', 'z_c_total', 'p_c_total',
             'n_metabolites_tested', 'top_consistent_mediator', 'top_mediator_id', 'top_mediator_sub_pathway',
             'top_indirect_mgdl', 'top_prop_mediated', 'top_sobel_p', 'n_bonf_consistent', 'n_bonf_suppressor',
             'n_bonf_all']].rename(columns={
    'feature': 'variation_id', 'eaf_sample': 'eaf', 'maf_sample': 'maf', 'n_selected': 'splits_selected_of_20'})
final.to_csv(OUT, index=False, encoding='utf-8-sig', float_format='%.6g')
print(final.groupby(['glgc_category', 'class']).size())
print('rows', len(final))
