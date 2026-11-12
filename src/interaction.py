"""Step 6b: SNP x metabolite interaction scan on LDL-C, covariates in every model, plus sensitivity refits.

Scan, for one SNP and every metabolite M (all participants, raw medication-adjusted LDL-C):
  LDL ~ 1 + SNP + C + M + SNP:M           -> beta3 (SNP:M), t-test p
Sensitivity for hits (Keller 2014): add every covariate x SNP and covariate x M term and refit; also HC3 robust SE and
an HC3 refit without the 1% tails of M (added 2026-10-02 after a low-frequency hit proved tail-driven).
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.ols import Residualizer, add_corrections, coef_two_extra

log = logging.getLogger(__name__)


def scan(dosage, y, C, M):
    n = len(y)
    R1 = Residualizer(np.hstack([np.ones((n, 1)), dosage[:, None], C]))
    df = n - R1.q - 2
    bM, seM, b3, se3 = coef_two_extra(R1(y), R1(M), R1(M * dosage[:, None]), df)
    return pd.DataFrame({"n": n, "beta_metabolite": bM, "se_metabolite": seM, "beta3": b3, "se3": se3,
                         "t3": b3 / se3, "p": 2 * stats.t.sf(np.abs(b3 / se3), df)})


def run(snps, snp_dosages, y, covariates, mets_t, cfg):
    ds = cfg["downstream"]
    testable = mets_t.columns[mets_t.notna().all()]
    skipped = mets_t.columns.difference(testable)
    C, M = covariates.values.astype(float), mets_t[testable].values
    parts = []
    for s in snps:
        res = scan(snp_dosages[s].values.astype(float), y.values.astype(float), C, M)
        res.insert(0, "metabolite", testable)
        res.insert(0, "snp", s)
        parts.append(res)
    full = pd.concat(parts, ignore_index=True)
    full["estimable"] = full["p"].notna()
    full = add_corrections(full, "p", "snp", ds["bonferroni_per_snp"], ds["alpha"], ds["fdr_method"])
    log.info("interaction: %d SNPs x %d metabolites = %d rows; %d not estimable (collinear); %d metabolites not "
             "tested (constant)", len(snps), len(testable), len(full), int((~full["estimable"]).sum()), len(skipped))
    return full, list(skipped)


def sensitivity(hits, snp_dosages, y, covariates, mets_t):
    """Refit each hit with covariate x SNP and covariate x metabolite terms; p of SNP:M before and after."""
    rows = []
    for _, h in hits.iterrows():
        g = snp_dosages[h["snp"]].values.astype(float)
        m = mets_t[h["metabolite"]].values.astype(float)
        C = covariates.values.astype(float)
        base = np.column_stack([np.ones(len(g)), g, m, g * m, C])
        extra = np.column_stack([C * g[:, None], C * m[:, None]])
        fit0 = sm.OLS(y.values, base).fit()
        fit1 = sm.OLS(y.values, np.hstack([base, extra])).fit()
        hc3 = sm.OLS(y.values, base).fit(cov_type="HC3")
        trim = (m > np.quantile(m, 0.01)) & (m < np.quantile(m, 0.99))     # drop the 1% tails of the metabolite
        if trim.sum() >= 100 and np.ptp(m[trim]) > 0 and np.ptp(g[trim]) > 0:
            fit2 = sm.OLS(y.values[trim], base[trim]).fit(cov_type="HC3")
            b_trim, p_trim = fit2.params[3], fit2.pvalues[3]
        else:                    # heavily tied metabolite (e.g. a drug absent in most participants): not estimable
            b_trim, p_trim = np.nan, np.nan
        rows.append({"snp": h["snp"], "metabolite": h["metabolite"], "beta3": fit0.params[3], "p": fit0.pvalues[3],
                     "beta3_adj_cov_interactions": fit1.params[3], "p_adj_cov_interactions": fit1.pvalues[3],
                     "p_hc3": hc3.pvalues[3], "beta3_trim1pct": b_trim, "p_trim1pct_hc3": p_trim})
    return pd.DataFrame(rows)


def top_per_snp(full, n_top):
    return full.sort_values("p").groupby("snp", sort=False).head(n_top).sort_values(["snp", "p"])
