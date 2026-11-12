"""Step 6a: SNP -> metabolite -> LDL-C mediation scan (Baron-Kenny paths + Sobel), covariates in every model.

Models, for one SNP and every metabolite M (all participants, raw medication-adjusted LDL-C):
  total   LDL ~ 1 + C + SNP                 -> c
  path a  M   ~ 1 + C + SNP                 -> a
  path b  LDL ~ 1 + C + SNP + M             -> b (and direct effect c' = c - a*b, exact for OLS)
  indirect = a*b, Sobel SE = sqrt(b² SE_a² + a² SE_b²), two-sided normal p.
"""
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.ols import Residualizer, add_corrections, coef_one_extra

log = logging.getLogger(__name__)


def scan(dosage, y, C, M):
    """dosage (n,), y (n,), C (n, p) covariates, M (n, m) transformed metabolites without NaN columns."""
    n = len(y)
    ones = np.ones((n, 1))
    R0 = Residualizer(np.hstack([ones, C]))
    R1 = Residualizer(np.hstack([ones, dosage[:, None], C]))
    df0, df1 = n - R0.q - 1, n - R1.q - 1
    r_snp, r_y0 = R0(dosage), R0(y)
    c, se_c = coef_one_extra(r_y0, r_snp[:, None], df0)
    RM0 = R0(M)
    sxx = r_snp @ r_snp
    sxy = RM0.T @ r_snp
    a = sxy / sxx
    se_a = np.sqrt(((RM0 ** 2).sum(0) - a * sxy) / df0 / sxx)
    b, se_b = coef_one_extra(R1(y), R1(M), df1)
    indirect = a * b
    sobel_se = np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2)
    z = indirect / sobel_se
    return pd.DataFrame({
        "n": n, "a": a, "se_a": se_a, "p_a": 2 * stats.t.sf(np.abs(a / se_a), df0),
        "b": b, "se_b": se_b, "p_b": 2 * stats.t.sf(np.abs(b / se_b), df1),
        "c_total": c[0], "se_c_total": se_c[0], "p_c_total": 2 * stats.t.sf(abs(c[0] / se_c[0]), df0),
        "c_direct": c[0] - indirect, "indirect": indirect, "sobel_se": sobel_se, "sobel_z": z,
        "sobel_p": 2 * stats.norm.sf(np.abs(z)), "prop_mediated": indirect / c[0],
    })


def run(snps, snp_dosages, y, covariates, mets_t, cfg):
    """snps: feature names; mets_t: transformed metabolites (NaN columns are skipped and logged)."""
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
    full = add_corrections(full, "sobel_p", "snp", ds["bonferroni_per_snp"], ds["alpha"], ds["fdr_method"])
    log.info("mediation: %d SNPs x %d metabolites = %d tests; %d metabolites not tested (constant): %s", len(snps),
             len(testable), len(full), len(skipped), list(skipped))
    return full, list(skipped)


def top_per_snp(full, n_top):
    return full.sort_values("sobel_p").groupby("snp", sort=False).head(n_top).sort_values(["snp", "sobel_p"])


def suppression_flags(full):
    """Suppression (inconsistent mediation) among Bonferroni hits: indirect and direct effects of opposite sign,
    and the direct effect larger in magnitude than the total effect."""
    out = full.copy()
    out["suppression"] = (out["bonferroni_sig"] & (np.sign(out["indirect"]) != np.sign(out["c_direct"]))
                          & (out["c_direct"].abs() > out["c_total"].abs()))
    return out


def suppression_per_snp(full):
    return full.groupby("snp").agg(rsid=("rsid", "first"), gene=("overlapping_genes", "first"),
                                   glgc_label=("glgc_label", "first"), c_total=("c_total", "first"),
                                   p_c_total=("p_c_total", "first"), n_hits=("bonferroni_sig", "sum"),
                                   n_suppression=("suppression", "sum"), min_sobel_p=("sobel_p", "min"))
