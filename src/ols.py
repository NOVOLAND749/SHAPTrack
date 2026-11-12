"""Vectorized exact OLS pieces shared by mediation and interaction scans (Frisch-Waugh-Lovell).

Every scan model is `outcome ~ base design Z + a few extra columns`; residualizing on Z once lets all
metabolites be tested with matrix operations. Results equal statsmodels OLS (checked in tests/test_ols.py).
"""
import logging

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)

SINGULAR_TOL = 1e-8   # relative determinant below which a two-column extension is treated as not estimable


def transform_metabolites(mets, how):
    """log then z-score (or rank-based inverse normal). Constant columns are returned as NaN and logged."""
    X = mets.astype(float)
    if how == "log_zscore":
        if (X <= 0).any().any():
            raise ValueError("non-positive metabolite values; cannot log-transform")
        X = np.log(X)
        sd = X.std(ddof=1)
        out = (X - X.mean()) / sd.where(sd > 0)
    elif how == "rank_int":
        r = X.rank(method="average")
        out = pd.DataFrame(_norm_ppf((r - 0.375) / (len(X) + 0.25)), index=X.index, columns=X.columns)
        out.loc[:, X.nunique() <= 1] = np.nan
    else:
        raise ValueError(f"unknown metabolite transform {how}")
    const = out.columns[out.isna().all()]
    if len(const):
        log.info("metabolite transform %s: %d constant columns set to NaN (not tested)", how, len(const))
    return out


def _norm_ppf(p):
    from scipy.stats import norm
    return norm.ppf(p)


class Residualizer:
    """Projection off the column space of Z (n x q, includes the intercept)."""

    def __init__(self, Z):
        self.Q, _ = np.linalg.qr(np.asarray(Z, dtype=float))
        self.q = self.Q.shape[1]
        self.n = self.Q.shape[0]

    def __call__(self, A):
        A = np.asarray(A, dtype=float)
        return A - self.Q @ (self.Q.T @ A)


def coef_one_extra(r_y, R_x, df):
    """Model y ~ Z + x_j for each column j: coefficient, SE of x_j. r_y (n,), R_x (n, m) residualized on Z."""
    sxx = (R_x ** 2).sum(axis=0)
    sxy = R_x.T @ r_y
    beta = sxy / sxx
    rss = r_y @ r_y - beta * sxy
    se = np.sqrt(rss / df / sxx)
    return beta, se


def coef_two_extra(r_y, R_a, R_b, df):
    """Model y ~ Z + a_j + b_j for each j: coefficients and SEs of a_j and b_j (columns residualized on Z)."""
    g11, g22, g12 = (R_a ** 2).sum(0), (R_b ** 2).sum(0), (R_a * R_b).sum(0)
    h1, h2 = R_a.T @ r_y, R_b.T @ r_y
    det = g11 * g22 - g12 ** 2
    # Not estimable when a_j and b_j are (near-)collinear after residualization, e.g. a metabolite that only
    # varies among non-carriers; those columns are returned as NaN.
    ok = det > SINGULAR_TOL * g11 * g22
    det = np.where(ok, det, np.nan)
    i11, i22, i12 = g22 / det, g11 / det, -g12 / det
    b1 = i11 * h1 + i12 * h2
    b2 = i12 * h1 + i22 * h2
    rss = r_y @ r_y - (b1 * h1 + b2 * h2)
    s2 = rss / df
    return b1, np.sqrt(s2 * i11), b2, np.sqrt(s2 * i22)


def add_corrections(df, p_col, snp_col, bonferroni_n, alpha, method):
    """Bonferroni within SNP (alpha / bonferroni_n) and BH-FDR across every row of the analysis."""
    out = df.copy()
    out["bonferroni_threshold"] = alpha / bonferroni_n
    out["bonferroni_sig"] = out[p_col] < out["bonferroni_threshold"]
    ok = out[p_col].notna()
    out["fdr_q"] = np.nan
    out.loc[ok, "fdr_q"] = multipletests(out.loc[ok, p_col], method=method)[1]
    out["fdr_sig"] = out["fdr_q"] < alpha
    out["n_tests_snp"] = out.groupby(snp_col)[p_col].transform("count")
    return out
