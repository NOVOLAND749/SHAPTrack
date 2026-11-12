"""Post hoc v3: cross-fitted metabolome LDL score and a genome-wide SNP -> score -> LDL-C mediation screen.

Per SNP (covariates C in every model, exact OLS via Frisch-Waugh-Lovell):
  total   LDL   ~ 1 + C + SNP          -> c
  path a  M_hat ~ 1 + C + SNP          -> a
  direct  LDL   ~ 1 + C + SNP + M_hat  -> c' (SNP) and b (M_hat);  c = c' + a*b
"""
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from statsmodels.stats.multitest import multipletests

from src import split
from src.ols import Residualizer

log = logging.getLogger(__name__)


def crossfit_score(mets_t, y, groups, n_folds, seed, alphas):
    """Out-of-fold ridge prediction of y from the metabolites (constant columns dropped)."""
    X = mets_t.loc[:, mets_t.notna().all()].values
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    used = []
    for tr, te in split.group_kfold(np.arange(len(y)), groups, n_folds, seed):
        model = RidgeCV(alphas=alphas).fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        used.append(float(model.alpha_))
    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    log.info("metabolome score: %d metabolites, out-of-fold R2 %.4f, alphas %s", X.shape[1], r2, used)
    return oof, {"n_metabolites": int(X.shape[1]), "r2_oof": float(r2), "alphas": used}


def scan(G, y, m, C, chunk=5000):
    """G: DataFrame of dosages (n, p); y, m: (n,); C: (n, q). One row per SNP."""
    n = len(y)
    R0 = Residualizer(np.hstack([np.ones((n, 1)), np.asarray(C, dtype=float)]))
    df0, df1 = n - R0.q - 1, n - R0.q - 2
    ry, rm = R0(np.asarray(y, dtype=float)), R0(np.asarray(m, dtype=float))
    syy, smm, smy = ry @ ry, rm @ rm, rm @ ry
    parts = []
    for i in range(0, G.shape[1], chunk):
        rg = R0(G.iloc[:, i:i + chunk].values.astype(float))
        sgg, sgy, sgm = (rg ** 2).sum(0), rg.T @ ry, rg.T @ rm
        c = sgy / sgg
        se_c = np.sqrt((syy - c * sgy) / df0 / sgg)
        a = sgm / sgg
        se_a = np.sqrt((smm - a * sgm) / df0 / sgg)
        det = sgg * smm - sgm ** 2
        i11, i22, i12 = smm / det, sgg / det, -sgm / det
        cp = i11 * sgy + i12 * smy
        b = i12 * sgy + i22 * smy
        s2 = (syy - (cp * sgy + b * smy)) / df1
        se_cp, se_b = np.sqrt(s2 * i11), np.sqrt(s2 * i22)
        ind = a * b
        sob = np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2)
        parts.append(pd.DataFrame({
            "c_total": c, "se_c": se_c, "p_c": 2 * stats.t.sf(np.abs(c / se_c), df0),
            "a": a, "se_a": se_a, "b": b, "se_b": se_b,
            "c_direct": cp, "se_c_direct": se_cp, "p_c_direct": 2 * stats.t.sf(np.abs(cp / se_cp), df1),
            "indirect": ind, "sobel_z": ind / sob, "sobel_p": 2 * stats.norm.sf(np.abs(ind / sob)),
            "prop_mediated": ind / c}, index=G.columns[i:i + chunk]))
    out = pd.concat(parts)
    out.index.name = "feature"
    return out


def classify(res, cfg):
    v3 = cfg["metabolome_v3"]
    out = res.copy()
    out["q_c"] = multipletests(out["p_c"], method="fdr_bh")[1]
    out["entered"] = out["q_c"] < v3["total_effect_fdr"]
    n_entered = int(out["entered"].sum())
    thr = cfg["downstream"]["alpha"] / max(n_entered, 1)
    sig = out["entered"] & (out["sobel_p"] < thr)
    same = np.sign(out["indirect"]) == np.sign(out["c_total"])
    unmask = (np.sign(out["indirect"]) != np.sign(out["c_direct"])) & (out["c_direct"].abs() > out["c_total"].abs())
    out["losing"] = sig & same & (out["prop_mediated"] >= v3["losing_min_prop_mediated"])
    for p in v3["losing_sensitivity_prop"]:
        out[f"losing_prop{p}"] = sig & same & (out["prop_mediated"] >= p)
    out["gaining"] = sig & unmask
    out["gaining_secondary"] = (out["sobel_p"] < cfg["downstream"]["alpha"] / len(out)) & unmask
    stats_ = {"n_snps": len(out), "n_entered": n_entered, "sobel_threshold": thr,
              "n_indirect_sig": int(sig.sum()), "n_losing": int(out["losing"].sum()),
              "n_gaining": int(out["gaining"].sum()), "n_gaining_secondary": int(out["gaining_secondary"].sum())}
    stats_.update({f"n_losing_prop{p}": int(out[f"losing_prop{p}"].sum()) for p in v3["losing_sensitivity_prop"]})
    return out, stats_
