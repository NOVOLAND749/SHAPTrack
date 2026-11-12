"""Step 2a: SNP pre-selection by F-test, computed on training rows only."""
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

log = logging.getLogger(__name__)


def rank_snps(X_train, y_train, min_variance):
    """Rank SNP columns of `X_train` by f_regression F (descending) after a variance filter.

    Returns a DataFrame indexed by SNP with variance, F, p and rank; SNPs failing the filter are absent.
    """
    var = X_train.var(axis=0, ddof=0)
    keep = var > min_variance
    Xv = X_train.loc[:, keep]
    F, p = f_regression(Xv.values, np.asarray(y_train, dtype=float))
    F = np.nan_to_num(F, nan=-np.inf)
    out = pd.DataFrame({"variance": var[keep].values, "F": F, "p": p}, index=Xv.columns)
    out = out.iloc[np.argsort(-out["F"].values, kind="stable")]
    out["rank"] = np.arange(1, len(out) + 1)
    log.info("prescreen: %d SNPs, %d pass variance > %g on %d training rows",
             X_train.shape[1], int(keep.sum()), min_variance, X_train.shape[0])
    return out


def prescreen(snps, y, train_idx, k, min_variance):
    """Top-k SNP names by F-test using only rows `train_idx` of `snps` and `y` (positional)."""
    ranking = rank_snps(snps.iloc[train_idx], np.asarray(y)[train_idx], min_variance)
    selected = ranking.index[:k].tolist()
    log.info("prescreen: kept top %d of %d", len(selected), len(ranking))
    return selected, ranking
