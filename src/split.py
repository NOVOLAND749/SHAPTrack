"""Step 1: seeded train / validation / test splits and outcome residualization fit on train only."""
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_seeds(cfg):
    s = cfg["splits"]
    return [s["seed_base"] + i for i in range(s["n_seeds"])]


@dataclass
class Split:
    """Positional indices. `train` = `fit` + `val`; `val` is used only for early stopping."""
    seed: int
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def train(self):
        return np.sort(np.concatenate([self.fit, self.val]))


def _holdout(idx, groups, frac, seed):
    """Split positions `idx` into (keep, holdout) with ~frac of samples held out, groups kept whole."""
    if groups is None:
        keep, hold = train_test_split(idx, test_size=frac, random_state=seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        k, h = next(gss.split(idx, groups=groups[idx]))
        keep, hold = idx[k], idx[h]
    return np.sort(keep), np.sort(hold)


def make_split(n, seed, test_size, val_fraction, groups=None):
    idx = np.arange(n)
    groups = None if groups is None else np.asarray(groups)
    train, test = _holdout(idx, groups, test_size, seed)
    fit, val = _holdout(train, groups, val_fraction, seed)
    return Split(seed, fit, val, test)


def group_kfold(idx, groups, n_folds, seed):
    """K folds over positions `idx`; whole groups are assigned to folds in a seeded random order."""
    idx = np.asarray(idx)
    g = np.asarray(groups)[idx] if groups is not None else idx
    uniq = np.unique(g)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = dict(zip(uniq[order], np.arange(len(uniq)) % n_folds))
    fold = np.array([fold_of[x] for x in g])
    return [(np.sort(idx[fold != f]), np.sort(idx[fold == f])) for f in range(n_folds)]


def residualize(y, covariates, train_idx):
    """OLS of y on 1 + covariates fitted on `train_idx` rows; residuals for all rows, z-scored with train stats."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(covariates, dtype=float)])
    beta, *_ = np.linalg.lstsq(X[train_idx], y[train_idx], rcond=None)
    r = y - X @ beta
    mu, sd = r[train_idx].mean(), r[train_idx].std(ddof=1)
    return (r - mu) / sd, {"beta": beta.tolist(), "resid_mean": float(mu), "resid_sd": float(sd)}
