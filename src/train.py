"""Step 2b: XGBoost with early stopping on the validation slice, then refit on the full training portion."""
import logging

import xgboost as xgb
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)

_NOT_XGB_ARGS = {"early_stopping_rounds", "refit_on_full_train_with_best_iteration"}


def xgb_params(cfg, seed, n_jobs=None):
    params = {k: v for k, v in cfg["xgboost"].items() if k not in _NOT_XGB_ARGS}
    params["random_state"] = seed
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    return params


def fit_model(X, y, split, cfg, seed, n_jobs=None):
    """Fit on split.fit with early stopping on split.val; optionally refit on split.train.

    X: DataFrame (all rows), y: array (all rows). Test rows are only used for scoring.
    """
    for a, b in (("fit", "val"), ("fit", "test"), ("val", "test")):
        if len(set(getattr(split, a)) & set(getattr(split, b))):
            raise ValueError(f"{a} and {b} rows overlap")
    params = xgb_params(cfg, seed, n_jobs)
    X_fit, X_val = X.iloc[split.fit], X.iloc[split.val]
    y_fit, y_val = y[split.fit], y[split.val]
    es = xgb.XGBRegressor(**params, early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"])
    es.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best = int(es.best_iteration)
    info = {"seed": seed, "n_features": X.shape[1], "best_iteration": best,
            "r2_val": float(r2_score(y_val, es.predict(X_val)))}
    if cfg["xgboost"]["refit_on_full_train_with_best_iteration"]:
        model = xgb.XGBRegressor(**{**params, "n_estimators": best + 1})
        model.fit(X.iloc[split.train], y[split.train], verbose=False)
    else:
        model = es
    info["r2_train"] = float(r2_score(y[split.train], model.predict(X.iloc[split.train])))
    info["r2_test"] = float(r2_score(y[split.test], model.predict(X.iloc[split.test])))
    log.info("xgb seed %d, %d features: best_iteration %d, R2 val %.4f train %.4f test %.4f",
             seed, X.shape[1], best, info["r2_val"], info["r2_train"], info["r2_test"])
    return model, info


def fit_early_stopping(X_tr, y_tr, X_va, y_va, cfg, seed, n_jobs=None):
    """CV fold of the main models: early stopping on the held-out fold (main_models cap and patience)."""
    es = cfg["main_models"]["early_stopping"]
    params = {**xgb_params(cfg, seed, n_jobs), "n_estimators": es["n_estimators_cap"]}
    m = xgb.XGBRegressor(**params, early_stopping_rounds=es["rounds"])
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return {"best_iteration": int(m.best_iteration), "r2_val": float(r2_score(y_va, m.predict(X_va)))}


def fit_fixed(X, y, train_idx, test_idx, n_trees, cfg, seed, n_jobs=None):
    """Final main model: fixed number of trees on the whole training portion, no early stopping."""
    if set(train_idx) & set(test_idx):
        raise ValueError("train and test rows overlap")
    model = xgb.XGBRegressor(**{**xgb_params(cfg, seed, n_jobs), "n_estimators": n_trees})
    model.fit(X.iloc[train_idx], y[train_idx], verbose=False)
    info = {"seed": seed, "n_features": X.shape[1], "n_trees": n_trees, "best_iteration": n_trees - 1,
            "r2_train": float(r2_score(y[train_idx], model.predict(X.iloc[train_idx]))),
            "r2_test": float(r2_score(y[test_idx], model.predict(X.iloc[test_idx])))}
    return model, info
