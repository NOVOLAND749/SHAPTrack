"""Leakage tests. Run from the repo root: python tests/test_no_leakage.py (or pytest if installed)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline, prescreen, split, train  # noqa: E402
from src.data import Dataset, load_config  # noqa: E402


def small_cfg():
    cfg = copy.deepcopy(load_config())
    cfg["xgboost"].update(n_estimators=60, early_stopping_rounds=10)
    cfg["nested_cv"]["k_grid"] = [5, 10]
    return cfg


def synthetic(cfg, n=400, n_snp=100, n_met=20, seed=0):
    rng = np.random.default_rng(seed)
    ids = pd.Index([f"p{i}" for i in range(n)])
    snps = pd.DataFrame(rng.binomial(2, 0.3, (n, n_snp)).astype(np.float32), index=ids,
                        columns=[f"chr1:{1000 + i}:A:G_A" for i in range(n_snp)])
    mets = pd.DataFrame(rng.normal(size=(n, n_met)).astype(np.float32), index=ids,
                        columns=[str(100 + i) for i in range(n_met)])
    cov = pd.DataFrame(rng.normal(size=(n, len(cfg["columns"]["covariates"]))), index=ids,
                       columns=cfg["columns"]["covariates"])
    y = pd.Series(0.5 * snps.iloc[:, 0] + mets.iloc[:, 0] + rng.normal(size=n), index=ids)
    groups = pd.Series(ids, index=ids)
    groups.iloc[1:60:2] = groups.iloc[0:60:2].values   # 30 related pairs
    return Dataset(snps, mets, y, cov, groups, pd.DataFrame(index=ids))


def test_split_partitions_disjoint_and_complete():
    groups = np.repeat(np.arange(700), [2] * 100 + [1] * 600)   # 100 pairs + 600 singletons
    for seed in (1, 2, 3):
        sp = split.make_split(len(groups), seed, 0.4, 0.2, groups)
        parts = [set(sp.fit), set(sp.val), set(sp.test)]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert parts[0] | parts[1] | parts[2] == set(range(len(groups)))
        where = {}
        for name, idx in (("fit", sp.fit), ("val", sp.val), ("test", sp.test)):
            for g in groups[idx]:
                assert where.setdefault(g, name) == name, "a family spans two partitions"
        assert abs(len(sp.test) / len(groups) - 0.4) < 0.05


def test_group_kfold_disjoint_and_complete():
    groups = np.repeat(np.arange(700), [2] * 100 + [1] * 600)
    idx = np.arange(len(groups))
    folds = split.group_kfold(idx, groups, 3, seed=7)
    tests = [set(te) for _, te in folds]
    assert set().union(*tests) == set(idx) and sum(len(t) for t in tests) == len(idx)
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_residualization_uses_train_only():
    rng = np.random.default_rng(1)
    n = 500
    C = rng.normal(size=(n, 12))
    y = C @ rng.normal(size=12) + rng.normal(size=n)
    sp = split.make_split(n, 3, 0.4, 0.2)
    r1, info1 = split.residualize(y, C, sp.train)
    y2, C2 = y.copy(), C.copy()
    y2[sp.test] = 1e6 * rng.normal(size=len(sp.test))
    C2[sp.test] = 1e3
    r2, info2 = split.residualize(y2, C2, sp.train)
    assert np.allclose(info1["beta"], info2["beta"])
    assert np.allclose(r1[sp.train], r2[sp.train])
    X = np.column_stack([np.ones(n), C])
    beta, *_ = np.linalg.lstsq(X[sp.train], y[sp.train], rcond=None)
    assert np.allclose(beta, info1["beta"])
    assert abs(r1[sp.train].mean()) < 1e-10 and abs(r1[sp.train].std(ddof=1) - 1) < 1e-10


def test_prescreen_ignores_test_and_val_rows():
    cfg = small_cfg()
    data = synthetic(cfg)
    seed = cfg["splits"]["seed_base"]
    sp = split.make_split(data.n, seed, cfg["splits"]["test_size"], cfg["splits"]["val_fraction_of_train"],
                          data.groups.values)
    y, _ = split.residualize(data.y.values, data.covariates.values, sp.fit)
    planted = np.zeros(data.n, dtype=np.float32)
    planted[sp.test] = y[sp.test]              # perfect predictor in test rows, constant elsewhere
    data.snps["chr2:1:A:G_A"] = planted
    planted_val = np.zeros(data.n, dtype=np.float32)
    planted_val[sp.val] = y[sp.val]            # perfect predictor in early-stopping rows, constant elsewhere
    data.snps["chr2:2:A:G_A"] = planted_val
    selected, _ = prescreen.prescreen(data.snps, y, sp.fit, 10, cfg["prescreen"]["min_variance"])
    assert "chr2:1:A:G_A" not in selected and "chr2:2:A:G_A" not in selected
    # Same check through the full per-split pipeline.
    _, dyn = pipeline.run_split(data, seed, cfg, k=10)
    assert "chr2:1:A:G_A" not in dyn.index and "chr2:2:A:G_A" not in dyn.index
    # Control: selecting on all rows would pick the test-row predictor first.
    all_rows, _ = prescreen.prescreen(data.snps, y, np.arange(data.n), 10, cfg["prescreen"]["min_variance"])
    assert all_rows[0] == "chr2:1:A:G_A"


def _record_fits():
    calls, original = [], xgb.XGBRegressor.fit

    def recording_fit(self, X, y, *args, **kwargs):
        eval_set = kwargs.get("eval_set")
        calls.append({"train": set(X.index), "eval": set(eval_set[0][0].index) if eval_set else set()})
        return original(self, X, y, *args, **kwargs)
    return calls, original, recording_fit


def test_early_stopping_set_disjoint_from_test():
    cfg = small_cfg()
    data = synthetic(cfg)
    seed = cfg["splits"]["seed_base"]
    sp = split.make_split(data.n, seed, cfg["splits"]["test_size"], cfg["splits"]["val_fraction_of_train"],
                          data.groups.values)
    test_ids, val_ids = set(data.y.index[sp.test]), set(data.y.index[sp.val])
    calls, original, recording_fit = _record_fits()
    xgb.XGBRegressor.fit = recording_fit
    try:
        pipeline.run_split(data, seed, cfg, k=10)
    finally:
        xgb.XGBRegressor.fit = original
    assert len(calls) == 4                      # early-stopping fit + refit, two models
    for c in calls:
        assert not c["train"] & test_ids
        assert not c["eval"] & test_ids
    assert [c["eval"] == val_ids for c in calls if c["eval"]] == [True, True]


def test_nested_cv_scoring_rows_never_used_for_fitting():
    cfg = small_cfg()
    data = synthetic(cfg, n=300)
    seen, original = [], train.fit_model

    def checking_fit_model(X, y, sp, *args, **kwargs):
        seen.append(sp)
        return original(X, y, sp, *args, **kwargs)
    train.fit_model = checking_fit_model
    try:
        nested, _ = pipeline.run_nested_cv(data, cfg)
    finally:
        train.fit_model = original
    nc = cfg["nested_cv"]
    assert len(seen) == nc["outer_folds"] * (nc["inner_folds"] * len(nc["k_grid"]) + 1)
    for sp in seen:
        assert not set(sp.test) & (set(sp.fit) | set(sp.val))
    outer_tests = [set(te) for _, te in split.group_kfold(np.arange(data.n), data.groups.values,
                                                          nc["outer_folds"], nc["seed"])]
    inner_scored = [set(sp.test) for sp in seen if set(sp.test) not in outer_tests]
    for o, ot in enumerate(outer_tests):
        block = inner_scored[o * len(nc["k_grid"]) * nc["inner_folds"]:(o + 1) * len(nc["k_grid"]) * nc["inner_folds"]]
        assert all(not (s & ot) for s in block), "inner scoring touched the outer test fold"


def test_v2_cv_early_stopping_never_touches_test():
    cfg = small_cfg()
    cfg["posthoc_v2"]["early_stopping"].update(n_estimators_cap=60, rounds=10)
    data = synthetic(cfg)
    seed = cfg["splits"]["seed_base"]
    sp = split.make_split(data.n, seed, cfg["splits"]["test_size"], cfg["splits"]["val_fraction_of_train"],
                          data.groups.values)
    y, _ = split.residualize(data.y.values, data.covariates.values, sp.train)
    planted = np.zeros(data.n, dtype=np.float32)
    planted[sp.test] = y[sp.test]
    data.snps["chr2:1:A:G_A"] = planted
    test_ids = set(data.y.index[sp.test])
    calls, original, recording_fit = _record_fits()
    xgb.XGBRegressor.fit = recording_fit
    try:
        _, dyn = pipeline.run_split_v2(data, seed, cfg, k=10)
    finally:
        xgb.XGBRegressor.fit = original
    folds = cfg["posthoc_v2"]["early_stopping"]["cv_folds"]
    assert len(calls) == 2 * folds + 2          # two models per CV fold + two final fits
    for c in calls:
        assert not c["train"] & test_ids and not c["eval"] & test_ids
    assert "chr2:1:A:G_A" not in dyn.index


def run_all():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
