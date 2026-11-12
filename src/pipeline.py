"""Orchestration of one split (Steps 1-3) and of nested CV for k (Step 7). Used by scripts/ and tests/."""
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import dynamics, explain, prescreen, split, train

log = logging.getLogger(__name__)


def _groups(data, cfg):
    return data.groups.values if cfg["splits"]["group_related"] else None


def run_split(data, seed, cfg, k, out_dir=None, n_jobs=None, save_shap=False):
    t0 = time.time()
    s = cfg["splits"]
    sp = split.make_split(data.n, seed, s["test_size"], s["val_fraction_of_train"], _groups(data, cfg))
    log.info("seed %d: fit %d, val %d, test %d", seed, len(sp.fit), len(sp.val), len(sp.test))

    # Residualization and pre-selection use fit rows only, so val stays an honest early-stopping set.
    y, resid = split.residualize(data.y.values, data.covariates.values, sp.fit)
    selected, ranking = prescreen.prescreen(data.snps, y, sp.fit, k, cfg["prescreen"]["min_variance"])

    X_gen = data.snps[selected]
    X_joint = pd.concat([X_gen, data.metabolites], axis=1)
    fp = cfg["shap"]["feature_perturbation"]
    results = {}
    for name, X in (("genomics", X_gen), ("joint", X_joint)):
        t = time.time()
        model, info = train.fit_model(X, y, sp, cfg, seed, n_jobs)
        info["fit_seconds"] = round(time.time() - t, 1)
        t = time.time()
        sv = explain.shap_values(model, X.iloc[sp.test], fp)
        info["shap_seconds"] = round(time.time() - t, 1)
        imp = explain.mean_abs_shap(sv, X.columns)
        results[name] = {"info": info, "imp": imp, "shap": sv}
        log.info("seed %d %s: fit %.0fs, SHAP %.0fs", seed, name, info["fit_seconds"], info["shap_seconds"])

    ranks = dynamics.split_ranks(results["genomics"]["imp"], results["joint"]["imp"], selected,
                                 cfg["dynamics"]["rank_method"])
    dyn, dyn_stats = dynamics.add_labels(ranks, cfg)
    log.info("seed %d: %d gaining, %d losing; sum(delta_rank) = %.1f; zero-SHAP SNPs genomics %d, joint %d",
             seed, dyn_stats["n_gaining"], dyn_stats["n_losing"], dyn_stats["sum_delta_rank"],
             dyn_stats["n_zero_shap_genomics"], dyn_stats["n_zero_shap_joint"])

    summary = {"seed": seed, "k": k, "n_fit": len(sp.fit), "n_val": len(sp.val), "n_test": len(sp.test),
               "group_related": s["group_related"], "residualization": resid,
               "genomics": results["genomics"]["info"], "joint": results["joint"]["info"],
               "dynamics": dyn_stats, "seconds": round(time.time() - t0, 1)}
    if out_dir is not None:
        _write_split(out_dir, data, sp, selected, ranking, k, results, {"genomics": X_gen, "joint": X_joint},
                     dyn, summary, cfg, save_shap)
    return summary, dyn


def _write_split(out_dir, data, sp, selected, ranking, k, results, X, dyn, summary, cfg, save_shap):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = data.y.index
    part = pd.Series("fit", index=ids)
    part.iloc[sp.val], part.iloc[sp.test] = "val", "test"
    part.rename("partition").rename_axis("id").to_csv(out / "partition.csv")
    (out / "selected_snps.txt").write_text("\n".join(selected) + "\n")
    ranking.head(k).rename_axis("feature").to_csv(out / "prescreen_topk.csv")
    snp_prefix = cfg["columns"]["snp_prefix"]
    explain.importance_table(results["genomics"]["imp"], snp_prefix).to_csv(out / "imp_genomics.csv", index=False)
    explain.importance_table(results["joint"]["imp"], snp_prefix).to_csv(out / "imp_joint.csv", index=False)
    dyn.to_csv(out / "dynamics.csv")
    if save_shap:
        for name in ("genomics", "joint"):
            np.savez_compressed(out / f"shap_{name}.npz", shap=results[name]["shap"].astype(np.float32),
                                features=np.array(X[name].columns), test_ids=np.array(ids[sp.test]))
    # r2.json last: its presence marks the split as complete.
    (out / "r2.json").write_text(json.dumps(summary, indent=2))


def cv_tree_counts(data, sp, cfg, k, seed, n_jobs=None):
    """Cross-validated early stopping (main models): per group-fold of the training portion, residualize and pre-select on the fold's training
    rows, fit both models with early stopping on the held-out fold. Returns best_iteration and val R² per fold."""
    es = cfg["main_models"]["early_stopping"]
    folds = split.group_kfold(sp.train, _groups(data, cfg), es["cv_folds"], seed)
    test = set(sp.test)
    out = {"genomics": [], "joint": []}
    for tr, va in folds:
        if test & (set(tr) | set(va)):
            raise ValueError("CV fold overlaps test rows")
        y, _ = split.residualize(data.y.values, data.covariates.values, tr)
        sel = prescreen.rank_snps(data.snps.iloc[tr], y[tr], cfg["prescreen"]["min_variance"]).index[:k].tolist()
        for name, X in (("genomics", data.snps[sel]), ("joint", pd.concat([data.snps[sel], data.metabolites], axis=1))):
            out[name].append(train.fit_early_stopping(X.iloc[tr], y[tr], X.iloc[va], y[va], cfg, seed, n_jobs))
    return out


def run_split_main(data, seed, cfg, k, out_dir=None, n_jobs=None, save_shap=False):
    """Main models: CV early stopping, final model on the whole training portion, SHAP-ratio labels."""
    t0 = time.time()
    s = cfg["splits"]
    sp = split.make_split(data.n, seed, s["test_size"], s["val_fraction_of_train"], _groups(data, cfg))
    log.info("main seed %d: train %d (CV early stopping), test %d", seed, len(sp.train), len(sp.test))
    folds = cv_tree_counts(data, sp, cfg, k, seed, n_jobs)

    y, resid = split.residualize(data.y.values, data.covariates.values, sp.train)
    selected, ranking = prescreen.prescreen(data.snps, y, sp.train, k, cfg["prescreen"]["min_variance"])
    X = {"genomics": data.snps[selected]}
    X["joint"] = pd.concat([X["genomics"], data.metabolites], axis=1)
    fp = cfg["shap"]["feature_perturbation"]
    results = {}
    for name in ("genomics", "joint"):
        t = time.time()
        n_trees = int(round(np.mean([f["best_iteration"] for f in folds[name]]))) + 1
        model, info = train.fit_fixed(X[name], y, sp.train, sp.test, n_trees, cfg, seed, n_jobs)
        info.update({"fold_best_iteration": [f["best_iteration"] for f in folds[name]],
                     "fold_r2_val": [f["r2_val"] for f in folds[name]], "fit_seconds": round(time.time() - t, 1),
                     "r2_val": float(np.mean([f["r2_val"] for f in folds[name]]))})   # CV mean, before refit
        sv = explain.shap_values(model, X[name].iloc[sp.test], fp)
        results[name] = {"info": info, "imp": explain.mean_abs_shap(sv, X[name].columns), "shap": sv}
        log.info("main seed %d %s: folds best_iteration %s -> %d trees, R2 train %.4f test %.4f", seed, name,
                 info["fold_best_iteration"], n_trees, info["r2_train"], info["r2_test"])

    ranks = dynamics.split_ranks(results["genomics"]["imp"], results["joint"]["imp"], selected,
                                 cfg["dynamics"]["rank_method"])
    dyn, dyn_stats = dynamics.add_labels_ratio(ranks, cfg)
    log.info("main seed %d: %d gaining, %d losing; median SHAP ratio %.3f", seed, dyn_stats["n_gaining"],
             dyn_stats["n_losing"], dyn_stats["median_ratio"])
    summary = {"version": "main", "seed": seed, "k": k, "n_train": len(sp.train), "n_test": len(sp.test),
               "n_fit": len(sp.fit), "n_val": len(sp.val), "group_related": s["group_related"],
               "residualization": resid, "genomics": results["genomics"]["info"], "joint": results["joint"]["info"],
               "dynamics": dyn_stats, "seconds": round(time.time() - t0, 1)}
    if out_dir is not None:
        _write_split(out_dir, data, sp, selected, ranking, k, results, X, dyn, summary, cfg, save_shap)
    return summary, dyn


def load_splits(splits_dir, seeds):
    """Read dynamics.csv and r2.json of completed splits; fail if any seed is missing."""
    ranks, r2 = {}, {}
    for s in seeds:
        d = Path(splits_dir) / f"seed_{s}"
        if not (d / "r2.json").exists():
            raise FileNotFoundError(f"split {s} incomplete: {d / 'r2.json'} missing")
        ranks[s] = pd.read_csv(d / "dynamics.csv", index_col="feature")
        r2[s] = json.loads((d / "r2.json").read_text())
    return ranks, r2


def write_aggregate(ranks, r2, cfg, out_dir):
    """Step 4 outputs: dynamics_aggregate, gaining, losing, sensitivity_grid, r2_per_split, r2_summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main = any(r.get("version") == "main" for r in r2.values())
    dy, n_seeds = (cfg["main_models"]["labels"] if main else cfg["dynamics"]), len(ranks)
    agg = dynamics.aggregate(ranks, n_seeds, dy["min_selection_fraction"], dy["stability_threshold"])
    agg.to_csv(out / "dynamics_aggregate.csv")
    for lab in ("gaining", "losing"):
        agg[agg["final_label"] == lab].to_csv(out / f"{lab}.csv")
    counts = dynamics.stage_counts(agg)
    counts["per_split_gaining"] = {s: int((d["label"] == "gaining").sum()) for s, d in ranks.items()}
    counts["per_split_losing"] = {s: int((d["label"] == "losing").sum()) for s, d in ranks.items()}
    (out / "stage_counts.json").write_text(json.dumps(counts, indent=2))
    grid_cfg = {**cfg, "splits": {**cfg["splits"], "n_seeds": n_seeds}}
    grid_fn = dynamics.sensitivity_grid_ratio if main else dynamics.sensitivity_grid
    grid_fn(ranks, grid_cfg, agg).to_csv(out / "sensitivity_grid.csv", index=False)
    per = pd.DataFrame([{"seed": s, "model": m, **{k: r[m][k] for k in
                         ("r2_train", "r2_val", "r2_test", "best_iteration", "n_features")}}
                        for s, r in r2.items() for m in ("genomics", "joint")])
    per.to_csv(out / "r2_per_split.csv", index=False)
    summ = per.groupby("model")["r2_test"].agg(["count", "mean", "std", "min", "max"])
    summ.to_csv(out / "r2_summary.csv")
    log.info("aggregate: %s", {k: v for k, v in counts.items() if not k.startswith("per_split")})
    return agg, counts, summ


def _score_k(data, train_pos, score_pos, k_grid, cfg, seed, n_jobs=None):
    """R² on score_pos for each k. Residualization and ranking on the fit part of train_pos, early stopping on
    its val part, refit on all of train_pos."""
    groups = _groups(data, cfg)
    fit, val = split._holdout(np.asarray(train_pos), groups, cfg["splits"]["val_fraction_of_train"], seed)
    sp = split.Split(seed, fit, val, np.asarray(score_pos))
    y, _ = split.residualize(data.y.values, data.covariates.values, sp.fit)
    ranking = prescreen.rank_snps(data.snps.iloc[sp.fit], y[sp.fit], cfg["prescreen"]["min_variance"])
    out = {}
    for k in k_grid:
        _, info = train.fit_model(data.snps[ranking.index[:k].tolist()], y, sp, cfg, seed, n_jobs)
        out[k] = info
    return out


def run_nested_cv(data, cfg, k_grid=None, n_jobs=None):
    nc = cfg["nested_cv"]
    k_grid = k_grid or nc["k_grid"]
    seed, groups = nc["seed"], _groups(data, cfg)
    outer = split.group_kfold(np.arange(data.n), groups, nc["outer_folds"], seed)
    rows, inner_rows = [], []
    for o, (otr, ote) in enumerate(outer, 1):
        inner_seed = seed + o
        inner = split.group_kfold(otr, groups, nc["inner_folds"], inner_seed)
        for i, (itr, iva) in enumerate(inner, 1):
            for k, info in _score_k(data, itr, iva, k_grid, cfg, seed, n_jobs).items():
                inner_rows.append({"outer_fold": o, "inner_fold": i, "inner_seed": inner_seed, "k": k,
                                   "r2": info["r2_test"], "best_iteration": info["best_iteration"]})
        inner_df = pd.DataFrame([r for r in inner_rows if r["outer_fold"] == o])
        means = inner_df.groupby("k")["r2"].mean()
        best_k = int(means.idxmax())
        outer_info = _score_k(data, otr, ote, [best_k], cfg, seed, n_jobs)[best_k]
        row = {"outer_fold": o, "n_outer_train": len(otr), "n_outer_test": len(ote), "best_k": best_k,
               "outer_r2": outer_info["r2_test"], "outer_best_iteration": outer_info["best_iteration"]}
        row.update({f"inner_mean_r2_k={k}": means[k] for k in k_grid})
        rows.append(row)
        log.info("nested CV outer fold %d: best_k %d, outer R2 %.4f", o, best_k, row["outer_r2"])
    return pd.DataFrame(rows), pd.DataFrame(inner_rows)


def choose_k(nested, k_grid, default_k):
    """Mode of best_k across outer folds, unless the inner curve is flat within 1 SD: then keep default_k.

    Flat: (max_k mean inner R²) - (mean inner R² at default_k) <= SD across outer folds at the argmax k.
    Tied modes: the tied k with the highest mean inner R².
    """
    cols = {k: f"inner_mean_r2_k={k}" for k in k_grid}
    mean = pd.Series({k: nested[c].mean() for k, c in cols.items()})
    sd = pd.Series({k: nested[c].std(ddof=1) for k, c in cols.items()})
    k_max = int(mean.idxmax())
    flat = bool(mean[k_max] - mean[default_k] <= sd[k_max])
    mode = nested["best_k"].mode()
    chosen = default_k if flat else int(mean[mode.values].idxmax())
    return {"inner_mean": mean.to_dict(), "inner_sd": sd.to_dict(), "k_max_mean": k_max,
            "flat_within_1sd": flat, "best_k_per_fold": nested["best_k"].tolist(), "chosen_k": chosen,
            "default_k": default_k}
