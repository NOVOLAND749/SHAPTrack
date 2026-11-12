"""Smoke test: leakage tests, then a small end-to-end run on real data. Must pass before any full run.

Uses config smoke_test (n_samples participants drawn with seed splits.seed_base, n_seeds splits, k SNPs).
Writes only to a temporary directory. Usage (from the repository root): python scripts/smoke_test.py
"""
import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import test_no_leakage  # noqa: E402
from src import pipeline, split  # noqa: E402
from src.data import load_config, load_dataset, setup_logging  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def main():
    t0 = time.time()
    print("== leakage tests ==")
    if test_no_leakage.run_all():
        sys.exit("smoke test FAILED: leakage tests")

    setup_logging()
    cfg = load_config()
    sm = cfg["smoke_test"]
    data = load_dataset(cfg)
    sample_seed = cfg["splits"]["seed_base"]
    ids = np.random.default_rng(sample_seed).choice(data.y.index.values, sm["n_samples"], replace=False)
    sub = data.subset(pd.Index(ids))
    seeds = split.split_seeds(cfg)[: sm["n_seeds"]]
    out = Path(tempfile.mkdtemp(prefix="shaptrack_smoke_"))
    print(f"== end to end: n = {sub.n} (sample seed {sample_seed}), seeds {seeds}, k = {sm['k']}, out {out} ==")

    for s in seeds:
        summary, dyn = pipeline.run_split(sub, s, cfg, sm["k"], out / "splits" / f"seed_{s}", save_shap=(s == seeds[0]))
        d = out / "splits" / f"seed_{s}"
        for f in ("r2.json", "imp_genomics.csv", "imp_joint.csv", "dynamics.csv", "selected_snps.txt", "partition.csv"):
            check((d / f).exists(), f"seed {s}: {f} written")
        part = pd.read_csv(d / "partition.csv", dtype={"id": str}, index_col="id")["partition"]
        check(len(part) == sub.n and part.index.is_unique, f"seed {s}: every participant in exactly one partition")
        fam = pd.DataFrame({"g": sub.groups, "p": part})
        check(fam.groupby("g")["p"].nunique().max() == 1, f"seed {s}: no family spans two partitions")
        check(len(dyn) == sm["k"], f"seed {s}: {len(dyn)} SNPs ranked")
        imp_j = pd.read_csv(d / "imp_joint.csv")
        check(len(imp_j) == sm["k"] + sub.metabolites.shape[1], f"seed {s}: joint importance covers SNPs + metabolites")
        for m in ("genomics", "joint"):
            check(all(math.isfinite(summary[m][k]) for k in ("r2_train", "r2_val", "r2_test")),
                  f"seed {s} {m}: R2 finite (test {summary[m]['r2_test']:.3f})")
        check(summary["n_fit"] + summary["n_val"] + summary["n_test"] == sub.n, f"seed {s}: fit + val + test = n")
    check((out / "splits" / f"seed_{seeds[0]}" / "shap_joint.npz").exists(), "per-sample SHAP saved for first seed")

    ranks, r2 = pipeline.load_splits(out / "splits", seeds)
    agg_cfg = copy.deepcopy(cfg)
    agg_cfg["splits"]["n_seeds"] = len(seeds)
    agg, counts, summ = pipeline.write_aggregate(ranks, r2, agg_cfg, out / "aggregate")
    for f in ("dynamics_aggregate.csv", "gaining.csv", "losing.csv", "sensitivity_grid.csv", "r2_summary.csv",
              "r2_per_split.csv", "stage_counts.json"):
        check((out / "aggregate" / f).exists(), f"aggregate: {f} written")
    grid = pd.read_csv(out / "aggregate" / "sensitivity_grid.csv")
    check(grid["is_primary"].sum() == 1, "sensitivity grid contains exactly one primary row")
    print(f"  stage counts: {json.dumps({k: v for k, v in counts.items() if not k.startswith('per_split')})}")

    s2 = seeds[0]
    summary2, dyn2 = pipeline.run_split_main(sub, s2, cfg, sm["k"], out / "main" / "splits" / f"seed_{s2}")
    check((out / "main" / "splits" / f"seed_{s2}" / "r2.json").exists(), f"main seed {s2}: outputs written")
    check(len(summary2["genomics"]["fold_best_iteration"]) == cfg["main_models"]["early_stopping"]["cv_folds"],
          f"main seed {s2}: CV early stopping per fold {summary2['genomics']['fold_best_iteration']}, "
          f"R2 test genomics {summary2['genomics']['r2_test']:.3f}, joint {summary2['joint']['r2_test']:.3f}")
    check(set(dyn2["label"]) <= {"gaining", "losing", "none"} and "rel_ratio" in dyn2,
          f"main seed {s2}: ratio labels {dyn2['label'].value_counts().to_dict()}")

    nested, _ = pipeline.run_nested_cv(sub, cfg, k_grid=[50, sm["k"]])
    decision = pipeline.choose_k(nested, [50, sm["k"]], sm["k"])
    check(len(nested) == cfg["nested_cv"]["outer_folds"], f"nested CV ran {len(nested)} outer folds")
    print(f"  nested CV decision (smoke only): {decision['chosen_k']}, flat {decision['flat_within_1sd']}")
    print(f"smoke test PASSED in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
