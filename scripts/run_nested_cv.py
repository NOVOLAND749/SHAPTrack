"""Step 7 (run once, before the split loop): nested CV over nested_cv.k_grid to choose k.

Writes results/nested_cv.csv (per outer fold), results/nested_cv_inner.csv (every inner fit) and
results/nested_cv_decision.json. Never edits config.yaml: if the rule picks a k other than
prescreen.k, it says so and exits non-zero.
Usage (from the repository root): python scripts/run_nested_cv.py [--n-jobs 24]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402
from src.data import load_config, load_dataset, setup_logging  # noqa: E402

log = logging.getLogger("run_nested_cv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing results/nested_cv.csv")
    args = ap.parse_args()
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    if (res / "nested_cv.csv").exists() and not args.force:
        sys.exit("results/nested_cv.csv exists; pass --force to overwrite")
    setup_logging(res / "logs" / f"run_nested_cv_{time.strftime('%Y%m%d_%H%M%S')}.log")
    nc = cfg["nested_cv"]
    log.info("nested CV: seed %d, outer %d, inner %d, k_grid %s, group_related %s", nc["seed"], nc["outer_folds"],
             nc["inner_folds"], nc["k_grid"], cfg["splits"]["group_related"])
    t0 = time.time()
    data = load_dataset(cfg)
    nested, inner = pipeline.run_nested_cv(data, cfg, n_jobs=args.n_jobs)
    nested.to_csv(res / "nested_cv.csv", index=False)
    inner.to_csv(res / "nested_cv_inner.csv", index=False)
    decision = pipeline.choose_k(nested, nc["k_grid"], nc["keep_k_if_flat_within_sd"])
    decision["seconds"] = round(time.time() - t0)
    (res / "nested_cv_decision.json").write_text(json.dumps(decision, indent=2))
    print(nested.to_string())
    print(json.dumps(decision, indent=2))
    if decision["chosen_k"] != cfg["prescreen"]["k"]:
        print(f"STOP: decision rule chose k = {decision['chosen_k']}, config prescreen.k = {cfg['prescreen']['k']}. "
              "Ask before changing config.yaml.")
        sys.exit(2)


if __name__ == "__main__":
    main()
