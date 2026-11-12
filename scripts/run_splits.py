"""Steps 1-4: the multi-split loop, then aggregation once every configured seed is complete.

Completed seeds (r2.json present) are skipped, so several processes can each take a subset of seeds.
Usage (from the repository root):
  python scripts/run_splits.py [--seeds 3001 3002 ...] [--n-jobs 24]
  python scripts/run_splits.py --aggregate-only
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline, split  # noqa: E402
from src.data import load_config, load_dataset, setup_logging  # noqa: E402

log = logging.getLogger("run_splits")


def main():
    cfg = load_config()
    all_seeds = split.split_seeds(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=all_seeds)
    ap.add_argument("--n-jobs", type=int, default=None, help="XGBoost threads (default: all cores)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute seeds that already have results")
    ap.add_argument("--version", choices=["v1", "v2"], default="v1", help="v1 = pre-specified, v2 = post hoc")
    args = ap.parse_args()
    unknown = set(args.seeds) - set(all_seeds)
    if unknown:
        sys.exit(f"seeds {sorted(unknown)} are not in config (seed_base + 0..n_seeds-1)")

    res = Path(cfg["paths"]["results_dir"])
    out_root = res / cfg["posthoc_v2"]["results_subdir"] if args.version == "v2" else res
    splits_dir = out_root / "splits"
    run = pipeline.run_split_v2 if args.version == "v2" else pipeline.run_split
    tag = "aggregate" if args.aggregate_only else f"{min(args.seeds)}-{max(args.seeds)}"
    setup_logging(res / "logs" / f"run_splits_{args.version}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    k = cfg["prescreen"]["k"]
    log.info("k = %d, seeds %s, n_jobs %s, group_related %s", k, args.seeds, args.n_jobs,
             cfg["splits"]["group_related"])

    if not args.aggregate_only:
        data = load_dataset(cfg)
        for s in args.seeds:
            d = splits_dir / f"seed_{s}"
            if (d / "r2.json").exists() and not args.force:
                log.info("seed %d already complete, skipping", s)
                continue
            summary, _ = run(data, s, cfg, k, d, n_jobs=args.n_jobs, save_shap=(s == cfg["splits"]["seed_base"]))
            log.info("seed %d done in %.0f s: test R2 genomics %.4f, joint %.4f", s, summary["seconds"],
                     summary["genomics"]["r2_test"], summary["joint"]["r2_test"])

    missing = [s for s in all_seeds if not (splits_dir / f"seed_{s}" / "r2.json").exists()]
    if missing:
        log.info("not aggregating: seeds %s incomplete", missing)
        return
    ranks, r2 = pipeline.load_splits(splits_dir, all_seeds)
    ks = {r["k"] for r in r2.values()}
    if ks != {k}:
        sys.exit(f"splits were run with k = {sorted(ks)}, config k = {k}; not aggregating")
    _, counts, summ = pipeline.write_aggregate(ranks, r2, cfg, out_root / "aggregate")
    print(summ.to_string())
    print({k: v for k, v in counts.items() if not k.startswith("per_split")})


if __name__ == "__main__":
    main()
