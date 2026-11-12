"""Single-metabolite mediation scan for every GLGC anchor SNP (p < 5e-8), independent of the SHAPTrack labels
(config main_models.anchor_mediation)                         -> <results_dir>/main/anchor_mediation/
Same Step 6 models: all participants, raw medication-adjusted LDL-C, covariates, log-z metabolites.
Usage (from the repository root): python scripts/anchor_mediation.py
"""
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import mediation  # noqa: E402
from src.data import load_config, load_dataset, metabolite_annotation, setup_logging  # noqa: E402
from src.ols import transform_metabolites  # noqa: E402

log = logging.getLogger("anchor_mediation")


def annotate(df, names, ann):
    out = df.join(names, on="metabolite")
    return out.join(ann[["rsid", "overlapping_genes", "glgc_label", "glgc_pvalue_neg_log10_GC"]], on="snp")


def main():
    cfg = load_config()
    base = Path(cfg["paths"]["results_dir"])
    res = base / cfg["main_models"]["results_subdir"]
    setup_logging(base / "logs" / f"anchor_mediation_{time.strftime('%Y%m%d_%H%M%S')}.log")
    t0 = time.time()
    data = load_dataset(cfg)
    ds = cfg["downstream"]
    mets_t = transform_metabolites(data.metabolites, ds["metabolite_transform"])
    names = metabolite_annotation(cfg)
    ann = pd.read_csv(base / "glgc" / "annotated_all_snps.csv", index_col="feature", dtype={"rsid": str})
    summary = {"n": data.n}

    # 1. every GLGC anchor
    anchors = ann.index[ann["glgc_label"] == "anchor"].tolist()
    out = res / "anchor_mediation"
    out.mkdir(parents=True, exist_ok=True)
    med, skipped = mediation.run(anchors, data.snps, data.y, data.covariates, mets_t, cfg)
    med = annotate(med, names, ann)
    med.to_csv(out / "full.csv", index=False)
    top = mediation.top_per_snp(med, ds["report_top_n_per_snp"])
    top.to_csv(out / "top_per_snp.csv", index=False)
    per_snp = med.groupby("snp").agg(rsid=("rsid", "first"), gene=("overlapping_genes", "first"),
                                     n_bonferroni=("bonferroni_sig", "sum"), n_fdr=("fdr_sig", "sum"),
                                     min_sobel_p=("sobel_p", "min"))
    per_snp.to_csv(out / "per_snp_summary.csv")
    summary["anchor_mediation"] = {"snps": len(anchors), "tests": len(med), "metabolites_not_tested": skipped,
                                   "snps_with_bonferroni_hit": int((per_snp["n_bonferroni"] > 0).sum()),
                                   "bonferroni_sig": int(med["bonferroni_sig"].sum()),
                                   "fdr_sig": int(med["fdr_sig"].sum())}
    log.info("anchor mediation: %s", summary["anchor_mediation"])

    summary["seconds"] = round(time.time() - t0)
    (res / "hypothesis_tests_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
