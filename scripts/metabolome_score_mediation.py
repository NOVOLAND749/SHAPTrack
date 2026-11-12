"""Metabolome-score mediation screen over all SNPs (config metabolome_score).

Writes results/metabolome_score/{all,nonlipid}/ (score.csv, scan.csv, losing*.csv, gaining*.csv), results/metabolome_score/summary.json and
results/metabolome_score/glgc/annotated.csv (primary sets).
Usage (from the repository root): python scripts/metabolome_score_mediation.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metabolome_mediation as mm  # noqa: E402
from src.data import load_config, load_dataset, metabolite_annotation, setup_logging  # noqa: E402
from src.ols import transform_metabolites  # noqa: E402

SET_COLS = ["losing", "gaining", "gaining_secondary"]


def main():
    cfg = load_config()
    ms = cfg["metabolome_score"]
    base = Path(cfg["paths"]["results_dir"])
    res = base / ms["results_subdir"]
    setup_logging(base / "logs" / f"metabolome_score_mediation_{time.strftime('%Y%m%d_%H%M%S')}.log")
    t0 = time.time()
    data = load_dataset(cfg)
    mets_t = transform_metabolites(data.metabolites, cfg["downstream"]["metabolite_transform"])
    names = metabolite_annotation(cfg)
    lipid = names.index[names["SUPER_PATHWAY"].isin(ms["sensitivity_exclude_super_pathway"])]
    ann = pd.read_csv(base / "glgc" / "annotated_all_snps.csv", index_col="feature", dtype={"rsid": str})
    groups = data.groups.values if cfg["splits"]["group_related"] else None
    sc = ms["score"]
    lo, hi, num = sc["alphas_log10"]
    alphas = np.logspace(lo, hi, num)
    ann_cols = ["rsid", "overlapping_genes", "upstream_gene", "downstream_gene", "glgc_label", "glgc_pvalue_neg_log10_GC"]

    summary = {"n": data.n}
    for tag, mets in (("all", mets_t), ("nonlipid", mets_t.drop(columns=mets_t.columns.intersection(lipid)))):
        out = res / tag
        out.mkdir(parents=True, exist_ok=True)
        m_hat, info = mm.crossfit_score(mets, data.y.values, groups, sc["folds"], sc["seed"], alphas)
        pd.Series(m_hat, index=data.y.index, name="m_hat").rename_axis("id").to_csv(out / "score.csv")
        scan = mm.scan(data.snps, data.y.values, m_hat, data.covariates.values)
        scan, st = mm.classify(scan, cfg)
        scan = scan.join(ann[ann_cols])
        scan.to_csv(out / "scan.csv")
        for col in [c for c in scan.columns if c.startswith(("losing", "gaining"))]:
            scan[scan[col]].sort_values("sobel_p").to_csv(out / f"{col}.csv")
        st["score"] = info
        for col in SET_COLS:
            st[f"glgc_{col}"] = scan.loc[scan[col], "glgc_label"].value_counts().to_dict()
        summary[tag] = st
        if tag == "all":
            sets = pd.concat([ann.loc[scan.index[scan[c]]].assign(set=c) for c in SET_COLS])
            (res / "glgc").mkdir(parents=True, exist_ok=True)
            sets.to_csv(res / "glgc" / "annotated.csv")
    summary["seconds"] = round(time.time() - t0)
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
