"""Single-metabolite mediation for the GLGC candidate SNPs (suggestive, 5e-8 <= p < 0.05) eligible for the share-based
labels (main models), same model as Step 6 / the anchor scan, to compare losing candidates with the others.

Writes <results_dir>/labels/candidate_mediation/{full.csv, top_per_snp.csv, summary.json}.
Usage (from the repository root): python scripts/candidate_mediation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import mediation  # noqa: E402
from src.data import load_config, load_dataset, metabolite_annotation, setup_logging  # noqa: E402
from src.ols import transform_metabolites  # noqa: E402
from scripts.anchor_mediation import annotate  # noqa: E402


def main():
    setup_logging()
    cfg = load_config()
    base = Path(cfg["paths"]["results_dir"])
    sl = cfg["share_labels"]
    lab = pd.read_csv(base / sl["results_subdir"] / sl["models"][0] / "snp_table.csv", index_col=0)
    cand = lab[(lab["set"] == "secondary") & lab["eligible"]]
    data = load_dataset(cfg)
    ds = cfg["downstream"]
    mets_t = transform_metabolites(data.metabolites, ds["metabolite_transform"])
    ann = pd.read_csv(base / "glgc" / "annotated_all_snps.csv", index_col="feature", dtype={"rsid": str})
    med, skipped = mediation.run(cand.index.tolist(), data.snps, data.y, data.covariates, mets_t, cfg)
    med = annotate(med, metabolite_annotation(cfg), ann)
    out = base / sl["results_subdir"] / "candidate_mediation"
    out.mkdir(parents=True, exist_ok=True)
    med.to_csv(out / "full.csv", index=False)
    mediation.top_per_snp(med, ds["report_top_n_per_snp"]).to_csv(out / "top_per_snp.csv", index=False)
    hit = med[med["bonferroni_sig"]]
    same = np.sign(hit["indirect"]) == np.sign(hit["c_direct"])
    supp = (~same) & (hit["c_direct"].abs() > hit["c_total"].abs())
    n_same = hit[same].groupby("snp").size().reindex(cand.index).fillna(0)
    n_supp = hit[supp].groupby("snp").size().reindex(cand.index).fillna(0)
    L = cand["losing"].astype(bool)
    G = cand["gaining"].astype(bool)
    tab = [[int((n_same[L] > 0).sum()), int((n_same[L] == 0).sum())], [int((n_same[~L] > 0).sum()), int((n_same[~L] == 0).sum())]]
    res = {"candidates_eligible": int(len(cand)), "tests": int(len(med)), "metabolites_not_tested": skipped,
           "losing": {"n": int(L.sum()), "with_same_dir_mediator": tab[0][0],
                      "share": float((n_same[L] > 0).mean()) if L.any() else None},
           "other": {"n": int((~L).sum()), "with_same_dir_mediator": tab[1][0],
                     "share": float((n_same[~L] > 0).mean()) if (~L).any() else None},
           "fisher_p_losing_vs_other": float(stats.fisher_exact(tab)[1]),
           "gaining": {"n": int(G.sum()), "with_suppression_mediator": int((n_supp[G] > 0).sum())},
           "snps_with_any_bonferroni_hit": int(hit["snp"].nunique())}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
