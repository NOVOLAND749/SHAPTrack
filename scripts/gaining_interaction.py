"""Interaction scan (Step 6b model) for the final gaining SNPs of the main models, one scan per GLGC set (primary =
anchors, secondary = candidates) so that BH-FDR runs over the tests of that set only (manuscript Table 2), plus the
sensitivity refits (covariate interactions, HC3, 1% trimming) for every Bonferroni hit. Reads the saved labels.

Writes <out>/{full_primary.csv, full_secondary.csv, sensitivity.csv, summary.json}; default <out> =
<results_dir>/labels/gaining_interaction.
Usage (from the repository root): python scripts/gaining_interaction.py [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import interaction  # noqa: E402
from src.data import load_config, load_dataset, metabolite_annotation, setup_logging  # noqa: E402
from src.ols import transform_metabolites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    setup_logging()
    cfg = load_config()
    base = Path(cfg["paths"]["results_dir"])
    out = Path(a.out) if a.out else base / cfg["share_labels"]["results_subdir"] / "gaining_interaction"
    out.mkdir(parents=True, exist_ok=True)
    sl = cfg["share_labels"]
    lab = pd.read_csv(base / sl["results_subdir"] / sl["models"][0] / "snp_table.csv", index_col=0)
    gain = lab[lab["eligible"] & (lab["label"] == "gaining")]
    data = load_dataset(cfg)
    mets_t = transform_metabolites(data.metabolites, cfg["downstream"]["metabolite_transform"])
    names = metabolite_annotation(cfg)
    summary, hits = {}, []
    for s in ("primary", "secondary"):
        snps = gain.index[gain["set"] == s].tolist()
        if not snps:
            summary[s] = {"snps": 0}
            continue
        full, skipped = interaction.run(snps, data.snps, data.y, data.covariates, mets_t, cfg)
        full = full.join(names[["CHEMICAL_NAME", "SUB_PATHWAY"]], on="metabolite")
        full = full.join(gain[["rsid", "gene"]], on="snp")
        full.to_csv(out / f"full_{s}.csv", index=False)
        hits.append(full[full["bonferroni_sig"]].assign(set=s))
        top = full.sort_values("p").groupby("snp").head(1)
        summary[s] = {"snps": len(snps), "tests": int(full["estimable"].sum()), "metabolites_not_tested": skipped,
                      "bonferroni_sig": int(full["bonferroni_sig"].sum()), "fdr_sig": int(full["fdr_sig"].sum()),
                      "top_per_snp": top[["rsid", "gene", "CHEMICAL_NAME", "beta3", "p", "fdr_q"]].to_dict("records")}
    hits = pd.concat(hits) if hits else pd.DataFrame()
    if len(hits):
        sens = interaction.sensitivity(hits, data.snps, data.y, data.covariates, mets_t)
        sens = sens.merge(hits[["snp", "metabolite", "set", "rsid", "CHEMICAL_NAME"]], on=["snp", "metabolite"])
        sens.to_csv(out / "sensitivity.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
