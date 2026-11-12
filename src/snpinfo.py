"""Allele and frequency columns for result tables (reads <results_dir>/glgc/snp_maf_info.csv; no analysis)."""
from pathlib import Path

import pandas as pd


def allele_table(features, results_dir):
    """effect (= counted) allele, other allele, EAF and MAF in the analysis sample, minor-allele carriers, imputation Rsq.

    Model columns are named chr:pos:ref:alt_<counted allele>; every beta in the pipeline is per counted allele.
    """
    info = pd.read_csv(Path(results_dir) / "glgc" / "snp_maf_info.csv", index_col="feature")
    out = pd.DataFrame(index=pd.Index(features, name="feature"))
    vid = out.index.str.rsplit("_", n=1).str[0]
    out["effect_allele"] = out.index.str.rsplit("_", n=1).str[1]
    ref, alt = vid.str.split(":").str[2], vid.str.split(":").str[3]
    out["other_allele"] = [a if e == r else r for e, r, a in zip(out["effect_allele"], ref, alt)]
    info = info.reindex(out.index)
    out["eaf_sample"] = info["counted_allele_freq"].values
    out["maf_sample"] = info["maf_sample"].values
    out["minor_allele_carriers"] = info["minor_allele_carriers"].values
    out["imputation_rsq"] = info["rsq_min"].values
    return out
