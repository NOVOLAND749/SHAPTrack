"""Per-SNP minor allele frequency in the analysis sample and imputation Rsq (CLSA .mfi files), for all model SNPs.

Writes <results_dir>/glgc/snp_maf_info.csv (feature, alt/counted allele, MAF in sample, carriers of the minor allele,
Rsq min over imputation batches). Usage (from the repository root): python scripts/snp_maf_info.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, load_dataset, setup_logging  # noqa: E402


def main():
    setup_logging()
    cfg = load_config()
    data = load_dataset(cfg)
    S = data.snps
    counted_freq = S.mean(axis=0) / 2                      # dosage counts the allele after "_" in the column name
    maf = np.minimum(counted_freq, 1 - counted_freq)
    minor_dosage = np.where(counted_freq <= 0.5, S.values, 2 - S.values)
    carriers = (minor_dosage >= 0.5).sum(axis=0)
    out = pd.DataFrame({"counted_allele_freq": counted_freq.values, "maf_sample": maf.values,
                        "minor_allele_carriers": carriers}, index=S.columns)
    out["variant_id"] = [c.rsplit("_", 1)[0] for c in S.columns]
    info = []
    for ch in sorted({c.split(":")[0].replace("chr", "") for c in S.columns}, key=lambda x: int(x)):
        f = cfg["paths"]["imputation_info"].format(chr=ch)
        t = pd.read_csv(f, sep=r"\s+", usecols=["rsid", "alt_allele_frequency.1", "Rsq.1", "Rsq.2"], dtype=str)
        for c in ("alt_allele_frequency.1", "Rsq.1", "Rsq.2"):
            t[c] = pd.to_numeric(t[c], errors="coerce")       # a few non-numeric entries in the .mfi files
        info.append(t[t["rsid"].isin(set(out["variant_id"]))])
    info = pd.concat(info).drop_duplicates("rsid").set_index("rsid")
    out["rsq_min"] = info[["Rsq.1", "Rsq.2"]].min(axis=1).reindex(out["variant_id"]).values
    out["alt_freq_clsa_imputation"] = info["alt_allele_frequency.1"].reindex(out["variant_id"]).values
    out.index.name = "feature"
    dest = Path(cfg["paths"]["results_dir"]) / "glgc" / "snp_maf_info.csv"
    out.to_csv(dest)
    print(f"{len(out)} SNPs; Rsq missing for {int(out.rsq_min.isna().sum())}; wrote {dest}")
    q = out["maf_sample"]
    print("MAF in sample: <0.01 %d, 0.01-0.05 %d, >=0.05 %d; median %.3f" % ((q < .01).sum(), q.between(.01, .05, inclusive="left").sum(), (q >= .05).sum(), q.median()))
    r = out["rsq_min"]
    print("Rsq: <0.3 %d, 0.3-0.8 %d, >=0.8 %d; median %.3f" % ((r < .3).sum(), r.between(.3, .8, inclusive="left").sum(), (r >= .8).sum(), r.median()))


if __name__ == "__main__":
    main()
