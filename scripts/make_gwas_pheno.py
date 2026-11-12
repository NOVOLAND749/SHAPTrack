"""Phenotype and keep files for the SNP-list GWAS: corrected LDL-C for the original GWAS_LDL_adjusted sample.

Same participants (FID/IID) as the original phenotype file; LDL replaced by the corrected value (untreated = measured
LDL-C, treated = TC/0.8 - HDL - TG/5, mg/dL); missing where the original was missing. Checks: untreated values equal the
original file. Also writes the analysis-sample keep file for the dosage export.
Usage (from the repository root): python scripts/make_gwas_pheno.py
Optional gwas.pheno_column (default LDL_corrected; e.g. ldl_measured for the measured-LDL run) and gwas.pheno_file
(default pheno_ldl_corrected.txt) select another column of corrected_table and the output name.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, setup_logging  # noqa: E402
from scripts.make_corrected_ldl import corrected_table  # noqa: E402

log = logging.getLogger("make_gwas_pheno")


def main():
    setup_logging()
    cfg = load_config()
    g = cfg["gwas"]
    out = Path(g["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    orig = pd.read_csv(g["pheno_original"], sep=r"\s+", dtype={"FID": str, "IID": str})
    orig.loc[orig["LDL"] <= 0, "LDL"] = np.nan                      # missing codes, if any
    tab = corrected_table(cfg)
    pcol = g.get("pheno_column", "LDL_corrected")
    log.info("GWAS phenotype column: %s", pcol)
    corr, lipid_med = tab[pcol], tab["lipid_med"]
    new = corr.reindex(orig["IID"]).values
    keep = orig["LDL"].notna()
    lost = keep & pd.isna(new)
    if lost.any():
        log.warning("%d participants with an original value lack a corrected value; set to NA", int(lost.sum()))
    ph = orig[["FID", "IID"]].copy()
    ph["LDL"] = np.where(keep, new, np.nan)
    treated = lipid_med.reindex(orig["IID"]).eq(True).values
    unt = keep.values & ~treated & ph["LDL"].notna().values
    diff = np.nanmax(np.abs(ph.loc[unt, "LDL"].values - orig.loc[unt, "LDL"].values))
    if diff > 1e-3:
        raise ValueError(f"untreated participants differ from the original phenotype (max abs diff {diff})")
    log.info("GWAS phenotype: %d rows, %d non-missing (original %d); treated %d, untreated %d (identical, max diff %.2g)",
             len(ph), int(ph["LDL"].notna().sum()), int(keep.sum()), int((keep.values & treated).sum()), int(unt.sum()), diff)
    s = ph["LDL"]
    log.info(pcol + ": median %.1f mean %.1f sd %.1f max %.1f | original: median %.1f mean %.1f sd %.1f max %.1f",
             s.median(), s.mean(), s.std(), s.max(), orig["LDL"].median(), orig["LDL"].mean(), orig["LDL"].std(),
             orig["LDL"].max())
    ph.to_csv(out / g.get("pheno_file", "pheno_ldl_corrected.txt"), sep=" ", index=False, na_rep="NA")
    ids = pd.read_csv(cfg["paths"]["outcome_file"], dtype={cfg["columns"]["id"]: str})[cfg["columns"]["id"]]
    pd.DataFrame({"FID": ids, "IID": ids}).to_csv(out / "keep_analysis.txt", sep=" ", index=False, header=False)
    log.info("analysis keep file: %d participants", len(ids))


if __name__ == "__main__":
    main()
