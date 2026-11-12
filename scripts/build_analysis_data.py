"""Analysis file for the rebuilt SNP universe: the original pkl with its SNP block replaced (never modifies the input data).

Reads the per-chromosome dosage exports of the index SNPs (scripts/run_gwas.sh; plink2 --export A, REF counted,
columns chr:pos:ref:alt_<counted>), maps them to the pkl rows by GWAS id, checks the allele coding against the original
pkl on SNPs present in both, and writes paths.data_pkl. All other columns (ids, metabolites, outcome) are unchanged.
Usage (from the repository root): python scripts/build_analysis_data.py
"""
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, setup_logging  # noqa: E402

log = logging.getLogger("build_analysis_data")


def main():
    setup_logging()
    cfg = load_config()
    paths, cols = cfg["paths"], cfg["columns"]
    gdir = Path(cfg["gwas"]["out_dir"])
    n_index = sum(1 for f in glob.glob(str(gdir / "clump" / "chr*_index.txt")) for _ in open(f))
    parts = []
    for f in sorted(glob.glob(str(gdir / "export" / "chr*.raw")), key=lambda x: int(Path(x).stem[3:])):
        r = pd.read_csv(f, sep="\t", dtype={"IID": str})
        parts.append(r.drop(columns=["FID", "PAT", "MAT", "SEX", "PHENOTYPE"]).set_index("IID").astype(np.float32))
    dos = pd.concat(parts, axis=1)
    log.info("exported dosages: %d participants x %d SNPs (index SNPs from clumping: %d)", *dos.shape, n_index)
    if dos.isna().any().any():
        raise ValueError(f"{int(dos.isna().sum().sum())} missing dosages")

    df = pd.read_pickle(paths["data_pkl_original"])
    old_snps = [c for c in df.columns if str(c).startswith(cols["snp_prefix"])]
    ids = pd.to_numeric(df[cols["id"]]).astype("int64").astype(str)
    if (~ids.isin(dos.index)).any():
        raise ValueError(f"{int((~ids.isin(dos.index)).sum())} pkl rows without exported genotypes")
    new = dos.reindex(ids.values)
    new.index = df.index
    shared = sorted(set(old_snps) & set(new.columns))
    diff = max(float(np.nanmax(np.abs(df[c].astype(np.float32).values - new[c].values))) for c in shared[:500]) if shared else np.nan
    log.info("SNP columns: original %d, new %d, shared %d; allele-coding check on %d shared SNPs: max |old - new| = %.4g",
             len(old_snps), new.shape[1], len(shared), min(len(shared), 500), diff)
    if shared and diff > 0.01:
        raise ValueError("dosages of shared SNPs differ from the original pkl: allele coding or sample mapping wrong")
    out = pd.concat([df.drop(columns=old_snps), new], axis=1)
    dest = Path(paths["data_pkl"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(dest)
    log.info("wrote %s: %d rows, %d SNP columns, %d other columns", dest, len(out), new.shape[1],
             out.shape[1] - new.shape[1])


if __name__ == "__main__":
    main()
