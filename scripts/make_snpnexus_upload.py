"""Write SNPnexus batch upload files for the SNP columns in the data pkl.

No header row:
SNPnexus rejects any line whose first field is not "Chromosome" or "dbsnp". Select assembly GRCh38 on upload.

Usage (from the repository root):
  python scripts/make_snpnexus_upload.py
      -> annotation/snpnexus_upload_all_snps.tsv, every SNP as chr pos ref alt (VCF style)
  python scripts/make_snpnexus_upload.py --missing-from annotation/snpnexus_genomic_mapping_<job>.tsv
      -> annotation/snpnexus_upload_indels_normalized.tsv, indels that got no dbSNP id, with the shared
         anchor base removed (pos + 1, empty allele written as "-"), the form SNPnexus reports for
         indels.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

OUT = Path("annotation/snpnexus_upload_all_snps.tsv")
OUT_INDELS = Path("annotation/snpnexus_upload_indels_normalized.tsv")
COLUMNS = ["Type", "Id", "Position", "Allele1", "Allele2", "Strand"]
# Column names look like chr10:100858896:G:A_G (chr:pos:ref:alt_countedAllele).
PATTERN = re.compile(r"chr([0-9XYMT]+):([0-9]+):([ACGT]+):([ACGT]+)_([ACGT]+)")


def all_snps():
    cfg = yaml.safe_load(open("config.yaml"))
    df = pd.read_pickle(cfg["paths"]["data_pkl"])
    snps = [c for c in df.columns if str(c).startswith(cfg["columns"]["snp_prefix"])]
    print(f"SNP columns in pkl: {len(snps)}")

    rows, bad = [], []
    for col in snps:
        m = PATTERN.fullmatch(col)
        if m is None:
            bad.append(col)
            continue
        chrom, pos, ref, alt, _ = m.groups()
        rows.append(("Chromosome", chrom, pos, ref, alt, "1"))
    if bad:
        raise ValueError(f"{len(bad)} SNP columns do not parse, e.g. {bad[:5]}")

    upload = pd.DataFrame(rows, columns=COLUMNS)
    if upload.duplicated().any():
        raise ValueError(f"{upload.duplicated().sum()} duplicated variants")
    return upload, OUT


def normalized_indels(result_path):
    res = pd.read_csv(result_path, sep="\t", dtype=str)
    missing = res[res["dbSNP"].isna()]
    rows, skipped = [], 0
    for vid in missing["Variant ID"]:
        chrom, pos, alleles, _ = vid.split(":")
        ref, alt = alleles.split("/")
        if len(ref) == 1 and len(alt) == 1:
            skipped += 1        # SNV without a dbSNP id; nothing to normalize
            continue
        if ref[0] != alt[0]:
            skipped += 1        # complex substitution without a shared anchor base
            continue
        rows.append(("Chromosome", chrom.removeprefix("chr"), str(int(pos) + 1), ref[1:] or "-", alt[1:] or "-", "1"))
    print(f"variants without dbSNP id: {len(missing)}; normalized indels written: {len(rows)}; "
          f"left out (SNV or no anchor base): {skipped}")
    return pd.DataFrame(rows, columns=COLUMNS), OUT_INDELS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing-from", help="SNPnexus genomic mapping result of the full upload")
    args = ap.parse_args()
    upload, out = normalized_indels(args.missing_from) if args.missing_from else all_snps()
    out.parent.mkdir(parents=True, exist_ok=True)
    upload.to_csv(out, sep="\t", index=False, header=False)
    print(f"Wrote {len(upload)} variants to {out}")
