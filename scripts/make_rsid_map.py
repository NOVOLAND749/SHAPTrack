"""rsID map (SNPnexus format) for the rebuilt SNP universe.

SNPs already in the previous SNPnexus result keep their row (rsID and genes). New SNPs get an rsID from UCSC snp141 on
hg38 (`paths.dbsnp_hg38`): SNV = chromEnd equals the GRCh38 position and the observed alleles contain REF and ALT;
deletion (REF = A + seq) = chromStart equals the position and chromEnd = position + len(seq); insertion (ALT = REF + seq)
= chromStart = chromEnd = position; alleles compared on the reported strand. Several matches: the lowest rs number. The
same rule is applied to the previously annotated SNPs to report its agreement with SNPnexus. Genes of new SNPs come from
the PLINK hg38 gene list (`paths.gene_list_hg38`: Chr, Pos1, Pos2, Gene): overlapping genes, or, when none overlaps, the
nearest gene to the left (Upstream Gene) and to the right (Downstream Gene) as GENE(distance bp), coordinate-based as in
the SNPnexus rows (final sets are also gene-checked with NCBI). Writes paths.snpnexus_annotation.
Usage (from the repository root): python scripts/make_rsid_map.py
"""
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, setup_logging  # noqa: E402

log = logging.getLogger("make_rsid_map")
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


def targets(ids):
    rows = []
    for v in ids:
        chrom, pos, ref, alt = v.split(":")
        pos = int(pos)
        if len(ref) == 1 and len(alt) == 1:
            kind, start, end, seq = "snv", pos - 1, pos, None
        elif len(alt) == 1 and ref[0] == alt:
            kind, start, end, seq = "del", pos, pos + len(ref) - 1, ref[1:]
        elif len(ref) == 1 and alt[0] == ref:
            kind, start, end, seq = "ins", pos, pos, alt[1:]
        else:
            kind, start, end, seq = "complex", None, None, None
        rows.append((v, chrom, pos, ref, alt, kind, start, end, seq))
    return pd.DataFrame(rows, columns=["id", "chrom", "pos", "ref", "alt", "kind", "start", "end", "seq"])


def dbsnp_matches(t, path, chunksize=4_000_000):
    t = t[t["kind"] != "complex"]
    keys = set(zip(t["chrom"], t["start"], t["end"]))
    cols = ["chrom", "chromStart", "chromEnd", "name", "strand", "observed", "class"]
    hits = []
    for ch in pd.read_csv(path, sep="\t", header=None, usecols=[1, 2, 3, 4, 6, 9, 11], names=cols,
                          dtype={"chrom": str, "name": str, "strand": str, "observed": str, "class": str},
                          chunksize=chunksize):
        k = list(zip(ch["chrom"], ch["chromStart"], ch["chromEnd"]))
        hits.append(ch[[x in keys for x in k]])
    db = pd.concat(hits, ignore_index=True)
    log.info("dbSNP rows at target coordinates: %d", len(db))
    m = t.merge(db, left_on=["chrom", "start", "end"], right_on=["chrom", "chromStart", "chromEnd"])

    def ok(r):
        obs = set(r["observed"].split("/"))
        if r["strand"] == "-":
            obs = {revcomp(a) if a != "-" else a for a in obs}
        if r["kind"] == "snv":
            return {r["ref"], r["alt"]} <= obs
        return "-" in obs and r["seq"] in obs
    m = m[m.apply(ok, axis=1)].copy()
    m["rsnum"] = m["name"].str.replace("rs", "", regex=False).astype(float)
    return m.sort_values("rsnum").drop_duplicates("id").set_index("id")["name"]


def gene_columns(t, path):
    """Overlapping / nearest left / nearest right gene per SNP from a PLINK glist (Chr Pos1 Pos2 Gene)."""
    g = pd.read_csv(path, sep=r"\s+", dtype={"Chr": str})
    out = pd.DataFrame("", index=t.index, columns=["Overlapping Genes", "Upstream Gene", "Downstream Gene"])
    for chrom, tc in t.groupby(t["chrom"].str.replace("chr", "", regex=False)):
        gc = g[g["Chr"] == chrom]
        st, en, nm = gc["Pos1"].values, gc["Pos2"].values, gc["Gene"].values
        for v, pos in zip(tc.index, tc["pos"]):
            ov = (st <= pos) & (en >= pos)
            if ov.any():
                out.at[v, "Overlapping Genes"] = ",".join(nm[ov])
                continue
            left, right = en < pos, st > pos
            if left.any():
                i = np.flatnonzero(left)[np.argmax(en[left])]
                out.at[v, "Upstream Gene"] = f"{nm[i]}({pos - en[i]}bp)"
            if right.any():
                i = np.flatnonzero(right)[np.argmin(st[right])]
                out.at[v, "Downstream Gene"] = f"{nm[i]}({st[i] - pos}bp)"
    return out.replace("", None)


def main():
    setup_logging()
    cfg = load_config()
    out = Path(cfg["paths"]["snpnexus_annotation"])
    ids = sorted({line.strip() for f in glob.glob(str(Path(cfg["gwas"]["out_dir"]) / "clump" / "chr*_index.txt"))
                  for line in open(f) if line.strip()})
    t = targets(ids)
    t["variant_id"] = t["chrom"] + ":" + t["pos"].astype(str) + ":" + t["ref"] + "/" + t["alt"] + ":1"
    old = pd.read_csv(cfg["paths"]["snpnexus_previous"], sep="\t", dtype=str).set_index("Variant ID")
    in_old = t["variant_id"].isin(old.index)
    log.info("SNPs: %d (SNV %d, deletion %d, insertion %d, complex %d); in previous SNPnexus result %d, new %d",
             len(t), *(int((t["kind"] == k).sum()) for k in ("snv", "del", "ins", "complex")), int(in_old.sum()),
             int((~in_old).sum()))
    rs = dbsnp_matches(t, cfg["paths"]["dbsnp_hg38"])
    # agreement of the dbSNP rule with SNPnexus on the previously annotated SNPs
    prev = t[in_old].set_index("id")
    nx = old.reindex(prev["variant_id"])["dbSNP"].values
    mine = rs.reindex(prev.index).values
    both = pd.notna(nx) & pd.notna(mine)
    log.info("check on %d previously annotated SNPs: SNPnexus rsID %d, dbSNP-141 rule %d, both %d, identical %d",
             len(prev), int(pd.notna(nx).sum()), int(pd.notna(mine).sum()), int(both.sum()),
             int((nx[both] == mine[both]).sum()))
    new = t[~in_old].set_index("id")
    rows = pd.DataFrame({"Variant ID": new["variant_id"], "Chrom.": new["chrom"].str.replace("chr", "", regex=False),
                         "Position": new["pos"].astype(str), "Alleles": new["ref"] + "/" + new["alt"],
                         "dbSNP": rs.reindex(new.index).values}).set_index("Variant ID")
    genes = gene_columns(new, cfg["paths"]["gene_list_hg38"])
    for c in genes.columns:
        rows[c] = genes[c].values
    log.info("new SNPs with genes from %s: overlapping %d, nearest left/right only %d, none %d",
             cfg["paths"]["gene_list_hg38"], int(genes["Overlapping Genes"].notna().sum()),
             int((genes["Overlapping Genes"].isna() & genes[["Upstream Gene", "Downstream Gene"]].notna().any(axis=1)).sum()),
             int(genes.isna().all(axis=1).sum()))
    for c in old.columns:
        if c not in rows:
            rows[c] = None
    res = pd.concat([old.reindex(t.loc[in_old, "variant_id"].values), rows[old.columns]])
    res.index.name = "Variant ID"
    log.info("new SNPs with an rsID from dbSNP 141: %d of %d (SNV %d of %d)", int(rows["dbSNP"].notna().sum()), len(rows),
             int(rows.loc[new["kind"].values == "snv", "dbSNP"].notna().sum()), int((new["kind"] == "snv").sum()))
    out.parent.mkdir(parents=True, exist_ok=True)
    res.reset_index().to_csv(out, sep="\t", index=False)
    log.info("wrote %s (%d rows, %d with rsID)", out, len(res), int(res["dbSNP"].notna().sum()))


if __name__ == "__main__":
    main()
