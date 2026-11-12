"""Step 5: rsID mapping (SNPnexus) and GLGC LDL-C evidence labels.

Every one of the SNP columns is annotated once, independent of the SHAP results; the final gaining/losing
sets are then looked up in that table. Merge with GLGC is on rsID only (CLSA is GRCh38, GLGC positions are b37);
alleles are used only to choose among GLGC rows that share an rsID (multi-allelic sites).

Run as a module to build results/glgc/annotated_all_snps.csv:  python -m src.glgc
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GLGC_COLS = ["rsID", "CHROM", "POS_b37", "REF", "ALT", "N", "POOLED_ALT_AF", "pvalue_neg_log10_GC", "pvalue_GC",
             "METAL_Effect", "METAL_StdErr"]
FEATURE = re.compile(r"(chr[0-9XYMT]+):([0-9]+):([ACGT]+):([ACGT]+)_([ACGT]+)")


def parse_features(features):
    rows = []
    for f in features:
        chrom, pos, ref, alt, counted = FEATURE.fullmatch(f).groups()
        rows.append((f, chrom, int(pos), ref, alt, counted, f"{chrom}:{pos}:{ref}/{alt}:1"))
    return pd.DataFrame(rows, columns=["feature", "chrom", "pos_grch38", "ref", "alt", "counted_allele",
                                       "variant_id"]).set_index("feature")


def snpnexus_map(features, path):
    nx = pd.read_csv(path, sep="\t", dtype=str).set_index("Variant ID")
    snps = parse_features(features)
    missing = ~snps["variant_id"].isin(nx.index)
    if missing.any():
        raise ValueError(f"{missing.sum()} SNPs absent from the SNPnexus result, e.g. {list(snps.index[missing][:3])}")
    ann = nx.reindex(snps["variant_id"])
    snps["rsid"] = ann["dbSNP"].values
    snps["overlapping_genes"] = ann["Overlapping Genes"].values
    snps["upstream_gene"] = ann["Upstream Gene"].values
    snps["downstream_gene"] = ann["Downstream Gene"].values
    snps["cytoband"] = ann["Cytoband"].values
    log.info("SNPnexus: %d SNPs, %d with rsID (%d SNVs / %d indels without)", len(snps), snps["rsid"].notna().sum(),
             int((snps["rsid"].isna() & (snps["ref"].str.len() == 1) & (snps["alt"].str.len() == 1)).sum()),
             int((snps["rsid"].isna() & ((snps["ref"].str.len() > 1) | (snps["alt"].str.len() > 1))).sum()))
    return snps


def read_glgc(path, rsids, chunksize=5_000_000):
    """GLGC rows whose rsID is in `rsids` (streams the 6 GB file in chunks)."""
    rsids = set(rsids)
    parts = []
    for chunk in pd.read_csv(path, sep="\t", usecols=GLGC_COLS, dtype={"rsID": str, "CHROM": str, "REF": str,
                                                                       "ALT": str}, chunksize=chunksize):
        parts.append(chunk[chunk["rsID"].isin(rsids)])
    out = pd.concat(parts, ignore_index=True)
    log.info("GLGC: %d rows for %d of %d rsIDs", len(out), out["rsID"].nunique(), len(rsids))
    return out


def _pick_row(snp, rows):
    """Choose the GLGC row for one SNP among rows sharing its rsID; return (row or None, match flag)."""
    ours = {snp["ref"], snp["alt"]}
    same = rows[[{r, a} == ours for r, a in zip(rows["REF"], rows["ALT"])]]
    if len(same) == 1:
        return same.iloc[0], "alleles_match"
    if len(same) > 1:
        return None, "ambiguous"
    if len(rows) == 1:
        return rows.iloc[0], "rsid_only_allele_mismatch"
    return None, "ambiguous"


def label(neglog10p, cfg):
    g = cfg["glgc"]
    return np.select([neglog10p > g["anchor_min_neglog10p"], neglog10p > g["candidate_min_neglog10p"],
                      neglog10p.notna()], ["anchor", "candidate", "unsupported"], "not_in_glgc")


def annotate(features, cfg):
    snps = snpnexus_map(features, cfg["paths"]["snpnexus_annotation"])
    glgc = read_glgc(cfg["paths"]["glgc_summary"], snps["rsid"].dropna())
    by_rsid = {k: v for k, v in glgc.groupby("rsID")}
    picked, flags = [], []
    for feature, snp in snps.iterrows():
        rows = by_rsid.get(snp["rsid"]) if isinstance(snp["rsid"], str) else None
        if rows is None:
            picked.append(None)
            flags.append("no_rsid" if not isinstance(snp["rsid"], str) else "rsid_not_in_glgc")
            continue
        row, flag = _pick_row(snp, rows)
        picked.append(row)
        flags.append(flag)
    g = pd.DataFrame([r if r is not None else pd.Series(dtype=object) for r in picked], index=snps.index)
    out = snps.join(g.add_prefix("glgc_"))
    out["glgc_match"] = flags
    p = out["glgc_pvalue_neg_log10_GC"].astype(float)
    out["glgc_label"] = label(p, cfg)
    out.loc[out["glgc_match"] == "no_rsid", "glgc_label"] = "no_rsid"
    # GLGC effect is per ALT (b37 alleles); express it per our counted allele.
    sign = np.where(out["glgc_ALT"] == out["counted_allele"], 1.0,
                    np.where(out["glgc_REF"] == out["counted_allele"], -1.0, np.nan))
    out["glgc_beta_per_counted_allele"] = sign * out["glgc_METAL_Effect"].astype(float)
    counts = out["glgc_match"].value_counts()
    log.info("GLGC merge: %s", counts.to_dict())
    log.info("GLGC labels (all SNPs): %s", out["glgc_label"].value_counts().to_dict())
    return out


def label_sets(annotated, sets):
    """sets: {"gaining": DataFrame indexed by feature, "losing": ...} -> long table and label x set counts."""
    rows = [annotated.loc[df.index].assign(set=name).join(df.add_prefix("dyn_")) for name, df in sets.items()]
    long = pd.concat(rows) if rows else pd.DataFrame()
    counts = pd.crosstab(long["glgc_label"], long["set"]) if len(long) else pd.DataFrame()
    return long, counts


def sanity(annotated, cfg):
    """Known anchors must be anchors if present (merge check)."""
    problems = []
    for rsid in ("rs174557", "rs55730499"):
        hit = annotated[annotated["rsid"] == rsid]
        if len(hit) and not (hit["glgc_label"] == "anchor").all():
            problems.append(f"{rsid} labeled {hit['glgc_label'].tolist()}")
    return problems


if __name__ == "__main__":
    from src.data import load_config, setup_logging
    setup_logging()
    cfg = load_config()
    # Feature names from the pkl so the table is keyed exactly like the model columns.
    feats = pd.read_pickle(cfg["paths"]["data_pkl"]).columns
    feats = [c for c in feats if str(c).startswith(cfg["columns"]["snp_prefix"])]
    ann = annotate(feats, cfg)
    out = Path(cfg["paths"]["results_dir"]) / "glgc"
    out.mkdir(parents=True, exist_ok=True)
    ann.to_csv(out / "annotated_all_snps.csv")
    problems = sanity(ann, cfg)
    print(ann["glgc_label"].value_counts().to_string())
    print(ann["glgc_match"].value_counts().to_string())
    print("sanity:", problems or "rs174557 and rs55730499 are anchors")
