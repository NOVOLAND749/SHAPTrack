"""Step 0: load the analysis dataset, covariates, genetic PCs and family clusters.

Run as a module to write the Step 0 tables:  python -m src.data
"""
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


def load_config(path=None):
    """config.yaml, or the file named by $SHAPTRACK_CONFIG (e.g. a local copy with site-specific paths)."""
    path = path or os.environ.get("SHAPTRACK_CONFIG", "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=handlers, force=True)


@dataclass
class Dataset:
    snps: pd.DataFrame          # dosages (REF allele count), float32, index = participant id (str)
    metabolites: pd.DataFrame   # float32, raw scale (min-imputed upstream)
    y: pd.Series                # medication-adjusted LDL-C, float64
    covariates: pd.DataFrame    # columns.covariates, float64; sex coded 0 = M, 1 = F
    groups: pd.Series           # family cluster id; unrelated participants are their own cluster
    meta: pd.DataFrame          # ancestry and genetic QC fields, not used as features

    @property
    def n(self):
        return len(self.y)

    def subset(self, ids):
        return Dataset(self.snps.loc[ids], self.metabolites.loc[ids], self.y.loc[ids],
                       self.covariates.loc[ids], self.groups.loc[ids], self.meta.loc[ids])


def family_clusters(ids, rel_path):
    """Connected components of the relatedness graph restricted to `ids`."""
    rel = pd.read_csv(rel_path, sep=r"\s+", dtype={"ADM_GWAS_COM_1": str, "ADM_GWAS_COM_2": str})
    ids = pd.Index(ids)
    pairs = rel[rel["ADM_GWAS_COM_1"].isin(ids) & rel["ADM_GWAS_COM_2"].isin(ids)]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["ADM_GWAS_COM_1"], pairs["ADM_GWAS_COM_2"]):
        parent[find(a)] = find(b)
    groups = pd.Series([find(i) if i in parent else i for i in ids], index=ids, name="family")
    return groups, pairs


def drop_qc_duplicates(df, met_cols, source_path, id_col):
    """Resolve participants that appear more than once in the pkl.

    Upstream, sample ids were reduced to their digits, so Metabolon QC samples "LQCnnnn" were merged onto
    real samples "nnnn". For each duplicated participant keep one row whose metabolite values equal the
    real sample "nnnn" in the source table (non-missing cells); fail if that is not unique.
    """
    dup = df[id_col].duplicated(keep=False)
    if not dup.any():
        return df
    src = pd.read_csv(source_path, dtype={"ADM_METABOLON2_COM": str}).set_index("ADM_METABOLON2_COM")
    src.columns = src.columns.astype(str)
    keep = ~dup
    for pid, rows in df[dup].groupby(id_col):
        ref = src.loc[str(int(rows["ADM_METABOLON2_COM"].iloc[0])), met_cols].astype(float).values
        observed = ~np.isnan(ref)
        match = [np.allclose(r[observed], ref[observed], rtol=1e-5, atol=1e-8)
                 for r in rows[met_cols].astype(float).values]
        matched = rows[match]
        if len(matched) == 0 or len(matched.drop_duplicates(subset=met_cols)) != 1:
            raise ValueError(f"participant {pid}: cannot identify a unique real-sample row among {len(rows)}")
        keep[matched.index[0]] = True
    log.info("duplicated participants: %d (%d rows); kept the real-sample row, dropped %d rows (QC-sample "
             "profiles and exact copies)", df.loc[dup, id_col].nunique(), int(dup.sum()), int((~keep).sum()))
    return df[keep]


def load_dataset(cfg):
    paths, cols = cfg["paths"], cfg["columns"]
    df = pd.read_pickle(paths["data_pkl"])
    snp_cols = [c for c in df.columns if str(c).startswith(cols["snp_prefix"])]
    met_cols = [c for c in df.columns if re.fullmatch(cols["metabolite_pattern"], str(c))]
    log.info("features: %d SNPs, %d metabolites", len(snp_cols), len(met_cols))
    qc = cfg.get("snp_qc")
    if qc:   # optional SNP QC (sensitivity): imputation Rsq and MAF in the analysis sample (scripts/snp_maf_info.py)
        info = pd.read_csv(qc["info_table"], index_col="feature").reindex(snp_cols)
        if info[["rsq_min", "maf_sample"]].isna().any().any():
            raise ValueError(f"SNP QC table {qc['info_table']} lacks some model SNPs")
        low_rsq, low_maf = info["rsq_min"] < qc["min_rsq"], info["maf_sample"] < qc["min_maf"]
        snp_cols = [c for c, drop in zip(snp_cols, (low_rsq | low_maf).values) if not drop]
        log.info("SNP QC: Rsq < %s %d, MAF < %s %d, both %d -> %d SNPs kept", qc["min_rsq"], int(low_rsq.sum()),
                 qc["min_maf"], int(low_maf.sum()), int((low_rsq & low_maf).sum()), len(snp_cols))

    n0 = len(df)
    df = drop_qc_duplicates(df, met_cols, paths["metabolomics_source"], cols["id"])
    df[cols["id"]] = df[cols["id"]].astype(str)
    df = df.set_index(cols["id"])
    if df.index.duplicated().any():
        raise ValueError(f"{df.index.duplicated().sum()} duplicated participant ids remain")
    n1 = len(df)
    if paths.get("outcome_file"):
        # Replace the pkl outcome (e.g. the corrected medication-adjusted LDL written by scripts/make_corrected_ldl.py).
        col = cols["outcome_file_column"]
        new = pd.read_csv(paths["outcome_file"], dtype={cols["id"]: str}).set_index(cols["id"])[col]
        new = new.reindex(df.index)
        if new.isna().any():
            raise ValueError(f"{int(new.isna().sum())} participants lack {col} in {paths['outcome_file']}")
        changed = (new - df[cols["outcome"]]).abs() > 1e-6
        log.info("outcome %s replaced by %s from %s: %d of %d values changed", cols["outcome"], col,
                 paths["outcome_file"], int(changed.sum()), len(df))
        df[cols["outcome"]] = new.values
    if cfg["sample"]["drop_outcome_le_zero"]:
        df = df.loc[df[cols["outcome"]] > 0]
    log.info("participants: %d rows loaded, %d unique, %d after %s > 0", n0, n1, len(df), cols["outcome"])

    # Covariates: age and sex from paths.covariates, genetic PCs from paths.genetic_pcs.
    cov = pd.read_csv(paths["covariates"], sep="\t", dtype={"#IID": str}).set_index("#IID")
    pcs = pd.read_csv(paths["genetic_pcs"], sep=r"\s+", dtype={"ADM_GWAS_COM": str}).set_index("ADM_GWAS_COM")
    pc_cols = [c for c in cols["covariates"] if c.startswith("PC")]
    other = [c for c in cols["covariates"] if c not in pc_cols]
    covariates = pd.concat([cov.reindex(df.index)[other], pcs.reindex(df.index)[pc_cols]], axis=1)
    missing = covariates.isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{missing.sum()} participants lack a covariate, e.g. {list(covariates.index[missing][:5])}")
    if "sex" in covariates:
        if not set(covariates["sex"].unique()) <= {1, 2}:
            raise ValueError(f"unexpected sex codes {sorted(covariates['sex'].unique())}")
        covariates["sex"] = (covariates["sex"] == 2).astype(float)
        discordant = (covariates["sex"].map({0.0: "M", 1.0: "F"}) != df["SEX_ASK_COM"]).sum()
        log.info("sex: covariate file vs SEX_ASK_COM discordant in %d participants", discordant)
    covariates = covariates[cols["covariates"]].astype(float)

    snps = df[snp_cols].astype(np.float32)
    mets = df[met_cols].astype(np.float32)
    for name, block in (("SNP", snps), ("metabolite", mets)):
        n_nan = int(block.isna().sum().sum())
        if n_nan:
            raise ValueError(f"{n_nan} NaNs in the {name} block")

    meta = pd.concat([df[[cols["ancestry"], "SEX_ASK_COM"]],
                      pcs.reindex(df.index)[["pca.cluster.id", "in.kinship"]]], axis=1)
    groups, pairs = family_clusters(df.index, paths["relatedness"])
    sizes = groups.value_counts()
    log.info("relatedness: %d pairs within sample, %d participants in %d family clusters",
             len(pairs), int(sizes[sizes > 1].sum()), int((sizes > 1).sum()))
    return Dataset(snps, mets, df[cols["outcome"]].astype(float), covariates, groups, meta)


def metabolite_annotation(cfg):
    ann = pd.read_csv(cfg["paths"]["metabolite_annotation"])
    ann["CHEM_ID"] = ann["CHEM_ID"].astype(str)
    return ann.set_index("CHEM_ID")[["CHEMICAL_NAME", "SUPER_PATHWAY", "SUB_PATHWAY"]]


def write_step0_tables(data, cfg):
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    anc = cfg["columns"]["ancestry"]
    tab = data.meta[anc].value_counts(dropna=False).rename("n").to_frame()
    tab["pct"] = 100 * tab["n"] / data.n
    tab.index.name = anc
    pca = data.meta["pca.cluster.id"].value_counts(dropna=False).rename("n").to_frame()
    pca["pct"] = 100 * pca["n"] / data.n
    pca.index = [f"pca.cluster.id={i}" for i in pca.index]
    pd.concat([tab, pca]).to_csv(out / "ancestry_table.csv")

    mets = data.metabolites
    at_min = mets.eq(mets.min(axis=0), axis=1).sum(axis=0)
    ann = pd.read_csv(cfg["paths"]["metabolite_annotation"])
    ann["CHEM_ID"] = ann["CHEM_ID"].astype(str)
    ann = ann.set_index("CHEM_ID")
    rate = pd.DataFrame({"n_at_min": at_min, "frac_at_min": at_min / data.n})
    rate = rate.join(ann[["CHEMICAL_NAME", "SUPER_PATHWAY", "SUB_PATHWAY"]], how="left")
    rate.index.name = "CHEM_ID"
    rate.sort_values("frac_at_min", ascending=False).to_csv(out / "metabolite_imputation_rate.csv")

    sizes = data.groups.value_counts()
    fam = sizes.value_counts().sort_index().rename("n_clusters").to_frame()
    fam.index.name = "cluster_size"
    fam.to_csv(out / "family_clusters.csv")
    log.info("wrote ancestry_table.csv, metabolite_imputation_rate.csv, family_clusters.csv to %s", out)
    return rate


if __name__ == "__main__":
    setup_logging()
    cfg = load_config()
    data = load_dataset(cfg)
    rate = write_step0_tables(data, cfg)
    print(f"n = {data.n}, SNPs = {data.snps.shape[1]}, metabolites = {data.metabolites.shape[1]}, "
          f"covariates = {data.covariates.shape[1]} {list(data.covariates.columns)}")
    print(f"metabolites with >1 value at column minimum: {(rate['n_at_min'] > 1).sum()}; "
          f"median fraction at min among those: {rate.loc[rate['n_at_min'] > 1, 'frac_at_min'].median():.3f}")
