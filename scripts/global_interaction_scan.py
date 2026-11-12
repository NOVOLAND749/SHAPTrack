"""Per split, every pre-selected SNP with its top SNP x metabolite interaction.

Interaction model = Step 6b (src/interaction.scan): LDL ~ 1 + SNP + covariates + M + SNP:M on all participants, metabolites
transformed as config downstream.metabolite_transform; one scan per SNP (the scan does not depend on the split, only the SNP
list does). Per SNP: top interaction (smallest p), per-SNP Bonferroni count (alpha / bonferroni_per_snp), BH q over every
SNP x metabolite test of this scan (the global q reported in the Discussion), best lipid-lowering-drug metabolite term. Per split: S = I_joint / (I_joint + I_genomics)
and the share-based split label (share_labels thresholds), joined to the SNP's top interaction.

Writes <results_dir>/main/split_interaction/{per_snp_top_interaction.csv, per_split_top_interaction.csv,
per_split_summary.csv, pvalues.npz, summary.json}.
Usage (from the repository root): python scripts/global_interaction_scan.py [--jobs 48]
"""
import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, load_dataset, metabolite_annotation, setup_logging  # noqa: E402
from src.interaction import scan  # noqa: E402
from src.ols import transform_metabolites  # noqa: E402

log = logging.getLogger("split_top_interaction")
DRUG = "statin|ezetimibe|fenofibr|gemfibroz|bezafibr|niacin"
G = {}


def _init(snps, y, C, M):
    G.update(snps=snps, y=y, C=C, M=M)


def _one(f):
    r = scan(G["snps"][f].values.astype(float), G["y"], G["C"], G["M"])
    return f, r["beta3"].values.astype(np.float32), r["se3"].values.astype(np.float32), r["p"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=48)
    a = ap.parse_args()
    setup_logging()
    t0 = time.time()
    cfg = load_config()
    ds, sl = cfg["downstream"], cfg["share_labels"]
    base = Path(cfg["paths"]["results_dir"])
    main = base / cfg["main_models"]["results_subdir"]
    out = main / "split_interaction"
    out.mkdir(parents=True, exist_ok=True)

    dyn = []
    for d in sorted((main / "splits").glob("seed_*")):
        x = pd.read_csv(d / "dynamics.csv")
        sel = pd.read_csv(d / "selected_snps.txt", header=None)[0]
        assert set(x["feature"]) == set(sel), d
        tot = x["shap_joint"] + x["shap_genomics"]
        x["S"] = (x["shap_joint"] / tot).where(tot > 0)
        x["split_label"] = np.where(x["S"] < sl["losing_max_share"], "losing",
                                       np.where(x["S"] > sl["gaining_min_share"], "gaining", "none"))
        x.loc[x["S"].isna(), "split_label"] = "undefined"
        dyn.append(x[["feature", "shap_genomics", "shap_joint", "rank_genomics", "rank_joint", "S", "split_label"]]
                   .assign(seed=int(d.name.split("_")[1])))
    dyn = pd.concat(dyn, ignore_index=True)
    snps_all = sorted(dyn["feature"].unique())
    log.info("%d splits, %d rows, %d distinct pre-selected SNPs", dyn["seed"].nunique(), len(dyn), len(snps_all))

    data = load_dataset(cfg)
    mets_t = transform_metabolites(data.metabolites, ds["metabolite_transform"])
    testable = mets_t.columns[mets_t.notna().all()]
    names = metabolite_annotation(cfg).reindex(testable.astype(str))
    M = mets_t[testable].values
    y, C = data.y.values.astype(float), data.covariates.values.astype(float)
    drug_cols = np.where(names["CHEMICAL_NAME"].str.contains(DRUG, case=False, na=False).values)[0]
    log.info("%d testable metabolites (%d not tested), %d drug metabolites: %s", len(testable),
             mets_t.shape[1] - len(testable), len(drug_cols), list(names["CHEMICAL_NAME"].iloc[drug_cols]))

    P = np.full((len(snps_all), len(testable)), np.nan, dtype=np.float64)
    B = np.full_like(P, np.nan, dtype=np.float32)
    SE = np.full_like(P, np.nan, dtype=np.float32)
    idx = {f: i for i, f in enumerate(snps_all)}
    t1 = time.time()
    with get_context("fork").Pool(a.jobs, initializer=_init, initargs=(data.snps, y, C, M)) as pool:
        for k, (f, b, se, p) in enumerate(pool.imap_unordered(_one, snps_all, chunksize=20)):
            i = idx[f]
            P[i], B[i], SE[i] = p, b, se
            if (k + 1) % 2000 == 0:
                log.info("scanned %d / %d SNPs (%.0f s)", k + 1, len(snps_all), time.time() - t1)
    log.info("scan done: %d SNPs x %d metabolites in %.0f s", len(snps_all), len(testable), time.time() - t1)

    ok = ~np.isnan(P)
    Q = np.full_like(P, np.nan)
    Q[ok] = multipletests(P[ok], method=ds["fdr_method"])[1]
    thr = ds["alpha"] / ds["bonferroni_per_snp"]
    Pf = np.where(ok, P, np.inf)
    top = Pf.argmin(1)
    r = np.arange(len(snps_all))
    Pd = Pf[:, drug_cols]
    dtop = drug_cols[Pd.argmin(1)]
    ann = pd.read_csv(base / "glgc" / "annotated_all_snps.csv", index_col="feature", dtype={"rsid": str})
    gene = ann["overlapping_genes"].fillna(ann["upstream_gene"].str.split("(").str[0] + " / " +
                                           ann["downstream_gene"].str.split("(").str[0])
    per_snp = pd.DataFrame({
        "snp": snps_all, "rsid": ann["rsid"].reindex(snps_all).values, "gene": gene.reindex(snps_all).values,
        "glgc_label": ann["glgc_label"].reindex(snps_all).values,
        "n_tests": ok.sum(1), "n_bonferroni": (P < thr).sum(1), "n_fdr_global": (Q < ds["alpha"]).sum(1),
        "top_metabolite_id": testable[top].astype(str), "top_metabolite": names["CHEMICAL_NAME"].values[top],
        "top_sub_pathway": names["SUB_PATHWAY"].values[top], "top_beta3": B[r, top], "top_se3": SE[r, top],
        "top_p": P[r, top], "top_bonferroni_sig": P[r, top] < thr, "top_fdr_q_global": Q[r, top],
        "top_is_drug": np.isin(top, drug_cols),
        "best_drug_metabolite": names["CHEMICAL_NAME"].values[dtop], "best_drug_beta3": B[r, dtop],
        "best_drug_p": P[r, dtop], "best_drug_fdr_q_global": Q[r, dtop],
        "n_selected": dyn.groupby("feature").size().reindex(snps_all).values,
    })
    per_snp.to_csv(out / "per_snp_top_interaction.csv", index=False)
    np.savez_compressed(out / "pvalues.npz", p=P.astype(np.float32), snps=np.array(snps_all),
                        metabolites=np.array(testable.astype(str)))

    per_split = dyn.merge(per_snp.drop(columns=["n_selected"]), left_on="feature", right_on="snp").drop(columns="snp")
    per_split = per_split.sort_values(["seed", "split_label", "top_p"])
    per_split.to_csv(out / "per_split_top_interaction.csv", index=False)

    rows = []
    for s, g in per_split.groupby("seed"):
        gain = g["split_label"] == "gaining"
        hit = g["top_bonferroni_sig"]
        tab = [[int((gain & hit).sum()), int((gain & ~hit).sum())], [int((~gain & hit).sum()), int((~gain & ~hit).sum())]]
        odds, fp = stats.fisher_exact(tab)
        rows.append({"seed": s, "n_snps": len(g), "n_gaining": int(gain.sum()), "n_losing": int((g["split_label"] == "losing").sum()),
                     "gaining_with_bonf_hit": tab[0][0], "other_with_bonf_hit": tab[1][0],
                     "share_gaining_bonf": tab[0][0] / max(1, gain.sum()), "share_other_bonf": tab[1][0] / max(1, (~gain).sum()),
                     "fisher_or": odds, "fisher_p": fp,
                     "gaining_drug_nominal": int((gain & (g["best_drug_p"] < ds["alpha"])).sum()),
                     "gaining_top_is_drug": int((gain & g["top_is_drug"]).sum()),
                     "gaining_drug_fdr_global": int((gain & (g["best_drug_fdr_q_global"] < ds["alpha"])).sum())})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "per_split_summary.csv", index=False)
    res = {"snps": len(snps_all), "metabolites": len(testable), "tests": int(ok.sum()), "bonferroni_threshold": thr,
           "snps_with_bonf_hit": int((per_snp["n_bonferroni"] > 0).sum()),
           "snps_with_global_fdr_hit": int((per_snp["n_fdr_global"] > 0).sum()),
           "expected_snps_with_bonf_hit_under_null": float(len(snps_all) * (1 - (1 - thr) ** len(testable))),
           "drug_tests_global_fdr": int((Q[:, drug_cols] < ds["alpha"]).sum()),
           "drug_min_p": float(np.nanmin(P[:, drug_cols])), "seconds": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
