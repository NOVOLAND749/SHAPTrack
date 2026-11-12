"""Steps 3-4: within-SNP ranks, per-split gain/loss labels, aggregation across splits."""
import logging
from itertools import product

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def split_ranks(imp_genomics, imp_joint, selected, rank_method="min"):
    """Ranks of mean |SHAP| among the selected SNPs only, in both models (1 = most important)."""
    g = imp_genomics.reindex(selected)
    j = imp_joint.reindex(selected)
    if g.isna().any() or j.isna().any():
        raise ValueError("selected SNPs missing from an importance table")
    df = pd.DataFrame({"shap_genomics": g.values, "shap_joint": j.values}, index=pd.Index(selected, name="feature"))
    df["rank_genomics"] = df["shap_genomics"].rank(ascending=False, method=rank_method)
    df["rank_joint"] = df["shap_joint"].rank(ascending=False, method=rank_method)
    df["delta_rank"] = df["rank_genomics"] - df["rank_joint"]
    return df


def label_split(df, gain_percentile, loss_percentile, floor, require_nonzero_shap=True):
    """Per-split labels: gaining / losing / none. Positive delta_rank = moved up in the joint model."""
    d = df["delta_rank"]
    hi, lo = d.quantile(gain_percentile), d.quantile(loss_percentile)
    gain = (d >= hi) & (d > 0) & (df["rank_joint"] <= floor)
    loss = (d <= lo) & (d < 0) & (df["rank_genomics"] <= floor)
    if require_nonzero_shap:
        gain &= df["shap_joint"] > 0
        loss &= df["shap_genomics"] > 0
    return pd.Series(np.select([gain, loss], ["gaining", "losing"], "none"), index=df.index, name="label"), hi, lo


def add_labels(df, cfg):
    dy = cfg["dynamics"]
    out = df.copy()
    out["label"], hi, lo = label_split(df, dy["gain_percentile"], dy["loss_percentile"],
                                       dy["importance_floor_rank"], dy["require_nonzero_shap"])
    stats = {"delta_rank_cut_gain": float(hi), "delta_rank_cut_loss": float(lo),
             "n_gaining": int((out["label"] == "gaining").sum()), "n_losing": int((out["label"] == "losing").sum()),
             "sum_delta_rank": float(df["delta_rank"].sum()),
             "n_zero_shap_genomics": int((df["shap_genomics"] == 0).sum()),
             "n_zero_shap_joint": int((df["shap_joint"] == 0).sum())}
    return out, stats


def aggregate(per_split, n_seeds, min_selection_fraction, stability_threshold):
    """per_split: {seed: DataFrame indexed by feature with delta_rank, ranks, shap values and label}."""
    long = pd.concat({s: d for s, d in per_split.items()}, names=["seed", "feature"]).reset_index()
    g = long.groupby("feature")
    agg = pd.DataFrame({
        "n_selected": g.size(),
        "n_gaining": g["label"].apply(lambda x: (x == "gaining").sum()),
        "n_losing": g["label"].apply(lambda x: (x == "losing").sum()),
        "mean_delta_rank": g["delta_rank"].mean(),
        "sd_delta_rank": g["delta_rank"].std(ddof=1),
        "mean_rank_genomics": g["rank_genomics"].mean(),
        "mean_rank_joint": g["rank_joint"].mean(),
        "mean_shap_genomics": g["shap_genomics"].mean(),
        "mean_shap_joint": g["shap_joint"].mean(),
    })
    agg["selection_fraction"] = agg["n_selected"] / n_seeds
    agg["eligible"] = agg["selection_fraction"] >= min_selection_fraction
    agg["stability_gaining"] = agg["n_gaining"] / agg["n_selected"]
    agg["stability_losing"] = agg["n_losing"] / agg["n_selected"]
    agg["final_label"] = np.select(
        [agg["eligible"] & (agg["stability_gaining"] >= stability_threshold),
         agg["eligible"] & (agg["stability_losing"] >= stability_threshold)],
        ["gaining", "losing"], "none")
    return agg.sort_values("mean_delta_rank", ascending=False)


def stage_counts(agg):
    return {
        "snps_selected_in_any_split": int(len(agg)),
        "eligible": int(agg["eligible"].sum()),
        "gaining_in_any_split": int((agg["n_gaining"] > 0).sum()),
        "losing_in_any_split": int((agg["n_losing"] > 0).sum()),
        "eligible_gaining_in_any_split": int((agg["eligible"] & (agg["n_gaining"] > 0)).sum()),
        "eligible_losing_in_any_split": int((agg["eligible"] & (agg["n_losing"] > 0)).sum()),
        "final_gaining": int((agg["final_label"] == "gaining").sum()),
        "final_losing": int((agg["final_label"] == "losing").sum()),
    }


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def sensitivity_grid(ranks, cfg, primary):
    """Final set sizes and Jaccard overlap with the primary sets over the pre-specified grid."""
    dy = cfg["dynamics"]
    grid = dy["sensitivity_grid"]
    n_seeds = cfg["splits"]["n_seeds"]
    prim = {lab: primary.index[primary["final_label"] == lab] for lab in ("gaining", "losing")}
    rows = []
    for p, floor, stab in product(grid["percentiles"], grid["floors"], grid["stability"]):
        labeled = {}
        for s, d in ranks.items():
            d = d.copy()
            d["label"], _, _ = label_split(d, 1 - p, p, floor, dy["require_nonzero_shap"])
            labeled[s] = d
        agg = aggregate(labeled, n_seeds, dy["min_selection_fraction"], stab)
        row = {"percentile": p, "floor": floor, "stability": stab}
        for lab in ("gaining", "losing"):
            members = agg.index[agg["final_label"] == lab]
            row[f"n_{lab}"] = len(members)
            row[f"jaccard_{lab}_vs_primary"] = _jaccard(members, prim[lab])
        row["is_primary"] = (p == dy["loss_percentile"] and floor == dy["importance_floor_rank"]
                             and stab == dy["stability_threshold"])
        rows.append(row)
    return pd.DataFrame(rows)


def label_split_ratio(df, losing_max_rel, gaining_min_rel, floor):
    """v2 per-split labels from the change in mean |SHAP| relative to the split's typical change.

    ratio = SHAP_joint / SHAP_genomics (inf when SHAP_genomics = 0); rel = ratio / median ratio over SNPs with
    SHAP_genomics > 0. Losing: rel <= losing_max_rel, rank_genomics <= floor, SHAP_genomics > 0.
    Gaining: rel >= gaining_min_rel, rank_joint <= floor, SHAP_joint > 0.
    """
    g, j = df["shap_genomics"], df["shap_joint"]
    ratio = pd.Series(np.inf, index=df.index)
    ratio[g > 0] = j[g > 0] / g[g > 0]
    med = float(ratio[g > 0].median())
    if not med > 0:
        # Rule undefined (more than half of the SNPs used by the genomics model are unused by the joint model): no SNP
        # is labelled in this split. Importances are kept.
        log.warning("median SHAP ratio is %s; v2 relative ratio undefined, no v2 labels in this split", med)
        none = pd.Series("none", index=df.index, name="label")
        return none, ratio, pd.Series(np.nan, index=df.index), med
    rel = ratio / med
    losing = (g > 0) & (rel <= losing_max_rel) & (df["rank_genomics"] <= floor)
    gaining = (j > 0) & (rel >= gaining_min_rel) & (df["rank_joint"] <= floor)
    label = pd.Series(np.select([gaining, losing], ["gaining", "losing"], "none"), index=df.index, name="label")
    return label, ratio, rel, med


def add_labels_v2(df, cfg):
    lb = cfg["posthoc_v2"]["labels"]
    out = df.copy()
    out["label"], out["shap_ratio"], out["rel_ratio"], med = label_split_ratio(
        df, lb["losing_max_rel"], lb["gaining_min_rel"], lb["importance_floor_rank"])
    stats = {"median_ratio": med, "v2_rule_defined": bool(med > 0), "n_gaining": int((out["label"] == "gaining").sum()),
             "n_losing": int((out["label"] == "losing").sum()),
             "n_zero_shap_genomics": int((df["shap_genomics"] == 0).sum()),
             "n_zero_shap_joint": int((df["shap_joint"] == 0).sum())}
    return out, stats


def sensitivity_grid_v2(ranks, cfg, primary):
    lb = cfg["posthoc_v2"]["labels"]
    grid = lb["sensitivity_grid"]
    n_seeds = cfg["splits"]["n_seeds"]
    prim = {lab: primary.index[primary["final_label"] == lab] for lab in ("gaining", "losing")}
    rows = []
    for lo, ga, floor, stab in product(grid["losing_max_rel"], grid["gaining_min_rel"], grid["floors"], grid["stability"]):
        labeled = {}
        for s, d in ranks.items():
            d = d.copy()
            d["label"], _, _, _ = label_split_ratio(d, lo, ga, floor)
            labeled[s] = d
        agg = aggregate(labeled, n_seeds, lb["min_selection_fraction"], stab)
        row = {"losing_max_rel": lo, "gaining_min_rel": ga, "floor": floor, "stability": stab}
        for lab in ("gaining", "losing"):
            members = agg.index[agg["final_label"] == lab]
            row[f"n_{lab}"] = len(members)
            row[f"jaccard_{lab}_vs_primary"] = _jaccard(members, prim[lab])
        row["is_primary"] = (lo == lb["losing_max_rel"] and ga == lb["gaining_min_rel"]
                             and floor == lb["importance_floor_rank"] and stab == lb["stability_threshold"])
        rows.append(row)
    return pd.DataFrame(rows)
