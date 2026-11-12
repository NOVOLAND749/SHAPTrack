"""Share-based losing / gaining labels (config share_labels; specified before running, after the pre-specified
rank-based rules labelled no SNP).

Re-labels the SAVED per-split SHAP importances (no refit). Per split and pre-selected SNP: S = I_joint / (I_joint + I_g)
(undefined, no label, when both are 0); split label losing if S < losing_max_share (1/3), gaining if S > gaining_min_share
(1/2). Across splits: eligible if pre-selected in >= min_selection_fraction of the splits; final label if the same label
in >= stability_threshold of the splits where selected. GLGC sets: primary = anchors, secondary = candidates.
Validation against the OLS mediation (metabolome score, single-metabolite scan of the anchors).

Writes <results_dir>/labels/<models>/{snp_table.csv, losing.csv, gaining.csv} and <results_dir>/labels/summary.json.
Usage (from the repository root): python scripts/classify_snps.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config  # noqa: E402
from src.snpinfo import allele_table  # noqa: E402


def split_dirs(base, model):
    """Per-split output folders of a model set: <results_dir>/<models>/splits (models: main or prespecified)."""
    d = base / model / "splits"
    return sorted(d.glob("seed_*"))


def per_split(dirs):
    rows = []
    for d in dirs:
        x = pd.read_csv(d / "dynamics.csv", index_col="feature")
        tot = x["shap_joint"] + x["shap_genomics"]
        rows.append(pd.DataFrame({"S": (x["shap_joint"] / tot).where(tot > 0), "seed": d.name}))
    return pd.concat(rows)


def label(long, n_splits, lose_max, gain_min, min_frac, stab):
    g = long.groupby(level=0)
    t = pd.DataFrame({"n_selected": g.size(), "n_S_defined": g["S"].count(),
                      "n_losing": g["S"].apply(lambda s: int((s < lose_max).sum())),
                      "n_gaining": g["S"].apply(lambda s: int((s > gain_min).sum())),
                      "mean_S": g["S"].mean(), "median_S": g["S"].median()})
    t["eligible"] = t["n_selected"] >= min_frac * n_splits
    t["stability_losing"] = t["n_losing"] / t["n_selected"]
    t["stability_gaining"] = t["n_gaining"] / t["n_selected"]
    t["losing"] = t["eligible"] & (t["stability_losing"] >= stab)
    t["gaining"] = t["eligible"] & (t["stability_gaining"] >= stab)
    return t


def main():
    cfg = load_config()
    sl = cfg["share_labels"]
    base = Path(cfg["paths"]["results_dir"])
    out = base / sl["results_subdir"]
    ann = pd.read_csv(base / "glgc" / "annotated_all_snps.csv", index_col="feature", dtype={"rsid": str})
    gene = ann["overlapping_genes"].fillna(ann["upstream_gene"].str.split("(").str[0] + " / " +
                                           ann["downstream_gene"].str.split("(").str[0])
    sets = {name: set(ann.index[ann["glgc_label"].isin(labs)]) for name, labs in sl["sets"].items()}
    score = pd.read_csv(base / cfg["metabolome_score"]["results_subdir"] / "all" / "scan.csv", index_col="feature")
    main = base / cfg["main_models"]["results_subdir"]
    med = pd.read_csv(main / "anchor_mediation" / "full.csv")
    tested = set(med["snp"])
    hit = med[med["bonferroni_sig"]]
    same = np.sign(hit["indirect"]) == np.sign(hit["c_direct"])
    supp = (~same) & (hit["c_direct"].abs() > hit["c_total"].abs())
    n_same, n_supp = hit[same].groupby("snp").size(), hit[supp].groupby("snp").size()
    ratio_gain = set(pd.read_csv(main / "aggregate" / "gaining.csv", index_col="feature").index)
    summary = {"k": cfg["prescreen"]["k"], "spec": sl, "models": {}}

    for model in sl["models"]:
        dirs = split_dirs(base, model)
        long = per_split(dirs)
        long = long[long.index.isin(set().union(*sets.values()))]
        t = label(long, len(dirs), sl["losing_max_share"], sl["gaining_min_share"], sl["min_selection_fraction"],
                  sl["stability_threshold"])
        t["set"] = ["primary" if f in sets["primary"] else "secondary" for f in t.index]
        sens = {}
        for st in sl["sensitivity"]["stability"]:
            s = label(long, len(dirs), sl["losing_max_share"], sl["gaining_min_share"], sl["min_selection_fraction"], st)
            sens[f"stability{st}"] = s
        for lm in sl["sensitivity"]["losing_max_share"]:
            s = label(long, len(dirs), lm, sl["gaining_min_share"], sl["min_selection_fraction"], sl["stability_threshold"])
            sens[f"losing_share{lm:.3f}"] = s
        for k, s in sens.items():
            t[f"sens_{k}_losing"], t[f"sens_{k}_gaining"] = s["losing"].reindex(t.index), s["gaining"].reindex(t.index)
        t["label"] = np.where(t["losing"], "losing", np.where(t["gaining"], "gaining", "none"))
        t["ratio_mean_S"] = t["mean_S"] / (1 - t["mean_S"])
        t["rsid"], t["gene"], t["glgc_label"] = ann["rsid"].reindex(t.index), gene.reindex(t.index), \
            ann["glgc_label"].reindex(t.index)
        t["score_prop_mediated"] = score["prop_mediated"].reindex(t.index)
        t["score_losing"] = score["losing"].reindex(t.index).fillna(False).astype(bool)
        t["med_tested"] = t.index.isin(tested)
        t["med_n_bonf_same_dir"] = n_same.reindex(t.index).fillna(0).astype(int)
        t["med_n_bonf_suppression"] = n_supp.reindex(t.index).fillna(0).astype(int)
        t["ratio_rule_gaining"] = t.index.isin(ratio_gain)
        t = t.join(allele_table(t.index, base))

        o = out / model
        o.mkdir(parents=True, exist_ok=True)
        t.sort_values(["label", "mean_S"]).to_csv(o / "snp_table.csv")
        for lab in ("losing", "gaining"):
            t[t[lab] & t["eligible"]].sort_values("mean_S", ascending=lab == "losing").to_csv(o / f"{lab}.csv")

        res = {"splits": len(dirs)}
        for k in ("primary", "secondary"):
            s = t[(t["set"] == k)]
            e = s[s["eligible"]]
            res[k] = {"eligible": int(len(e)), "losing": int(e["losing"].sum()), "gaining": int(e["gaining"].sum()),
                      "losing_rsids": e.loc[e["losing"], "rsid"].fillna("no rsID").tolist(),
                      "gaining_rsids": e.loc[e["gaining"], "rsid"].fillna("no rsID").tolist(),
                      "sensitivity": {kk: {"losing": int(e[f"sens_{kk}_losing"].sum()),
                                           "gaining": int(e[f"sens_{kk}_gaining"].sum())} for kk in sens}}
        a = t[(t["set"] == "primary") & t["eligible"]]
        frac = lambda x, c: float((x.loc[x["med_tested"], c] > 0).mean()) if x["med_tested"].any() else None  # noqa: E731
        L, Lo, G, Go = a[a["losing"]], a[~a["losing"]], a[a["gaining"]], a[~a["gaining"]]
        mw = lambda x, y: float(stats.mannwhitneyu(x.dropna(), y.dropna()).pvalue) if len(x.dropna()) and len(y.dropna()) else None  # noqa: E731
        res["validation_primary"] = {
            "losing_vs_other": {"n": [int(len(L)), int(len(Lo))],
                                "median_score_prop_mediated": [float(L["score_prop_mediated"].median()) if len(L) else None,
                                                            float(Lo["score_prop_mediated"].median())],
                                "mann_whitney_p": mw(L["score_prop_mediated"], Lo["score_prop_mediated"]),
                                "share_with_same_dir_mediator": [frac(L, "med_n_bonf_same_dir") if len(L) else None,
                                                                 frac(Lo, "med_n_bonf_same_dir")],
                                "score_losing_in_losing": int(L["score_losing"].sum())},
            "gaining_vs_other": {"n": [int(len(G)), int(len(Go))],
                                 "median_score_prop_mediated": [float(G["score_prop_mediated"].median()) if len(G) else None,
                                                             float(Go["score_prop_mediated"].median())],
                                 "share_with_suppression_mediator": [frac(G, "med_n_bonf_suppression") if len(G) else None,
                                                                     frac(Go, "med_n_bonf_suppression")],
                                 "ratio_rule_gaining_in_gaining": int(G["ratio_rule_gaining"].sum())}}
        summary["models"][model] = res
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["models"], indent=1))


if __name__ == "__main__":
    main()
