"""Table 2 (extended) from the GLGC anchors to every final v6-gaining SNP
of the v2 model (anchor, candidate, unsupported), with the same columns.

v6 rule (config posthoc_v6 thresholds, unchanged) on every eligible SNP. Per GLGC group, as in gaining_interaction.py
(Table 2 convention, BH over the tests of that group only): the Step 6a mediation scan (suppressor = per-SNP Bonferroni,
indirect and direct of opposite sign, |c'| > |c|; consistent = per-SNP Bonferroni, same sign) and the Step 6b interaction
scan over all testable metabolites; Step 6b sensitivity refits (covariate interactions, HC3, 1% trimming) for each SNP's
top interaction and every per-SNP Bonferroni interaction. The anchor rows are checked against the saved Table 2 outputs.

Writes <results_dir>/v6/gaining_all_groups/{gaining_table_all_groups.xlsx, gaining_table_all_groups.csv, bonferroni_hits.csv}.
Usage (from the repository root): python scripts/gaining_table_all_groups.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import interaction, mediation  # noqa: E402
from src.data import load_config, load_dataset, metabolite_annotation, setup_logging  # noqa: E402
from src.ols import transform_metabolites  # noqa: E402
from scripts.classify_snps import label, per_split, split_dirs  # noqa: E402

GROUPS = {"anchor": "GLGC anchor (p < 5 × 10⁻⁸)", "candidate": "GLGC candidate (5 × 10⁻⁸ ≤ p < 0.05)",
          "unsupported": "GLGC unsupported (p ≥ 0.05)", "not_in_glgc": "Not in GLGC", "no_rsid": "No rsID"}


def metname(names, mid):
    m = str(mid)
    return names.at[m, "CHEMICAL_NAME"] if m in names.index and pd.notna(names.at[m, "CHEMICAL_NAME"]) \
        else f"not in annotation table (CHEM_ID {m})"


def subpath(names, mid):
    m = str(mid)
    return names.at[m, "SUB_PATHWAY"] if m in names.index else np.nan


def verdict(r, thr):
    parts = []
    if r["n_supp"]:
        parts.append(f"suppression: {r['n_supp']} suppressor metabolites")
    if r["n_cons"]:
        parts.append(f"{r['n_cons']} consistent mediators")
    if r["n_bonf_int"]:
        ok = lambda p: "yes" if pd.notna(p) and p < thr else ("not estimable" if pd.isna(p) else "no")  # noqa: E731
        parts.append(f"{r['n_bonf_int']} interactions pass per-SNP Bonferroni (strongest still below the threshold with "
                     f"HC3: {ok(r['p_hc3'])}; without 1% tails: {ok(r['p_trim'])}; with covariate interactions: {ok(r['p_cov'])})")
    else:
        parts.append("no interaction passes per-SNP Bonferroni")
    if not r["n_supp"] and not r["n_cons"] and not r["n_bonf_int"]:
        return "no suppressor, no Bonferroni-significant interaction"
    return "; ".join(parts)

def main():
    setup_logging()
    cfg = load_config()
    ds, v6 = cfg["downstream"], cfg["posthoc_v6"]
    thr = ds["alpha"] / ds["bonferroni_per_snp"]
    base = Path(cfg["paths"]["results_dir"])
    dirs = split_dirs(base, v6["models"][0])
    t = label(per_split(dirs), len(dirs), v6["losing_max_share"], v6["gaining_min_share"], v6["min_selection_fraction"],
              v6["stability_threshold"])
    ann = pd.read_csv(base / "glgc" / "annotated_all_snps.csv", index_col="feature", dtype={"rsid": str})
    gene = ann["overlapping_genes"].fillna(ann["upstream_gene"].str.split("(").str[0] + " / " +
                                           ann["downstream_gene"].str.split("(").str[0])
    maf = pd.read_csv(base / "glgc" / "snp_maf_info.csv", index_col=0)["maf_sample"]
    t["glgc_label"] = ann["glgc_label"].reindex(t.index)
    t["retention"] = t["mean_S"] / (1 - t["mean_S"])
    gain = t[t["eligible"] & t["gaining"]]
    print("eligible", int(t["eligible"].sum()), "gaining by GLGC label:", gain["glgc_label"].value_counts().to_dict())

    data = load_dataset(cfg)
    mets_t = transform_metabolites(data.metabolites, ds["metabolite_transform"])
    names = metabolite_annotation(cfg)
    names.index = names.index.astype(str)

    rows, hits = [], []
    for g in [k for k in GROUPS if k in set(gain["glgc_label"])]:
        snps = gain.index[gain["glgc_label"] == g].tolist()
        fi, _ = interaction.run(snps, data.snps, data.y, data.covariates, mets_t, cfg)
        fm, _ = mediation.run(snps, data.snps, data.y, data.covariates, mets_t, cfg)
        SUB_TESTS[g] = f"{int(fi['estimable'].sum()):,}"
        fm = mediation.suppression_flags(fm)
        fm["consistent"] = fm["bonferroni_sig"] & (np.sign(fm["indirect"]) == np.sign(fm["c_direct"]))
        top = fi[fi["estimable"]].sort_values("p").groupby("snp").head(1).set_index("snp")
        bint = fi[fi["bonferroni_sig"]]
        sens_in = pd.concat([top.reset_index()[["snp", "metabolite"]], bint[["snp", "metabolite"]]]).drop_duplicates()
        sens = interaction.sensitivity(sens_in, data.snps, data.y, data.covariates, mets_t).set_index(["snp", "metabolite"])
        for f in snps:
            tp, sm_ = top.loc[f], fm[(fm["snp"] == f) & fm["suppression"]].sort_values("sobel_p")
            s = sens.loc[(f, tp["metabolite"])]
            r = {"group": g, "feature": f, "rsid": ann.at[f, "rsid"], "gene": gene.get(f),
                 "glgc_p": 10 ** -ann.at[f, "glgc_pvalue_neg_log10_GC"], "maf": maf.get(f),
                 "n_selected": int(gain.at[f, "n_selected"]), "stability": gain.at[f, "stability_gaining"],
                 "retention": gain.at[f, "retention"], "n_supp": int(sm_.shape[0]),
                 "n_cons": int(fm.loc[fm["snp"] == f, "consistent"].sum()),
                 "top_supp": metname(names, sm_["metabolite"].iloc[0]) if len(sm_) else np.nan,
                 "top_supp_p": sm_["sobel_p"].iloc[0] if len(sm_) else np.nan,
                 "int_met": metname(names, tp["metabolite"]), "int_sub": subpath(names, tp["metabolite"]),
                 "beta3": tp["beta3"], "p": tp["p"], "q": tp["fdr_q"],
                 "n_bonf_int": int((bint["snp"] == f).sum()), "p_hc3": s["p_hc3"], "p_trim": s["p_trim1pct_hc3"],
                 "p_cov": s["p_adj_cov_interactions"], "metabolite_id": str(tp["metabolite"])}
            r["verdict"] = verdict(r, thr)
            rows.append(r)
        for _, h in bint.iterrows():
            s = sens.loc[(h["snp"], h["metabolite"])]
            hits.append({"group": g, "rsid": ann.at[h["snp"], "rsid"], "gene": gene.get(h["snp"]), "type": "interaction",
                         "metabolite_id": str(h["metabolite"]), "metabolite": metname(names, h["metabolite"]),
                         "sub_pathway": subpath(names, h["metabolite"]), "effect": h["beta3"], "p": h["p"],
                         "fdr_q_within_group": h["fdr_q"], "p_hc3": s["p_hc3"], "p_trim1pct_hc3": s["p_trim1pct_hc3"],
                         "p_adj_cov_interactions": s["p_adj_cov_interactions"]})
        for _, h in fm[fm["suppression"] | fm["consistent"]].iterrows():
            hits.append({"group": g, "rsid": ann.at[h["snp"], "rsid"], "gene": gene.get(h["snp"]),
                         "type": "suppressor" if h["suppression"] else "consistent mediator",
                         "metabolite_id": str(h["metabolite"]), "metabolite": metname(names, h["metabolite"]),
                         "sub_pathway": subpath(names, h["metabolite"]), "effect": h["indirect"], "p": h["sobel_p"],
                         "fdr_q_within_group": h["fdr_q"], "c_total": h["c_total"], "c_direct": h["c_direct"]})
    T = pd.DataFrame(rows)
    T["_g"] = T["group"].map({k: i for i, k in enumerate(GROUPS)})
    T = T.sort_values(["_g", "retention"], ascending=[True, False]).drop(columns="_g").reset_index(drop=True)
    H = pd.DataFrame(hits)
    H["_g"] = H["group"].map({k: i for i, k in enumerate(GROUPS)})
    H = H.sort_values(["_g", "rsid", "type", "p"]).drop(columns="_g")

    # check the anchor rows against the saved Table 2 outputs (gaining_interaction.py, classify_snps.py)
    fp = pd.read_csv(base / "v6" / "gaining_interaction" / "full_primary.csv")
    t2 = fp[fp["estimable"]].sort_values("p").groupby("rsid").head(1).set_index("rsid")
    sv = pd.read_csv(base / "v6" / "v2" / "gaining.csv").set_index("rsid")
    a = T[T["group"] == "anchor"].set_index("rsid")
    assert set(a.index) == set(t2.index), (set(a.index), set(t2.index))
    assert np.allclose(a["p"], t2.loc[a.index, "p"], rtol=1e-6) and np.allclose(a["q"], t2.loc[a.index, "fdr_q"], rtol=1e-6)
    assert (a["n_supp"] == sv.loc[a.index, "med_n_bonf_suppression"]).all()
    assert np.allclose(a["retention"], sv.loc[a.index, "ratio_mean_S"])
    print("anchor rows match the saved Table 2 outputs")

    out = base / "v6" / "gaining_all_groups"
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "gaining_table_all_groups.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    H.to_csv(out / "bonferroni_hits.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    write_xlsx(T, H, out / "gaining_table_all_groups.xlsx", thr, cfg)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 45)
    print(T[["group", "rsid", "gene", "glgc_p", "retention", "n_supp", "n_cons", "int_met", "beta3", "p", "q",
             "n_bonf_int", "p_hc3", "p_trim"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(T[["rsid", "verdict"]].to_string(index=False))


COLS = [("rsid", "rsID", None, 12), ("feature", "Variation ID", None, 24), ("gene", "Gene", None, 18),
        ("glgc_p", "GLGC P", "sci", 10), ("maf", "MAF", "0.000", 7), ("n_selected", "Selected (of 20)", "0", 9),
        ("stability", "Gaining stability", "0.00", 9), ("retention", "Retention", "0.00", 9),
        ("n_supp", "Suppressor metabolites (n)", "0", 11), ("n_cons", "Consistent mediators (n)", "0", 11),
        ("top_supp", "Strongest suppressor", None, 30), ("top_supp_p", "Suppressor Sobel P", "sci", 11),
        ("int_met", "Metabolite Name (top interaction)", None, 34), ("int_sub", "Sub-pathway", None, 26),
        ("beta3", "Coef (β3)", "+0.00;−0.00", 9), ("p", "P-val", "sci", 10), ("q", "FDR q (within group)", "q", 10),
        ("n_bonf_int", "Interactions passing Bonferroni (n)", "0", 12), ("p_hc3", "P (HC3)", "sci", 10),
        ("p_trim", "P (1% trimmed, HC3)", "sci", 10), ("p_cov", "P (+ covariate interactions)", "sci", 10),
        ("verdict", "Summary", None, 60)]
HCOLS = [("group", "Group", None, 12), ("rsid", "rsID", None, 12), ("gene", "Gene", None, 18), ("type", "Type", None, 18),
         ("metabolite_id", "CHEM_ID", None, 11), ("metabolite", "Metabolite", None, 40), ("sub_pathway", "Sub-pathway", None, 26),
         ("effect", "β3 or indirect effect (mg/dL)", "+0.00;−0.00", 12), ("p", "P (interaction or Sobel)", "sci", 11),
         ("fdr_q_within_group", "FDR q (within group)", "q", 11), ("c_total", "Total effect c", "+0.00;−0.00", 10),
         ("c_direct", "Direct effect c′", "+0.00;−0.00", 10), ("p_hc3", "P (HC3)", "sci", 10),
         ("p_trim1pct_hc3", "P (1% trimmed, HC3)", "sci", 10), ("p_adj_cov_interactions", "P (+ cov. interactions)", "sci", 10)]
FILL = {"anchor": "E2EFDA", "candidate": "FFF2CC", "unsupported": "FCE4D6", "not_in_glgc": "EDEDED", "no_rsid": "EDEDED"}
THIN = Side(style="thin", color="BFBFBF")


def fmt(cell, v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        cell.value = "–"
        cell.alignment = Alignment(horizontal="center")
        return
    cell.value = v.item() if hasattr(v, "item") else v
    if f == "sci":
        cell.number_format = "0.00E+00"
    elif f == "q":
        cell.number_format = "0.00E+00" if v < 0.001 else "0.000"
    elif f:
        cell.number_format = f


def header(ws, row, cols):
    for j, (_, h, _, w) in enumerate(cols, 1):
        c = ws.cell(row, j, h)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="44546A")
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[row].height = 45


def write_xlsx(T, H, path, thr, cfg):
    wb = Workbook()
    ws = wb.active
    ws.title = "Table2_extended"
    ws.cell(1, 1, "Gaining-importance SNPs (final share-based label, v2 models), by GLGC support").font = Font(bold=True, size=12)
    header(ws, 2, COLS)
    r = 3
    for g, sub in T.groupby("group", sort=False):
        c = ws.cell(r, 1, f"{GROUPS[g]} — {len(sub)} SNP{'s' if len(sub) > 1 else ''}; FDR q over the {sub_tests(sub)} interaction tests of this group")
        c.font = Font(bold=True)
        for j in range(1, len(COLS) + 1):
            ws.cell(r, j).fill = PatternFill("solid", fgColor=FILL[g])
        r += 1
        for _, x in sub.iterrows():
            for j, (k, _, f, _) in enumerate(COLS, 1):
                cell = ws.cell(r, j)
                fmt(cell, x[k], f)
                cell.border = Border(bottom=THIN)
                if k == "verdict":
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            if x["n_bonf_int"] or x["n_supp"]:
                ws.cell(r, 1).font = Font(bold=True)
            r += 1
    ws.freeze_panes = "D3"

    wh = wb.create_sheet("Bonferroni_hits")
    wh.cell(1, 1, f"Every per-SNP Bonferroni hit (p < 0.05/{cfg['downstream']['bonferroni_per_snp']} = {thr:.2e}) "
                  "for the SNPs above: interactions, suppressors, consistent mediators").font = Font(bold=True, size=12)
    header(wh, 2, HCOLS)
    for i, (_, x) in enumerate(H.iterrows(), 3):
        for j, (k, _, f, _) in enumerate(HCOLS, 1):
            fmt(wh.cell(i, j), x.get(k, np.nan), f)
        for j in range(1, len(HCOLS) + 1):
            wh.cell(i, j).fill = PatternFill("solid", fgColor=FILL[x["group"]])
    wh.freeze_panes = "D3"

    wn = wb.create_sheet("Notes")
    notes = [
        ("SNPs", "Final gaining SNPs of the v2 models (config posthoc_v6 rule) on every eligible SNP, grouped by GLGC support: "
                 "anchor (p < 5e-8; manuscript Table 2), candidate (5e-8 <= p < 0.05), unsupported (p >= 0.05)."),
        ("Check", "The anchor rows are asserted to equal the saved outputs of scripts/gaining_interaction.py and "
                  "scripts/classify_snps.py."),
        ("Retention", "mean S / (1 - mean S): joint-model importance divided by genomics-only importance."),
        ("Gaining stability", "Share of the splits where the SNP was selected in which it was labelled gaining (S > 0.5); "
                              "final label if >= 0.70."),
        ("Suppressor metabolites (n)", "Sobel p < 0.05/1257, indirect and direct effects of opposite sign, |c'| > |c|."),
        ("Consistent mediators (n)", "Sobel p < 0.05/1257, indirect and direct effects of the same sign."),
        ("Metabolite Name / Coef / P-val", "Metabolite with the lowest interaction p-value; beta3 in mg/dL LDL-C per allele per "
                                            "SD of log metabolite."),
        ("FDR q (within group)", "Benjamini-Hochberg over the interaction tests of the SNPs in the group."),
        ("P (HC3) / trimmed / covariate interactions", "Step 6b sensitivity refits of the strongest interaction of each SNP; "
                                                       "the Summary column states whether each p stays below 0.05/1257."),
        ("Bonferroni_hits", "Every per-SNP Bonferroni interaction, suppressor and consistent mediator."),
        ("Metabolite names", "'not in annotation table (CHEM_ID ...)': measured metabolite without a row in the annotation file."),
        ("Note", "SNPs of the unsupported group have no external GLGC support; their results are exploratory."),
    ]
    wn.column_dimensions["A"].width = 30
    wn.column_dimensions["B"].width = 120
    for i, (k, v) in enumerate(notes, 1):
        wn.cell(i, 1, k).font = Font(bold=True)
        wn.cell(i, 2, v).alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def sub_tests(sub):
    return SUB_TESTS.get(sub["group"].iloc[0], "?")


SUB_TESTS = {}

if __name__ == "__main__":
    main()
