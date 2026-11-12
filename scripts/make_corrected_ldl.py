"""Corrected medication-adjusted LDL-C for the re-run (writes paths.outcome_file; never modifies the input data).

The pkl's LDL had two unit errors for participants on lipid-lowering drugs: TC converted with the triglyceride factor
(88.57) instead of 38.67, and ln HDL / ln TG entered the Friedewald formula. Here, all in mg/dL:
  untreated (LIPID_MED_baseline FALSE): LDL = measured LDL-C
  treated   (LIPID_MED_baseline TRUE):  LDL = TC / tc_med_factor - HDL - TG / 5
Checks: every analysis participant gets a value; untreated values equal the pkl's LDL exactly.

Usage (from the repository root): python scripts/make_corrected_ldl.py
"""
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_config, load_dataset, setup_logging  # noqa: E402

log = logging.getLogger("make_corrected_ldl")
LIPIDS = ["BLD_CHOL_COM", "BLD_HDL_COM", "BLD_TRIG_COM", "BLD_LDL_COM"]


def corrected_table(cfg):
    """Corrected LDL-C (mg/dL) for every participant with a GWAS id in the CLSA baseline file."""
    paths, oc, id_col = cfg["paths"], cfg["outcome_correction"], cfg["columns"]["id"]
    chol, trig, factor = oc["mmol_to_mgdl_cholesterol"], oc["mmol_to_mgdl_triglyceride"], oc["tc_med_factor"]
    base = pd.read_csv(paths["clsa_baseline"], usecols=["entity_id", id_col] + LIPIDS, dtype={id_col: str})
    base = base.dropna(subset=[id_col])
    med = pd.read_csv(paths["lipid_med"], usecols=["entity_id", "LIPID_MED_baseline"])
    d = base.merge(med, on="entity_id", how="left").set_index(id_col)
    if d.index.duplicated().any():
        raise ValueError(f"{d.index.duplicated().sum()} duplicated {id_col} in {paths['clsa_baseline']}")
    for c in LIPIDS:                                   # mmol/L; anything outside (0, 50) is a missing code
        bad = ~d[c].between(0, 50, inclusive="neither")
        d.loc[bad, c] = np.nan
    out = pd.DataFrame(index=d.index)
    out["entity_id"] = d["entity_id"]
    out["lipid_med"] = d["LIPID_MED_baseline"]
    out["ldl_measured"] = d["BLD_LDL_COM"] * chol
    out["tc"] = d["BLD_CHOL_COM"] * chol
    out["hdl"] = d["BLD_HDL_COM"] * chol
    out["tg"] = d["BLD_TRIG_COM"] * trig
    treated = out["lipid_med"].eq(True)
    out["LDL_corrected"] = np.where(treated, out["tc"] / factor - out["hdl"] - out["tg"] / 5, out["ldl_measured"])
    out.loc[out["lipid_med"].isna(), "LDL_corrected"] = np.nan
    out["tg_over_400"] = out["tg"] > 400
    return out


def main():
    setup_logging()
    cfg = load_config()
    paths, id_col = cfg["paths"], cfg["columns"]["id"]
    out = corrected_table(cfg)

    # Check against the analysis sample and the pkl's (uncorrected) outcome.
    cfg0 = copy.deepcopy(cfg)
    cfg0["paths"].pop("outcome_file")
    y_old = load_dataset(cfg0).y
    sub = out.reindex(y_old.index)
    if sub["LDL_corrected"].isna().any():
        raise ValueError(f"{int(sub['LDL_corrected'].isna().sum())} analysis participants lack a corrected LDL")
    t = sub["lipid_med"].eq(True)
    diff_untreated = (sub.loc[~t, "LDL_corrected"] - y_old[~t]).abs().max()
    if diff_untreated > 1e-3:
        raise ValueError(f"untreated participants differ from the pkl LDL (max abs diff {diff_untreated})")
    sub = sub.assign(LDL_previous=y_old)
    log.info("analysis sample n = %d: treated %d, untreated %d (untreated identical to pkl LDL, max diff %.2g)",
             len(sub), int(t.sum()), int((~t).sum()), diff_untreated)
    for name, s in [("previous", sub["LDL_previous"]), ("corrected", sub["LDL_corrected"])]:
        log.info("%-9s all: median %.1f mean %.1f sd %.1f min %.1f max %.1f >300: %d | treated median %.1f",
                 name, s.median(), s.mean(), s.std(), s.min(), s.max(), int((s > 300).sum()), s[t].median())
    log.info("treated: measured LDL median %.1f; TG > 400 mg/dL (Friedewald outside its range) %d; corrected <= 0: %d",
             sub.loc[t, "ldl_measured"].median(), int(sub.loc[t, "tg_over_400"].sum()),
             int((sub["LDL_corrected"] <= 0).sum()))

    dest = Path(paths["outcome_file"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.index.name = id_col
    sub.to_csv(dest)
    log.info("wrote %s (%d rows)", dest, len(sub))


if __name__ == "__main__":
    main()
