"""Step 2c: SHAP on test rows and mean |SHAP| per feature."""
import numpy as np
import pandas as pd
import shap


def shap_values(model, X, feature_perturbation="tree_path_dependent"):
    explainer = shap.TreeExplainer(model, feature_perturbation=feature_perturbation)
    return np.asarray(explainer.shap_values(X))


def mean_abs_shap(values, columns):
    return pd.Series(np.abs(values).mean(axis=0), index=columns, name="mean_abs_shap")


def importance_table(imp, snp_prefix):
    out = imp.rename_axis("feature").reset_index()
    out["type"] = np.where(out["feature"].str.startswith(snp_prefix), "snp", "metabolite")
    return out.sort_values("mean_abs_shap", ascending=False, kind="stable")
