"""Vectorized mediation / interaction scans must equal statsmodels OLS. Run: python tests/test_ols.py"""
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import interaction, mediation, metabolome_mediation  # noqa: E402


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, 12))
    g = rng.binomial(2, 0.3, n).astype(float)
    M = rng.normal(size=(n, 5)) + 0.4 * g[:, None] + C[:, :5] * 0.2
    y = 0.3 * g + 0.5 * M[:, 0] + 0.2 * g * M[:, 1] + C @ rng.normal(size=12) + rng.normal(size=n)
    return g, y, C, M


def test_mediation_matches_statsmodels():
    g, y, C, M = _data()
    res = mediation.scan(g, y, C, M)
    one = np.ones((len(y), 1))
    tot = sm.OLS(y, np.hstack([one, C, g[:, None]])).fit()
    for j in range(M.shape[1]):
        fa = sm.OLS(M[:, j], np.hstack([one, C, g[:, None]])).fit()
        fb = sm.OLS(y, np.hstack([one, g[:, None], C, M[:, [j]]])).fit()
        r = res.iloc[j]
        assert np.isclose(r["a"], fa.params[-1]) and np.isclose(r["se_a"], fa.bse[-1])
        assert np.isclose(r["p_a"], fa.pvalues[-1])
        assert np.isclose(r["b"], fb.params[-1]) and np.isclose(r["se_b"], fb.bse[-1])
        assert np.isclose(r["p_b"], fb.pvalues[-1])
        assert np.isclose(r["c_total"], tot.params[-1]) and np.isclose(r["se_c_total"], tot.bse[-1])
        assert np.isclose(r["c_direct"], fb.params[1])


def test_interaction_matches_statsmodels():
    g, y, C, M = _data(seed=1)
    res = interaction.scan(g, y, C, M)
    one = np.ones((len(y), 1))
    for j in range(M.shape[1]):
        f = sm.OLS(y, np.hstack([one, g[:, None], C, M[:, [j]], (g * M[:, j])[:, None]])).fit()
        r = res.iloc[j]
        assert np.isclose(r["beta3"], f.params[-1]) and np.isclose(r["se3"], f.bse[-1])
        assert np.isclose(r["p"], f.pvalues[-1]) and np.isclose(r["beta_metabolite"], f.params[-2])


def test_metabolome_scan_matches_statsmodels():
    import pandas as pd
    g, y, C, M = _data(seed=2)
    rng = np.random.default_rng(3)
    G = pd.DataFrame(np.column_stack([g, rng.binomial(2, 0.2, len(y)), rng.binomial(2, 0.4, len(y))]).astype(float))
    m = M[:, 0]
    res = metabolome_mediation.scan(G, y, m, C, chunk=2)
    one = np.ones((len(y), 1))
    for j in range(G.shape[1]):
        x = G.iloc[:, [j]].values
        tot = sm.OLS(y, np.hstack([one, C, x])).fit()
        fa = sm.OLS(m, np.hstack([one, C, x])).fit()
        fb = sm.OLS(y, np.hstack([one, C, x, m[:, None]])).fit()
        r = res.iloc[j]
        assert np.isclose(r["c_total"], tot.params[-1]) and np.isclose(r["p_c"], tot.pvalues[-1])
        assert np.isclose(r["a"], fa.params[-1]) and np.isclose(r["se_a"], fa.bse[-1])
        assert np.isclose(r["c_direct"], fb.params[-2]) and np.isclose(r["se_c_direct"], fb.bse[-2])
        assert np.isclose(r["b"], fb.params[-1]) and np.isclose(r["se_b"], fb.bse[-1])
        assert np.isclose(r["c_total"], r["c_direct"] + r["a"] * r["b"])


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
