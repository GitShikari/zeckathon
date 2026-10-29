"""Deliverable 02 — which drivers actually matter.
Two complementary views:
  (a) OLS with HAC (Newey-West) standard errors -> signed, significance-tested
      effect of each driver on net_flow.
  (b) LightGBM gain importance -> non-linear predictive contribution.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")

DRIVERS = ["advance_tax", "fy_end", "q_end_h2", "april_rebound",
           "nifty_ret_1m", "nifty_ret_3m", "repo_rate", "repo_chg_3m",
           "sys_liq", "sys_liq_chg_3m", "net_lag1", "net_lag12"]


def ols_view(df):
    import statsmodels.api as sm
    d = df.dropna(subset=DRIVERS + ["net_flow"])
    X = sm.add_constant(d[DRIVERS].astype(float))
    y = d["net_flow"].astype(float)
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out = pd.DataFrame({"coef": m.params, "t": m.tvalues, "p": m.pvalues})
    out["sig"] = out["p"].apply(lambda p: "***" if p < .01 else "**" if p < .05
                                else "*" if p < .1 else "")
    return out.round(2), m.rsquared


def lgbm_view(df):
    import lightgbm as lgb
    d = df.dropna(subset=DRIVERS + ["net_flow"])
    mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                            min_child_samples=5, verbose=-1)
    mdl.fit(d[DRIVERS], d["net_flow"])
    imp = pd.Series(mdl.booster_.feature_importance("gain"),
                    index=DRIVERS).sort_values(ascending=False)
    return (imp / imp.sum()).round(3)


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cat in ["liquid", "overnight"]:
        d = panel[panel["category"] == cat]
        ols, r2 = ols_view(d)
        imp = lgbm_view(d)
        print(f"\n===== {cat.upper()} — OLS drivers (HAC SE), R^2={r2:.2f} =====")
        print(ols.to_string())
        print(f"\n{cat.upper()} — LightGBM gain importance (share):")
        print(imp.to_string())
        o = ols.copy(); o["category"] = cat; o["driver"] = o.index
        o["lgbm_importance"] = o["driver"].map(imp)
        rows.append(o)
    pd.concat(rows).to_csv(os.path.join(OUT, "driver_analysis.csv"), index=False)
    print("\nSaved -> outputs/driver_analysis.csv")


if __name__ == "__main__":
    main()
