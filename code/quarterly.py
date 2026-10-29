"""Quarterly view: aggregate monthly net flows to calendar quarters (which align
with the advance-tax / fiscal quarter-ends), train on all quarters up to the last
COMPLETE quarter, forecast that quarter, and compare point + intervals to actual.
Also emits a forward forecast for the next (in-progress) quarter.

Quarterly seasonal lag = 4 (same quarter last year).
"""
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")
QS = [0.025, 0.10, 0.50, 0.90, 0.975]


def to_quarterly(df):
    """Sum monthly net flow per calendar quarter; keep only complete (3-month) quarters."""
    d = df.set_index("date").sort_index()
    g = d["net_flow"].resample("QE").agg(["sum", "count"])
    g = g[g["count"] == 3]                 # complete quarters only
    out = g["sum"].rename("net_flow").to_frame()
    out["q"] = out.index.to_period("Q").astype(str)
    return out


def seasonal_naive_q(train):
    """Forecast = same quarter last year (lag 4). sigma from lag-4 errors."""
    point = float(train["net_flow"].iloc[-4])
    errs = (train["net_flow"] - train["net_flow"].shift(4)).dropna()
    sigma = float(errs.std()) if len(errs) > 3 else float(train["net_flow"].std())
    return point, sigma


def sarimax_q(train):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = train["net_flow"].astype(float).values
    try:
        m = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.get_forecast(1)
        return float(fc.predicted_mean[0]), float(np.sqrt(fc.var_pred_mean[0]))
    except Exception:
        return seasonal_naive_q(train)


def quants(point, sigma):
    return {q: point + norm.ppf(q) * sigma for q in QS}


def validate_last_quarter(q, model_fn):
    """Train on all but the last complete quarter; forecast it; compare to actual."""
    train, test = q.iloc[:-1], q.iloc[-1]
    point, sigma = model_fn(train)
    qp = quants(point, sigma)
    actual = float(test["net_flow"])
    return {
        "quarter": test["q"], "actual": actual, "point": point,
        "err": actual - point, "ape": abs((actual - point) / actual) if actual else np.nan,
        "lo80": qp[0.10], "hi80": qp[0.90], "lo95": qp[0.025], "hi95": qp[0.975],
        "in80": bool(qp[0.10] <= actual <= qp[0.90]),
        "in95": bool(qp[0.025] <= actual <= qp[0.975]),
    }


def rolling_q(q, model_fn, n=8):
    recs = []
    for i in range(len(q) - n, len(q)):
        tr, te = q.iloc[:i], q.iloc[i]
        p, s = model_fn(tr)
        a = float(te["net_flow"])
        recs.append({"quarter": te["q"], "actual": a, "point": p, "ae": abs(a - p),
                     "in80": bool(p + norm.ppf(.1) * s <= a <= p + norm.ppf(.9) * s)})
    r = pd.DataFrame(recs)
    return {"WAPE": r["ae"].sum() / r["actual"].abs().sum(),
            "MAE_cr": r["ae"].mean(), "cov80": r["in80"].mean()}


def forward_quarter(q, model_fn):
    """Forecast the NEXT quarter after the last complete one (in-progress quarter)."""
    point, sigma = model_fn(q)
    nxt = (pd.Period(q["q"].iloc[-1], "Q") + 1)
    qp = quants(point, sigma)
    buf95 = max(0.0, -(point + norm.ppf(0.05) * sigma))
    return {"quarter": str(nxt), "point": point, "lo80": qp[0.10], "hi80": qp[0.90],
            "lo95": qp[0.025], "hi95": qp[0.975], "buffer_p95": buf95}


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    allout = {"models": {}, "forward": {}, "validation": {}}
    for cat in ["liquid", "overnight"]:
        q = to_quarterly(panel[panel.category == cat][["date", "net_flow"]])
        print(f"\n########## {cat.upper()}  (quarterly, {q.index[0].to_period('Q')} … {q.index[-1].to_period('Q')}) ##########")
        for name, fn in [("seasonal_naive_q", seasonal_naive_q), ("sarimax_q", sarimax_q)]:
            v = validate_last_quarter(q, fn)
            roll = rolling_q(q, fn)
            print(f"\n[{name}]  validate {v['quarter']}:")
            print(f"   forecast {v['point']:>12,.0f}   actual {v['actual']:>12,.0f}   "
                  f"error {v['err']:>+11,.0f}  ({v['ape']*100:4.1f}% )")
            print(f"   80% PI [{v['lo80']:>11,.0f}, {v['hi80']:>11,.0f}]  actual in 80%? {v['in80']}   in 95%? {v['in95']}")
            print(f"   rolling 8q: WAPE {roll['WAPE']:.2f}  MAE {roll['MAE_cr']:,.0f} cr  cov80 {roll['cov80']:.2f}")
            allout["validation"].setdefault(cat, {})[name] = v
            allout["models"].setdefault(cat, {})[name] = roll
        fwd = forward_quarter(q, seasonal_naive_q)
        allout["forward"][cat] = fwd
        print(f"\n>>> FORWARD forecast {fwd['quarter']} (in-progress): "
              f"point {fwd['point']:,.0f}  80%[{fwd['lo80']:,.0f}, {fwd['hi80']:,.0f}]  "
              f"buffer(p95) {fwd['buffer_p95']:,.0f}")
        q.to_csv(os.path.join(OUT, f"quarterly_{cat}.csv"))
    import json
    json.dump(allout, open(os.path.join(OUT, "quarterly.json"), "w"), indent=2, default=str)
    print("\nSaved -> outputs/quarterly.json , outputs/quarterly_<cat>.csv")


if __name__ == "__main__":
    main()
