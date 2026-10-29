"""Deliverable 01 + 05 inputs — 3-month forecast with 80/95 intervals, plus the
recommended cash buffer. Writes outputs/forecast.json for the dashboard.

Targets Jul / Aug / Sep of the latest year for which those months exist; the model
is trained ONLY on data before the first target month (no leakage), so the forecasts
can be compared against the actuals that printed for those months.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

from models import MODELS

HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)
# Selected per category by best backtest pinball loss.
SELECTED = {"liquid": "sarimax", "overnight": "sarimax"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TARGET_MONTHS = [1, 2, 3]   # Jan, Feb, Mar (the PS's "15 March" quarter)


def _target_dates(panel):
    """Latest year in which all TARGET_MONTHS are present; return their 1st-of-month dates."""
    dates = pd.to_datetime(panel["date"]).dt.to_period("M")
    for yr in sorted(panel["date"].dt.year.unique(), reverse=True):
        wanted = [pd.Period(f"{yr}-{m:02d}", "M") for m in TARGET_MONTHS]
        if all(w in set(dates) for w in wanted):
            return [w.to_timestamp() for w in wanted]
    raise ValueError("target months not found in data")


def _sarimax_multistep(train, h):
    """Proper multi-step SARIMAX forecast: fit once, forecast h steps.
    Returns list of (point, sigma) for horizons 1..h."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = train["net_flow"].astype(float).values
    m = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12), trend="c",
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = m.get_forecast(steps=h)
    return [(float(fc.predicted_mean[i]), float(np.sqrt(fc.var_pred_mean[i]))) for i in range(h)]


def forward(panel, category, targets):
    df = panel[panel["category"] == category].sort_values("date").reset_index(drop=True)
    model_name = SELECTED[category]
    fn = MODELS[model_name]
    cutoff = targets[0]                         # first target month
    train = df[df["date"] < cutoff]             # strictly before -> no leakage

    # SARIMAX is a true multi-step model (widens with horizon natively);
    # lag-based models are direct per-month forecasts.
    sx_steps = _sarimax_multistep(train, len(targets)) if model_name == "sarimax" else None

    rows = []
    prev_flow = train["net_flow"].iloc[-1]       # last actual before first target
    for k, fdate in enumerate(targets, start=1):
        lag12_date = fdate - pd.DateOffset(months=12)
        lag12 = train.loc[train["date"] == lag12_date, "net_flow"]
        row = {
            "net_lag12": float(lag12.iloc[0]) if len(lag12) else train["net_flow"].tail(12).mean(),
            "net_lag1": float(prev_flow),        # recursive: prior forecast for k>1
            "advance_tax": int(fdate.month in (3, 6, 9, 12)),
            "fy_end": int(fdate.month == 3),
        }
        if sx_steps is not None:
            point, sigma_h = sx_steps[k - 1]     # native multi-step point + sigma
        else:
            point, qp = fn(train, pd.Series(row))
            prev_flow = point                    # feed forward for next horizon
            sigma = (qp[0.975] - qp[0.025]) / (2 * norm.ppf(0.975))
            sigma_h = sigma * np.sqrt(k) ** 0.5  # widen modestly with horizon
        q = {p: point + norm.ppf(p) * sigma_h for p in [0.025, 0.10, 0.50, 0.90, 0.975]}
        outflow_p95 = max(0.0, -q[0.05] if 0.05 in q else -(point + norm.ppf(0.05) * sigma_h))
        buf_p95 = max(0.0, -(point + norm.ppf(0.05) * sigma_h))
        buf_p75 = max(0.0, -(point + norm.ppf(0.25) * sigma_h))
        # actual, if it exists in the data (for forecast-vs-actual comparison)
        act = df.loc[df["date"] == fdate, "net_flow"]
        actual = float(act.iloc[0]) if len(act) else None
        rows.append({
            "category": category,
            "month": f"{MONTHS[fdate.month-1]} {fdate.year}",
            "date": fdate.date().isoformat(),
            "point": point,
            "lo80": q[0.10], "hi80": q[0.90],
            "lo95": q[0.025], "hi95": q[0.975],
            "buffer_p75": buf_p75, "buffer_p95": buf_p95,
            "actual": actual,
            "abs_pct_err": (abs((actual - point) / actual) if actual else None),
        })
    return rows


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    targets = _target_dates(panel)
    out = {"generated_last_actual": str(panel["date"].max().date()),
           "trained_through": str((targets[0] - pd.DateOffset(months=1)).date()),
           "forecasts": {}}
    for cat in ["liquid", "overnight"]:
        rows = forward(panel, cat, targets)
        out["forecasts"][cat] = rows
        print(f"\n===== {cat.upper()} — forecast {rows[0]['month']}–{rows[-1]['month']} "
              f"(trained thru {out['trained_through'][:7]}) =====")
        for r in rows:
            av = f"{r['actual']:>12,.0f}" if r['actual'] is not None else "         n/a"
            pe = f"{r['abs_pct_err']*100:5.1f}%" if r['abs_pct_err'] is not None else "  n/a"
            print(f"  {r['month']}: point {r['point']:>12,.0f}  actual {av}  err {pe}  "
                  f"80%[{r['lo80']:>11,.0f},{r['hi80']:>11,.0f}]  buf95 {r['buffer_p95']:>11,.0f}")
    with open(os.path.join(OUT, "forecast.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> outputs/forecast.json")


if __name__ == "__main__":
    main()
