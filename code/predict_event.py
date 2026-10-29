"""Predict the cash a fund must hold for a specific advance-tax EVENT month
(15 Mar / 15 Jun / 15 Sep / 15 Dec). Analogous to the PS's "15 March" question,
but for any target month. Forecasts the NEXT occurrence of that month after the
latest actual, with point + 80/95% intervals and the recommended buffer.

Usage:  python predict_event.py            # defaults to March (the PS question)
        python predict_event.py 9          # September
        python predict_event.py 9 12 6 3   # several events
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from models import MODELS

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
QS = [0.025, 0.10, 0.50, 0.90, 0.975]


def next_occurrence(last_date, target_month):
    """First 1st-of-month on/after the month AFTER last_date whose month==target."""
    d = (last_date + pd.DateOffset(months=1)).replace(day=1)
    while d.month != target_month:
        d += pd.DateOffset(months=1)
    return d


def forecast_month(df, target_date):
    """Forecast net flow for target_date (seasonal-naive primary, SARIMAX check).
    Horizon = months from last actual. Seasonal-naive uses same-month-last-year;
    SARIMAX forecasts multi-step to the target."""
    last = df["date"].max()
    h = (target_date.year - last.year) * 12 + (target_date.month - last.month)

    # --- seasonal naive: same calendar month, most recent available year ---
    same_month = df[df["date"].dt.month == target_date.month].sort_values("date")
    point_sn = float(same_month["net_flow"].iloc[-1])
    errs = (df["net_flow"] - df["net_flow"].shift(12)).dropna()
    sigma_sn = float(errs.std()) if len(errs) > 3 else float(df["net_flow"].std())

    # --- SARIMAX multi-step to horizon h ---
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        y = df.sort_values("date")["net_flow"].astype(float).values
        m = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = m.get_forecast(steps=h)
        point_sx = float(fc.predicted_mean[-1])
        sigma_sx = float(np.sqrt(fc.var_pred_mean[-1]))
    except Exception:
        point_sx, sigma_sx = point_sn, sigma_sn

    def pack(point, sigma):
        q = {p: point + norm.ppf(p) * sigma for p in QS}
        return {
            "point": point, "sigma": sigma,
            "lo80": q[0.10], "hi80": q[0.90], "lo95": q[0.025], "hi95": q[0.975],
            # buffer = cash covering the (1-p) percentile of OUTFLOW
            "buffer_p75": max(0.0, -(point + norm.ppf(0.25) * sigma)),
            "buffer_p95": max(0.0, -(point + norm.ppf(0.05) * sigma)),
            "buffer_p99": max(0.0, -(point + norm.ppf(0.01) * sigma)),
        }

    return h, {"seasonal_naive": pack(point_sn, sigma_sn),
               "sarimax": pack(point_sx, sigma_sx)}


def main():
    targets = [int(x) for x in sys.argv[1:]] or [3]   # default March (PS headline)
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    last = panel["date"].max()
    result = {"as_of": str(last.date()), "events": []}

    for tm in targets:
        tdate = next_occurrence(last, tm)
        ev = {"event": f"15 {MONTHS[tm-1]} {tdate.year}", "target_month": tm,
              "target_date": tdate.date().isoformat(), "categories": {}}
        print(f"\n{'='*66}\n  CASH TO HOLD ON 15 {MONTHS[tm-1].upper()} {tdate.year}"
              f"   (forecast as of {MONTHS[last.month-1][:3]} {last.year})\n{'='*66}")
        for cat in ["liquid", "overnight"]:
            d = panel[panel.category == cat].sort_values("date").reset_index(drop=True)
            h, fcs = forecast_month(d, tdate)
            ev["categories"][cat] = {"horizon_months": h, **fcs}
            rec = fcs["sarimax"]  # selected model
            direction = "net OUTFLOW" if rec["point"] < 0 else "net inflow"
            print(f"\n  {cat.upper():9s} (+{h}m ahead)  —  forecast net flow: "
                  f"{rec['point']:>+12,.0f} cr  ({direction})")
            print(f"    80% interval : [{rec['lo80']:>12,.0f}, {rec['hi80']:>12,.0f}]")
            print(f"    95% interval : [{rec['lo95']:>12,.0f}, {rec['hi95']:>12,.0f}]")
            print(f"    >> BUFFER to hold (95th-pct outflow): "
                  f"₹{rec['buffer_p95']:>12,.0f} cr   (p75 ₹{rec['buffer_p75']:,.0f} · p99 ₹{rec['buffer_p99']:,.0f})")
            sn = fcs["seasonal_naive"]
            print(f"    [seasonal-naive cross-check: point {sn['point']:>+12,.0f} · buffer(p95) ₹{sn['buffer_p95']:,.0f}]")
        result["events"].append(ev)

    json.dump(result, open(os.path.join(OUT, "event_forecast.json"), "w"), indent=2)
    print(f"\nSaved -> outputs/event_forecast.json")


if __name__ == "__main__":
    main()
