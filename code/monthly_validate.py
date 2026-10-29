"""Predict MARCH (the PS's "15 March" month): train on all months before the
latest complete March, forecast it, compare point + 80/95% intervals to actual.
Writes outputs/monthly.json for the forecast-vs-actual webpage.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

from models import MODELS

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)
SHOW = ["seasonal_naive", "seasonal_naive_growth", "ens_growth_ar1",
        "ar1_calendar", "sarimax", "ets", "lgbm_cqr"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TARGET_MONTH = 3   # March — the PS headline month


def validate_month(df, fn, target_date):
    """Train on all rows strictly before target_date; forecast that month."""
    train = df[df["date"] < target_date]
    row = df[df["date"] == target_date].iloc[0]
    point, qp = fn(train, row)
    actual = float(row["net_flow"])
    return {
        "actual": actual, "point": float(point),
        "err": actual - float(point),
        "ape": abs((actual - float(point)) / actual) if actual else None,
        "lo80": float(qp[0.10]), "hi80": float(qp[0.90]),
        "lo95": float(qp[0.025]), "hi95": float(qp[0.975]),
        "in80": bool(qp[0.10] <= actual <= qp[0.90]),
        "in95": bool(qp[0.025] <= actual <= qp[0.975]),
    }


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    bt = pd.read_csv(os.path.join(OUT, "backtest_summary.csv"))
    # latest complete March in the data
    march_dates = panel.loc[panel["date"].dt.month == TARGET_MONTH, "date"]
    target = march_dates.max()
    label = f"{MONTHS[target.month-1]} {target.year}"
    out = {"month": label, "validation": {}, "models": {}}
    for cat in ["liquid", "overnight"]:
        d = panel[panel.category == cat].sort_values("date").reset_index(drop=True)
        print(f"\n########## {cat.upper()} — validate {label} (train thru Feb {target.year}) ##########")
        for name in SHOW:
            v = validate_month(d, MODELS[name], target)
            out["validation"].setdefault(cat, {})[name] = v
            r = bt[(bt.category == cat) & (bt.model == name)]
            out["models"].setdefault(cat, {})[name] = {
                "WAPE": float(r.WAPE.iloc[0]) if len(r) else None,
                "pinball": float(r.pinball.iloc[0]) if len(r) else None,
            }
            ap = f"{v['ape']*100:5.1f}%" if v["ape"] is not None else "  n/a"
            print(f"[{name:15s}] forecast {v['point']:>12,.0f}  actual {v['actual']:>12,.0f}  "
                  f"err {v['err']:>+11,.0f} ({ap})  in80 {v['in80']}  in95 {v['in95']}")
    json.dump(out, open(os.path.join(OUT, "monthly.json"), "w"), indent=2)
    print("\nSaved -> outputs/monthly.json")


if __name__ == "__main__":
    main()
