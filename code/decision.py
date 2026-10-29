"""Deliverable 04 — cash-buffer decision layer + cost backtest.

Buffer rule: hold cash = 95th percentile of the forecast OUTFLOW distribution.
  Using the selected model's predictive distribution of net_flow,
  B_t = -(forecast net_flow quantile at p=0.05) = -point + 1.645*sigma.
  (Outflow exceeds B only 5% of the time.)

Cost model (PS: "yield given up on idle cash + a 3x penalty on any shortfall").
  This is a newsvendor cost with a 1:3 asymmetry (under-provisioning is 3x worse):
  realised_outflow O_t = max(0, -actual_net_flow)
  cost_t = 1 * max(0, B_t - O_t)  +  3 * max(0, O_t - B_t)      [INR crore units]
  The cost-minimising buffer under a 1:c asymmetry is the c/(1+c) quantile of the
  outflow distribution -> here 3/4 = the 75th percentile. We report BOTH the
  PS example rule (95th pct) and the cost-optimal quantile, vs the naive baseline.

Baseline (naive) rule: B_t = max(0, -net_lag12)  (last year's same-month outflow).
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from models import MODELS, QUANTILES

HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")

SELECTED = {"liquid": "sarimax", "overnight": "sarimax"}
SHORTFALL_PENALTY = 3.0
# Rupee P&L scaling: carry spread given up on idle cash (deployable yield minus
# cash yield). ~1.5% annualised -> monthly. Shortfall charged at 3x this rate.
CARRY_M = 0.015 / 12


def sigma_from_quantiles(qp):
    return (qp[0.975] - qp[0.025]) / (2 * norm.ppf(0.975))


def cost(buffer, outflow):
    idle = max(0.0, buffer - outflow)
    short = max(0.0, outflow - buffer)
    return idle + SHORTFALL_PENALTY * short, short


def buffer_at(point, sigma, p):
    """Buffer covering the (1-p) quantile of outflow = -(p-quantile of net_flow)."""
    return max(0.0, -(point + norm.ppf(p) * sigma))


def run(panel, category, n_test=24):
    df = panel[panel["category"] == category].sort_values("date").reset_index(drop=True)
    fn = MODELS[SELECTED[category]]
    # candidate buffer quantiles (of outflow): p is the net_flow lower-tail prob
    grid = {"p75": 0.25, "p90": 0.10, "p95": 0.05, "p99": 0.01}
    recs = []
    for i in range(len(df) - n_test, len(df)):
        train, row = df.iloc[:i], df.iloc[i]
        point, qp = fn(train, row)
        sigma = sigma_from_quantiles(qp)
        outflow = max(0.0, -row["net_flow"])
        naive_buf = max(0.0, -row["net_lag12"]) if not pd.isna(row["net_lag12"]) else 0.0
        rec = {"date": row["date"], "actual_net": row["net_flow"], "outflow": outflow,
               "buffer_naive": naive_buf}
        rec["cost_naive"], rec["short_naive"] = cost(naive_buf, outflow)
        for name, p in grid.items():
            b = buffer_at(point, sigma, p)
            rec[f"buffer_{name}"] = b
            rec[f"cost_{name}"], rec[f"short_{name}"] = cost(b, outflow)
        recs.append(rec)
    r = pd.DataFrame(recs)
    rules = list(grid) + ["naive"]
    summ = {"category": category}
    for name in rules:
        summ[f"cost_{name}"] = r[f"cost_{name}"].sum()
        summ[f"nshort_{name}"] = int((r[f"short_{name}"] > 0).sum())
        summ[f"avgbuf_{name}"] = r[f"buffer_{name}"].mean()
    best = min(grid, key=lambda n: r[f"cost_{n}"].sum())
    summ["best_rule"] = best
    summ["cost_reduction_vs_naive_%"] = 100 * (1 - r[f"cost_{best}"].sum() / r["cost_naive"].sum())
    # Rupee P&L (INR crore) over the 24 months: unit cost scaled by carry.
    for name in rules:
        summ[f"pnl_cr_{name}"] = summ[f"cost_{name}"] * CARRY_M
    return r, summ


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    os.makedirs(OUT, exist_ok=True)
    summ_rows = []
    for cat in ["liquid", "overnight"]:
        r, s = run(panel, cat)
        r.to_csv(os.path.join(OUT, f"decision_{cat}.csv"), index=False)
        summ_rows.append(s)
        print(f"\n===== {cat.upper()} — buffer cost backtest (24m, 1:3 asymmetry) =====")
        for k, v in s.items():
            print(f"  {k:26s}: {v:,.1f}" if isinstance(v, float) else f"  {k:26s}: {v}")
    pd.DataFrame(summ_rows).to_csv(os.path.join(OUT, "decision_summary.csv"), index=False)
    print("\ncost_* in INR crore (idle=1x, shortfall=3x). nshort = # months with a shortfall.")


if __name__ == "__main__":
    main()
