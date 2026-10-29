"""Build the monthly modeling panel: AMFI flows + calendar/event features
+ equity (Nifty) + RBI repo rate.

Target variable: net_flow (INR crore) per category (liquid / overnight).
All driver features are KNOWN at forecast time (calendar) or lagged (market),
so the panel is safe for out-of-sample forecasting.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
RAW = os.path.join(HERE, "..", "data", "raw")

# RBI repo rate history (effective date -> rate %). Source: RBI monetary policy
# decisions. Step function; we forward-fill to month-start.
REPO_HISTORY = {
    "2019-01-01": 6.50, "2019-02-07": 6.25, "2019-04-04": 6.00,
    "2019-06-06": 5.75, "2019-08-07": 5.40, "2019-10-04": 5.15,
    "2020-03-27": 4.40, "2020-05-22": 4.00,
    "2022-05-04": 4.40, "2022-06-08": 4.90, "2022-08-05": 5.40,
    "2022-09-30": 5.90, "2022-12-07": 6.25, "2023-02-08": 6.50,
    "2024-10-09": 6.50,  # held
    "2025-02-07": 6.25, "2025-04-09": 6.00, "2025-06-06": 5.50,
    "2025-08-06": 5.50,
}


# Banking-system net liquidity (surplus +, deficit -), INR lakh crore.
# APPROXIMATE monthly series interpolated from RBI published anchors (half-year /
# quarter average net LAF absorption & WSS figures). Provenance below; used ONLY
# as an exploratory driver, never in the scored forecast.
#   2020-21 COVID surplus ~+5-6; 2022 narrowing; 2023-24 deficit; 2024 deep deficit;
#   H2 FY25-26 avg +1.43; H1 FY26-27 avg +3.21; 18-Oct-2026 ~+1.86.
SYS_LIQ_ANCHORS = {
    "2019-04-01": -1.0, "2019-10-01": 0.5, "2020-04-01": 4.0, "2020-09-01": 5.5,
    "2021-03-01": 6.0, "2021-09-01": 7.5, "2022-03-01": 6.5, "2022-09-01": 2.5,
    "2023-03-01": 1.0, "2023-09-01": -0.5, "2024-01-01": -1.8, "2024-06-01": -1.0,
    "2024-12-01": -2.0, "2025-03-01": -1.3, "2025-09-01": 1.0, "2026-03-01": 1.8,
    "2026-09-01": 2.8,
}


def sys_liquidity_series(index):
    a = pd.Series(SYS_LIQ_ANCHORS); a.index = pd.to_datetime(a.index)
    a = a.sort_index()
    s = a.reindex(a.index.union(index)).interpolate("time").reindex(index)
    return s.ffill().bfill()


def repo_rate_series(index):
    hist = pd.Series(REPO_HISTORY)
    hist.index = pd.to_datetime(hist.index)
    hist = hist.sort_index()
    # value effective at each month start = last change on/before that month
    out = []
    for d in index:
        out.append(hist[hist.index <= d].iloc[-1] if (hist.index <= d).any()
                   else hist.iloc[0])
    return pd.Series(out, index=index)


def calendar_features(df):
    m = df["date"].dt.month
    df["month"] = m
    # Advance-tax / quarter-end outflow months (Jun, Sep, Dec, Mar)
    df["advance_tax"] = m.isin([3, 6, 9, 12]).astype(int)
    df["fy_end"] = (m == 3).astype(int)          # fiscal year-end (strongest)
    df["q_end_h2"] = m.isin([9, 12]).astype(int)  # H2 quarter ends
    df["april_rebound"] = (m == 4).astype(int)    # cash returns post FY-end
    # cyclical encodings
    df["sin_m"] = np.sin(2 * np.pi * m / 12)
    df["cos_m"] = np.cos(2 * np.pi * m / 12)
    return df


def build():
    flows = pd.read_csv(os.path.join(PROC, "amfi_flows.csv"), parse_dates=["date"])
    nifty = pd.read_csv(os.path.join(RAW, "nifty_monthly.csv"), parse_dates=["date"])

    panels = []
    for cat in flows["category"].unique():
        d = flows[flows["category"] == cat].sort_values("date").copy()
        d = calendar_features(d)
        # market features (lagged -> known at forecast time)
        d = d.merge(nifty, on="date", how="left")
        d["nifty_ret_1m"] = d["nifty_close"].pct_change()
        d["nifty_ret_3m"] = d["nifty_close"].pct_change(3)
        d["repo_rate"] = repo_rate_series(d["date"]).values
        d["repo_chg_3m"] = d["repo_rate"].diff(3)
        # approximate system liquidity (surplus+/deficit-), exploratory driver only
        d["sys_liq"] = sys_liquidity_series(d["date"]).values
        d["sys_liq_chg_3m"] = d["sys_liq"].diff(3)
        # autoregressive features
        d["net_lag1"] = d["net_flow"].shift(1)
        d["net_lag12"] = d["net_flow"].shift(12)   # same month last year
        d["aum_lag1"] = d["aum_eom"].shift(1)
        # scale target by prior AUM -> flow rate (more stationary); keep raw too
        d["flow_rate"] = d["net_flow"] / d["aum_lag1"]
        panels.append(d)

    panel = pd.concat(panels, ignore_index=True)
    out = os.path.join(PROC, "panel.csv")
    os.makedirs(PROC, exist_ok=True)
    panel.to_csv(out, index=False)
    print(f"Saved panel {panel.shape} -> {out}")
    print("features:", [c for c in panel.columns])
    return panel


if __name__ == "__main__":
    build()
