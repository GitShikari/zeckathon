"""Rolling-origin backtest over the last N monthly releases.
Refit every model at each origin; forecast 1 step ahead; score.

Metrics:
  - WAPE  : sum|e| / sum|y|   (robust when target crosses zero)
  - MAPE  : mean|e/y|         (reported per PS; guarded for |y|<floor)
  - MAE   : mean|e|           (INR crore)
  - pinball: mean pinball loss across the 5 quantiles (primary prob. metric)
  - cov80 / cov95 : share of actuals inside the 80% / 95% intervals
"""
import os
import numpy as np
import pandas as pd

from models import MODELS, QUANTILES

HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")

MAPE_FLOOR = 10000.0  # INR crore; below this |actual|, skip MAPE term (ill-defined)


def pinball(actual, qpreds):
    losses = []
    for q in QUANTILES:
        pred = qpreds[q]
        d = actual - pred
        losses.append(max(q * d, (q - 1) * d))
    return np.mean(losses)


def backtest(panel, category, n_test=24):
    df = panel[panel["category"] == category].sort_values("date").reset_index(drop=True)
    origins = range(len(df) - n_test, len(df))
    records = []
    for name, fn in MODELS.items():
        for i in origins:
            train = df.iloc[:i]
            row = df.iloc[i]
            actual = row["net_flow"]
            point, qp = fn(train, row)
            records.append({
                "model": name, "date": row["date"], "actual": actual,
                "point": point,
                "q10": qp[0.10], "q90": qp[0.90],
                "q025": qp[0.025], "q975": qp[0.975],
                "ae": abs(actual - point),
                "ape": abs((actual - point) / actual) if abs(actual) > MAPE_FLOOR else np.nan,
                "pinball": pinball(actual, qp),
                "in80": int(qp[0.10] <= actual <= qp[0.90]),
                "in95": int(qp[0.025] <= actual <= qp[0.975]),
            })
    res = pd.DataFrame(records)
    summ = (res.groupby("model")
            .apply(lambda g: pd.Series({
                "WAPE": g["ae"].sum() / g["actual"].abs().sum(),
                "MAPE": g["ape"].mean(),
                "MAE_cr": g["ae"].mean(),
                "pinball": g["pinball"].mean(),
                "cov80": g["in80"].mean(),
                "cov95": g["in95"].mean(),
            }), include_groups=False)
            .sort_values("pinball"))
    return res, summ


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    os.makedirs(OUT, exist_ok=True)
    all_summ = []
    for cat in ["liquid", "overnight"]:
        res, summ = backtest(panel, cat)
        res.to_csv(os.path.join(OUT, f"backtest_{cat}.csv"), index=False)
        summ_r = summ.round(4)
        summ_r.insert(0, "category", cat)
        all_summ.append(summ_r.reset_index())
        print(f"\n===== {cat.upper()} — rolling-origin backtest (last 24 months) =====")
        print(summ_r.to_string())
    pd.concat(all_summ).to_csv(os.path.join(OUT, "backtest_summary.csv"), index=False)
    print("\nTarget 80% coverage ~0.80, 95% coverage ~0.95. Lower pinball/WAPE better.")


if __name__ == "__main__":
    main()
