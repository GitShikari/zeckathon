# Liquid-Fund Liquidity Buffer Forecast

Forecasts monthly **net flows** for India's **liquid** and **overnight** mutual-fund
categories, quantifies uncertainty, identifies drivers, and converts the forecast
into an actionable **cash-buffer policy**. Answers the brief in `PS.md`.

## Headline results (rolling-origin backtest, last 24 releases)

| | Selected model | Pinball | WAPE | 80% cov | 95% cov |
|---|---|---|---|---|---|
| Liquid | Seasonal Naive | 7,125 | 0.52 | 0.88 | 1.00 |
| Overnight | Seasonal Naive +growth | 3,282 | 0.80 | 0.75 | 0.92 |

Model is selected **per category by pinball loss**. `seasonal_naive_growth` scales the
same-month-last-year value by trailing YoY AUM growth (~10–15%/yr); it wins for Overnight
and ties for Liquid, and sharply improves the 1-step Liquid forecast (Sep 2026 error 31%→7%).

- **Drivers that matter:** advance-tax months (−₹45k cr, p<0.01) and fiscal-year-end
  (−₹44k cr) for Liquid; strong mean-reversion (prior-month flow) in both. Money-market
  rates, equity moves, and banking-system liquidity are **not** statistically significant.
- **Model choice:** classical seasonal models beat the ML challenger. The raw
  LightGBM-quantile under-calibrates badly (80% coverage ≈0.29); a **conformalized**
  variant (LightGBM-CQR, split-conformal) lifts coverage to ~0.71–0.88 but still trails
  on pinball — so the classical pick is justified against a *fairly-calibrated* challenger.
- **Decision layer:** a buffer at the **95th percentile of forecast outflow** drives
  liquidity breaches to ~0 (Liquid 8→0, Overnight 10→2 months over 24) vs the naive
  "last-year same-month outflow" baseline. Under the 1:3 cost asymmetry the cost-optimal
  buffer is the **75th** percentile; the 95th is recommended for near-zero breach risk.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py          # fetch -> features -> backtest -> drivers -> decision -> forecast
```

Outputs land in `outputs/` (CSVs + `forecast.json`). The dashboard is
`outputs/dashboard.html` (self-contained; open in a browser).

## Layout

| Path | Purpose |
|---|---|
| `src/fetch_amfi.py` | Download AMFI monthly reports, extract Liquid/Overnight flows |
| `src/features.py` | Build modeling panel: calendar + lags + repo rate + Nifty |
| `src/models.py` | Seasonal Naive, SARIMAX, ETS, LightGBM-quantile |
| `src/backtest.py` | Rolling-origin backtest: pinball, WAPE, MAPE, interval coverage |
| `src/driver_analysis.py` | OLS (HAC s.e.) + LightGBM importance |
| `src/decision.py` | Buffer rules + cost backtest vs naive baseline |
| `src/forecast.py` | Forward 3-month forecast with 80/95 intervals + buffer |
| `outputs/dashboard.html` | One-page fund-manager dashboard |

## Data sources

- **AMFI** monthly category reports (`portal.amfiindia.com/spages/am<mon><yr>repo.xls`) — target flows & AUM.
- **NSE** Nifty-50 monthly (via yfinance `^NSEI`) — equity feature.
- **RBI** repo-rate change history — hardcoded in `features.py` (step function).
- **Calendar** — advance-tax / quarter-end / FY-end flags built deterministically.

## Notes & limitations

- Sample is ~90 months; deep-learning models were not warranted. The ML challenger is
  included both raw and conformalized (CQR) for a fair, calibrated comparison.
- **Banking-system liquidity** (`sys_liq` in `features.py`) is an *approximate* monthly
  series interpolated from RBI's published half-year/quarter net-LAF anchors — used
  **only** in the driver test, never in the scored forecast. Replace `SYS_LIQ_ANCHORS`
  with an exact RBI WSS series for production.
- Decision cost uses a 1:3 idle-vs-shortfall newsvendor penalty; `decision.py` also
  reports **rupee P&L** (`pnl_cr_*`) by scaling with a 1.5%/yr carry spread (`CARRY_M`).
- Not investment advice.
