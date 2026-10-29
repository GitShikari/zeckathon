# Liquidity Buffer Forecast — Hackathon Submission

**How much cash must a liquid fund hold on 15 March?** A reproducible monthly net-flow
forecast for India's **liquid** and **overnight** fund categories, with calibrated
uncertainty, an evidence-backed driver analysis, model benchmarking, and an actionable
cash-buffer policy centred on the advance-tax / fiscal-year-end event.

---

## Deliverables map

| # | Deliverable | Where to find it |
|---|---|---|
| **01** | **Forecast model** — monthly net flows (Jan/Feb/**Mar**) + 80% & 95% intervals | `code/forecast.py` → `results/forecast.json`; `dashboards/dashboard.html` |
| **02** | **Driver analysis** — calendar, rates, liquidity, equity | `code/driver_analysis.py` → `results/driver_analysis.csv` |
| **03** | **Model benchmarking** — classical vs ML, out-of-sample | `code/backtest.py` → `results/backtest_summary.csv` |
| **04** | **Decision layer** — buffer rule + cost vs naive baseline | `code/decision.py` → `results/decision_summary.csv` |
| **05** | **Fund-manager dashboard** — one page | `dashboards/dashboard.html` |
| — | **Forecast vs actual** (validation) | `dashboards/monthly_report.html`; `results/monthly.json` |
| ★ | **"Cash on 15 March" event forecast** (the PS headline) | `code/predict_event.py` → `results/event_forecast.json` |
| Tie-break | **Live next-release forecast** | `results/forecast.json` |

---

## Headline answer: cash to hold on 15 March

Forecast for **15 March 2027** (as of the Sep 2026 release), selected model:

| Category | Forecast net flow | 95% interval | **Buffer to hold (95th-pct outflow)** |
|---|---|---|---|
| **Liquid** | −₹134,988 cr (outflow) | [−229,312, −40,664] | **₹214,147 cr** |
| **Overnight** | −₹40,228 cr (outflow) | [−75,163, −5,293] | **₹69,546 cr** |

Validation on the last complete March (Mar 2026, trained through Dec 2025):
Liquid forecast −₹123,645 vs actual −₹134,988 → **8.4% error**, actual inside the 80% band.

## Selected model: Growth + AR(1) ensemble (both categories)
2:1 blend of AUM-growth-scaled seasonal naive + AR(1)+calendar regression. Chosen per
category by out-of-sample pinball loss (rolling-origin over last 24 releases, refit each step).

| Category | WAPE | Pinball | 80% cov | 95% cov |
|---|---|---|---|---|
| Liquid | 0.46 | 6,291 | 83% | 100% |
| Overnight | 0.73 | 3,033 | 75% | 96% |

Benchmarked vs Seasonal Naive, SARIMAX, ETS, LightGBM (raw + conformalized CQR), and
custom trained models (Ridge/ElasticNet/stacking) — all lose on this ~90-month series.

## Drivers (Deliverable 02)
Advance-tax months (Liquid −₹45k cr, p<0.01) and **FY-end March** (−₹44k cr) dominate,
plus strong month-to-month mean-reversion. Rates, equity, and system liquidity: not significant.

## Decision layer (Deliverable 04)
Buffer = 95th-pct of forecast outflow. Cost = idle carry (1×) + shortfall (3×). Vs the naive
"last-year same-month outflow" baseline, the forecast buffer cuts liquidity breaches to near
zero (Liquid 8→0, Overnight 10→2 over 24 months).

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code && python run_all.py
```

Runs **fully offline** — raw inputs bundled in `data/raw/` (AMFI reports + Nifty). Delete
`data/raw/` to force a fresh download (needs internet). Outputs land in `../outputs/`.
Open `dashboards/dashboard.html` (self-contained). `predict_event.py 9` → September, etc.

Data sources: AMFI (target flows & AUM), NSE (Nifty-50), RBI (repo rate), Income-Tax
advance-tax calendar. Not investment advice.
