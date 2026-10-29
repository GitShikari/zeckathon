FORECASTING • LIQUIDITY RISK • DECISION SCIENCE

## How Much Cash Must a Liquid Fund Hold on 15 March?

Build a reliable monthly flow forecast for India’s liquid and overnight mutual fund categories—and turn it into a practical cash-buffer policy that balances return drag against redemption risk.

## THE CHALLENGE

Forecast the next 3 months of net flows, quantify uncertainty, identify meaningful drivers, and recommend a buffer the fund manager can act on.

## Business context

Corporate treasuries park surplus cash in liquid funds and redeem in large bursts around quarter-end advance-tax dates (15 June, 15 September, 15 December and 15 March), GST payments, and quarter-end balance-sheet dressing. Fund managers must choose between cash and overnight instruments—which are liquid but may yield less—and higher-yielding T-bills and commercial paper. Too little liquidity can force distress selling; too much idle cash can dilute returns.

The CIO needs a defensible number—not just a narrative: how much liquidity should the fund hold, how uncertain is the forecast, and what does the policy cost under historical conditions?

## What teams must deliver

| ID | DELIVERABLE | REQUIREMENT |
| --- | --- | --- |
| 01 | Forecast model | Forecast monthly net flows for liquid and overnight funds for the next 3 months. Provide |
|   |   | point forecasts and 80% and 95% prediction intervals. |
| 02 | Driver analysis | Test calendar effects (advance tax, quarter-end and fiscal year-end), money-market rates, |
|   |   | banking-system liquidity surplus/deficit, and equity-market moves. Show which drivers |
|   |   | matter in practice. |
| 03 | Model benchmarking | Compare at least one classical model (seasonal naive, SARIMAX or ETS) with at least |
|   |   | one ML or deep-learning model. Justify the selected model using out-of-sample results. |
| 04 | Decision layer | Propose a cash-buffer rule—for example, cash equal to the 95th percentile of forecast |
|   |   | outflow—and backtest its cost against a simple baseline. |
| 05 | Fund-manager | Create a one-page dashboard suitable for use on the first day of each month, showing |
|   | dashboard | forecasts, uncertainty, key drivers and the recommended buffer. |

## Suggested data sources

| AMFI | Monthly category-wise flow and AUM reports. |
| --- | --- |
| RBI DBIE | Repo / overnight rates and liquidity adjustment facility data. |
| CCIL & FBIL | Treasury-bill and commercial-paper yield data, where available. |
| NSE | Equity index levels and market-move features. |


| Income Tax Department | Advance-tax calendar dates for known event features. |
| --- | --- |

## Evaluation and scoring

| METRIC | HOW IT IS MEASURED |
| --- | --- |
| Forecast accuracy | Rolling-origin backtest over the last 24 monthly releases. Refit at each step; report MAPE and |
|   | pinball loss. |
| Interval calibration | Share of actual observations contained within the 80% and 95% prediction intervals. |
| Decision cost | Yield given up on idle cash + a 3× penalty on any redemption shortfall. Compare with the naive |
|   | rule: hold last year’s same-month outflow. |
| Live forecast (tie-breaker) | Submit a forecast for the next AMFI release; it will be scored once the release is published. |

## WHAT SUCCESS LOOKS LIKE

A reproducible forecast, calibrated uncertainty, evidence-backed driver selection, and a cash-buffer recommendation whose historical cost is clearly compared with the baseline.
