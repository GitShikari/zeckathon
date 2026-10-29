"""Assemble outputs/dashboard_data.json and re-inject it into dashboard.html.
Run after the pipeline (backtest/driver/decision/forecast) has produced its CSVs.
"""
import os, re, json
import pandas as pd

HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "processed")
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def hist(panel, cat, n=24):
    d = panel[panel.category == cat].sort_values("date").tail(n)
    return [{"date": x.date().isoformat(), "m": x.strftime("%b %y"), "net": round(v)}
            for x, v in zip(d.date, d.net_flow)]


def main():
    panel = pd.read_csv(os.path.join(PROC, "panel.csv"), parse_dates=["date"])
    fc = json.load(open(os.path.join(OUT, "forecast.json")))
    bt = pd.read_csv(os.path.join(OUT, "backtest_summary.csv"))
    dec = pd.read_csv(os.path.join(OUT, "decision_summary.csv"))
    drv = pd.read_csv(os.path.join(OUT, "driver_analysis.csv"))

    out = {"last_actual": fc["generated_last_actual"],
           "hist": {c: hist(panel, c) for c in ["liquid", "overnight"]},
           "forecast": fc["forecasts"],
           "backtest": bt.round(3).to_dict("records"),
           "decision": dec.round(1).to_dict("records"), "drivers": {}}
    for c in ["liquid", "overnight"]:
        d = drv[drv.category == c][["driver", "coef", "t", "p", "sig", "lgbm_importance"]]
        d = d[d.driver != "const"]
        out["drivers"][c] = d.round(3).astype(object).where(pd.notnull(d), 0).to_dict("records")

    data_path = os.path.join(OUT, "dashboard_data.json")
    json.dump(out, open(data_path, "w"), indent=2)

    # re-inject into the dashboard HTML if present
    html_path = os.path.join(OUT, "dashboard.html")
    if os.path.exists(html_path):
        html = open(html_path).read()
        data = open(data_path).read()
        pat = r'(<script id="data" type="application/json">).*?(</script>)'
        if re.search(pat, html, re.S):
            html = re.sub(pat, lambda m: m.group(1) + data + m.group(2), html, 1, re.S)
        else:
            html = html.replace("__DATA__", data)
        open(html_path, "w").write(html)
        print("dashboard_data.json rebuilt and injected into dashboard.html")
    else:
        print("dashboard_data.json rebuilt (dashboard.html not found)")


if __name__ == "__main__":
    main()
