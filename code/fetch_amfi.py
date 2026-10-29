"""Download AMFI monthly 'resource mobilization' reports and extract
Liquid Fund and Overnight Fund monthly flows (inflow / outflow / net / AUM).

Source URL pattern: https://portal.amfiindia.com/spages/am<mon><year>repo.xls
"""
import io
import os
import sys
import time
import warnings
import datetime as dt

import requests
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "amfi_flows.csv")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Category names have drifted over time; match on normalised substrings.
TARGETS = {
    "overnight": "overnight fund",
    "liquid": "liquid fund",
}

COLS = {4: "inflow", 5: "outflow", 6: "net_flow", 7: "aum_eom", 8: "aaum"}


def url_for(year, mon_idx):
    return f"https://portal.amfiindia.com/spages/am{MONTHS[mon_idx]}{year}repo.xls"


def download(year, mon_idx):
    os.makedirs(RAW_DIR, exist_ok=True)
    fn = os.path.join(RAW_DIR, f"am{MONTHS[mon_idx]}{year}.xls")
    if os.path.exists(fn) and os.path.getsize(fn) > 5000:
        return fn
    url = url_for(year, mon_idx)
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        print(f"  ! {url} -> {e}")
        return None
    if r.status_code != 200 or len(r.content) < 5000:
        return None
    with open(fn, "wb") as f:
        f.write(r.content)
    return fn


def parse(fn, year, mon_idx):
    try:
        df = pd.ExcelFile(fn).parse(0, header=None)
    except Exception as e:
        print(f"  ! parse {fn}: {e}")
        return []
    rows = []
    for _, row in df.iterrows():
        name = str(row.iloc[1]).strip().lower()
        for key, needle in TARGETS.items():
            # exact-ish: the scheme name IS the category (avoid matching sub-totals)
            if name == needle:
                rec = {"category": key,
                       "year": year, "month": mon_idx + 1,
                       "date": dt.date(year, mon_idx + 1, 1).isoformat()}
                for ci, cn in COLS.items():
                    try:
                        rec[cn] = float(row.iloc[ci])
                    except (ValueError, TypeError):
                        rec[cn] = None
                rows.append(rec)
    return rows


def main():
    start_year, start_mon = 2019, 3  # Apr 2019 (overnight category well-established)
    today = dt.date.today()
    all_rows = []
    y, m = start_year, start_mon
    while (y, m) <= (today.year, today.month - 1):  # last completed month
        fn = download(y, m)
        if fn:
            recs = parse(fn, y, m)
            all_rows.extend(recs)
            tag = "ok" if recs else "no-target-rows"
            print(f"{y}-{m+1:02d}: {tag} ({len(recs)} rows)")
        else:
            print(f"{y}-{m+1:02d}: MISSING")
        time.sleep(0.3)
        m += 1
        if m == 12:
            y, m = y + 1, 0
    out = pd.DataFrame(all_rows).sort_values(["category", "date"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT}")
    print(out.groupby("category")["date"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
