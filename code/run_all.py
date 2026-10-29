"""Run the full pipeline end-to-end:
  fetch AMFI -> build features -> backtest -> drivers -> decision -> forecast.
Then rebuild outputs/dashboard_data.json (see src/build_dashboard_data.py).
"""
import subprocess, sys, os
HERE = os.path.dirname(__file__)
for step in ["fetch_amfi.py", "features.py", "backtest.py",
             "driver_analysis.py", "decision.py", "forecast.py",
             "monthly_validate.py", "build_dashboard_data.py"]:
    print(f"\n### {step} ###")
    subprocess.run([sys.executable, os.path.join(HERE, step)], check=True)

# Event-specific buffer forecast for 15 March (the PS headline question).
print("\n### predict_event.py (15 March) ###")
subprocess.run([sys.executable, os.path.join(HERE, "predict_event.py"), "3"], check=True)
print("\nPipeline complete. Open outputs/dashboard.html (data already injected).")
