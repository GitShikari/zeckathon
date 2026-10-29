"""Forecast models. Each returns point forecast + quantile forecasts for a
set of quantiles, given a training frame and the feature row(s) to predict.

Quantiles used: 0.025, 0.10, 0.50, 0.90, 0.975
-> gives 80% interval [0.10,0.90] and 95% interval [0.025,0.975].
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

QUANTILES = [0.025, 0.10, 0.50, 0.90, 0.975]
FEATURES = ["advance_tax", "fy_end", "q_end_h2", "april_rebound",
            "sin_m", "cos_m", "nifty_ret_1m", "nifty_ret_3m",
            "repo_rate", "repo_chg_3m", "net_lag1", "net_lag12"]


def _normal_intervals(point, sigma):
    from scipy.stats import norm
    return {q: point + norm.ppf(q) * sigma for q in QUANTILES}


# ---------------------------------------------------------------- seasonal naive
def seasonal_naive(train, target_row):
    """Forecast = same calendar month last year. Intervals from the empirical
    distribution of seasonal-naive errors on the training history."""
    point = target_row["net_lag12"]
    if pd.isna(point):
        point = train["net_flow"].tail(12).mean()
    errs = (train["net_flow"] - train["net_lag12"]).dropna()
    sigma = errs.std() if len(errs) > 3 else train["net_flow"].std()
    return point, _normal_intervals(point, sigma)


# -------------------------------------------------- growth-scaled seasonal naive
def _aum_growth(train):
    """YoY growth of AUM using the most recent 13 months of train (factor ~1.1)."""
    aum = train["aum_eom"].dropna()
    if len(aum) >= 13 and aum.iloc[-13] > 0:
        g = float(aum.iloc[-1] / aum.iloc[-13])
        # guard against outliers
        return min(max(g, 0.5), 2.0)
    return 1.0


def seasonal_naive_growth(train, target_row):
    """Same calendar month last year, scaled by YoY AUM growth. Captures the
    level/trend that plain seasonal naive misses (book grows ~10-15%/yr).
    Intervals from the empirical errors of the growth-scaled rule itself."""
    g = _aum_growth(train)
    point = target_row["net_lag12"]
    if pd.isna(point):
        point = train["net_flow"].tail(12).mean()
    point = point * g
    # residuals of the growth-scaled rule across history (one-step, g per origin)
    t = train.reset_index(drop=True)
    preds, acts = [], []
    for i in range(13, len(t)):
        aum_prev, aum_prev13 = t["aum_eom"].iloc[i - 1], t["aum_eom"].iloc[i - 13]
        gi = (aum_prev / aum_prev13) if aum_prev13 and aum_prev13 > 0 else 1.0
        gi = min(max(float(gi), 0.5), 2.0)
        lag12 = t["net_flow"].iloc[i - 12]
        preds.append(lag12 * gi)
        acts.append(t["net_flow"].iloc[i])
    if len(acts) > 3:
        sigma = float(np.std(np.array(acts) - np.array(preds)))
    else:
        sigma = float(train["net_flow"].std())
    return point, _normal_intervals(point, sigma)


# ----------------------------------------------- AR(1) + calendar regression
AR1_FEATURES = ["net_lag1", "advance_tax", "fy_end"]


def ar1_calendar(train, target_row):
    """OLS: net_flow ~ prior-month flow + advance-tax + FY-end. Captures the
    strong month-to-month mean-reversion seen in the driver analysis."""
    import statsmodels.api as sm
    t = train.dropna(subset=AR1_FEATURES + ["net_flow"])
    if len(t) < 12 or pd.isna(target_row.get("net_lag1")):
        return seasonal_naive(train, target_row)
    X = sm.add_constant(t[AR1_FEATURES].astype(float))
    m = sm.OLS(t["net_flow"].astype(float), X).fit()
    xr = pd.DataFrame([[1.0] + [float(target_row[f]) for f in AR1_FEATURES]],
                      columns=["const"] + AR1_FEATURES)
    point = float(m.predict(xr)[0])
    sigma = float(np.std(m.resid)) if len(m.resid) > 3 else float(train["net_flow"].std())
    return point, _normal_intervals(point, sigma)


def ens_growth_ar1(train, target_row):
    """2:1 weighted ensemble of growth-scaled seasonal naive and AR(1)+calendar.
    Best backtest model for the (noisy, mean-reverting) overnight series."""
    p1, q1 = seasonal_naive_growth(train, target_row)
    p2, q2 = ar1_calendar(train, target_row)
    point = (2 * p1 + p2) / 3
    # blend sigma via interval half-widths (same 2:1 weighting)
    s1 = (q1[0.975] - q1[0.025]) / (2 * 1.959963985)
    s2 = (q2[0.975] - q2[0.025]) / (2 * 1.959963985)
    sigma = (2 * s1 + s2) / 3
    return point, _normal_intervals(point, sigma)


# ---------------------------------------------------------------- SARIMAX
def sarimax(train, target_row):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = train["net_flow"].astype(float).values
    try:
        m = SARIMAX(y, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12),
                    trend="c", enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        fc = m.get_forecast(steps=1)
        point = float(fc.predicted_mean[0])
        sigma = float(np.sqrt(fc.var_pred_mean[0]))
    except Exception:
        return seasonal_naive(train, target_row)
    return point, _normal_intervals(point, sigma)


# ---------------------------------------------------------------- ETS
def ets(train, target_row):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = train["net_flow"].astype(float)
    try:
        m = ExponentialSmoothing(y, trend="add", seasonal="add",
                                 seasonal_periods=12).fit()
        point = float(m.forecast(1).iloc[0])
        resid = m.resid
        sigma = float(np.std(resid)) if len(resid) > 3 else float(y.std())
    except Exception:
        return seasonal_naive(train, target_row)
    return point, _normal_intervals(point, sigma)


# ---------------------------------------------------------------- LightGBM quantile
def lgbm_quantile(train, target_row):
    import lightgbm as lgb
    tr = train.dropna(subset=FEATURES + ["net_flow"])
    if len(tr) < 24:
        return seasonal_naive(train, target_row)
    X, y = tr[FEATURES], tr["net_flow"]
    Xp = pd.DataFrame([target_row[FEATURES]])
    preds = {}
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=7,
                  min_child_samples=5, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1)
    for q in QUANTILES:
        mdl = lgb.LGBMRegressor(objective="quantile", alpha=q, **params)
        mdl.fit(X, y)
        preds[q] = float(mdl.predict(Xp)[0])
    # enforce monotone quantiles
    vals = np.sort([preds[q] for q in QUANTILES])
    preds = {q: vals[i] for i, q in enumerate(QUANTILES)}
    return preds[0.50], preds


# ---------------------------------------------------------------- LightGBM + CQR
def lgbm_cqr(train, target_row):
    """Conformalized Quantile Regression (Romano et al. 2019).
    Split train into proper-fit (first 80%) and calibration (last 20%, most
    recent). Fit quantile models on the proper set; widen each interval by the
    empirical quantile of calibration conformity scores so coverage is honest.
    """
    import lightgbm as lgb
    tr = train.dropna(subset=FEATURES + ["net_flow"]).reset_index(drop=True)
    if len(tr) < 36:
        return lgbm_quantile(train, target_row)
    cut = int(len(tr) * 0.8)
    fit, cal = tr.iloc[:cut], tr.iloc[cut:]
    Xf, yf = fit[FEATURES], fit["net_flow"]
    Xc, yc = cal[FEATURES], cal["net_flow"].values
    Xp = pd.DataFrame([target_row[FEATURES]])
    params = dict(n_estimators=250, learning_rate=0.03, num_leaves=7,
                  min_child_samples=5, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, verbose=-1)

    def q_model(alpha):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        m.fit(Xf, yf)
        return m

    preds = {}
    # point (median)
    preds[0.50] = float(q_model(0.50).predict(Xp)[0])
    # conformalize each interval level separately
    for lvl, (lo_a, hi_a) in {0.80: (0.10, 0.90), 0.95: (0.025, 0.975)}.items():
        mlo, mhi = q_model(lo_a), q_model(hi_a)
        lo_c, hi_c = mlo.predict(Xc), mhi.predict(Xc)
        scores = np.maximum(lo_c - yc, yc - hi_c)        # CQR conformity score
        n = len(scores)
        k = min(n - 1, int(np.ceil((n + 1) * lvl)) - 1)  # (1-alpha)=lvl quantile
        Q = np.sort(scores)[max(0, k)]
        lo_p = float(mlo.predict(Xp)[0]) - Q
        hi_p = float(mhi.predict(Xp)[0]) + Q
        if lvl == 0.80:
            preds[0.10], preds[0.90] = lo_p, hi_p
        else:
            preds[0.025], preds[0.975] = lo_p, hi_p
    vals = np.sort([preds[q] for q in QUANTILES])
    preds = {q: vals[i] for i, q in enumerate(QUANTILES)}
    return preds[0.50], preds


MODELS = {
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_growth": seasonal_naive_growth,
    "ar1_calendar": ar1_calendar,
    "ens_growth_ar1": ens_growth_ar1,
    "sarimax": sarimax,
    "ets": ets,
    "lgbm_quantile": lgbm_quantile,
    "lgbm_cqr": lgbm_cqr,
}
