"""
recommendations.py

Rules-based engine to analyze a time series and recommend sktime transform pipelines.
The implementation favors interpretability and robustness over heavy ML.
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

from .utils import detect_frequency, series_summary


def _has_missing_values(series: pd.Series) -> bool:
    return int(series.isnull().sum()) > 0


def _iqr_outliers(series: pd.Series, threshold: float = 1.5) -> bool:
    if series.dropna().empty:
        return False
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - threshold * iqr
    ub = q3 + threshold * iqr
    return ((series < lb) | (series > ub)).any()


def _rolling_zscore_outliers(series: pd.Series, window: int = 7, z_thresh: float = 3.0) -> bool:
    s = series.dropna()
    if s.empty or len(s) < window + 1:
        return False
    roll_mean = s.rolling(window=window, min_periods=1, center=False).mean()
    roll_std = s.rolling(window=window, min_periods=1, center=False).std().replace(0, np.nan)
    z = (s - roll_mean) / roll_std
    return (z.abs() > z_thresh).any()


def _detect_trend(series: pd.Series) -> bool:
    """
    Quick heuristic for trend: check if rolling mean changes systematically,
    and run KPSS to detect non-stationarity (trend).
    """
    s = series.dropna()
    if s.empty or len(s) < 10:
        return False
    try:
        stat, p_value, _, _ = kpss(s, nlags="auto")
        # KPSS null: series is stationary; if p < 0.05 then not stationary (trend)
        return p_value < 0.05
    except Exception:
        # fallback heuristic: check slope of linear fit
        idx = np.arange(len(s))
        coef = np.polyfit(idx, s.values, 1)[0]
        return abs(coef) > 1e-8


def _detect_seasonality_multi(series: pd.Series, candidate_periods: List[int]) -> List[int]:
    """
    Attempt seasonal_decompose for several candidate periods and return those with strong seasonality.
    Use a simple seasonal_strength metric:
        seasonal_strength = var(seasonal) / var(seasonal + resid)
    Returns periods that pass threshold 0.35 (tunable).
    """
    strong = []
    s = series.dropna()
    if s.empty:
        return strong

    for p in candidate_periods:
        if len(s) < max(2 * p, 20):  # not enough data
            continue
        try:
            dec = seasonal_decompose(s, model='additive', period=p, two_sided=False, extrapolate_trend='freq')
            denom = (dec.seasonal + dec.resid).var() if (dec.seasonal is not None and dec.resid is not None) else np.nan
            if np.isnan(denom) or denom <= 0:
                continue
            seasonal_strength = float(dec.seasonal.var() / denom)
            if seasonal_strength >= 0.35:
                strong.append(p)
        except Exception:
            # skip problematic p
            continue
    return strong


def _check_variance_instability(series: pd.Series, window: int = 30, pct_threshold: float = 0.5) -> bool:
    """
    Check whether rolling std changes a lot (coefficient of variation of rolling std exceeds threshold)
    """
    s = series.dropna()
    if s.empty or len(s) < window + 1:
        return False
    roll_std = s.rolling(window=window, min_periods=1).std()
    cv = roll_std.std() / (roll_std.mean() + 1e-9)
    return cv > pct_threshold


def _stationarity_tests(series: pd.Series) -> Dict[str, Any]:
    """Return ADF + KPSS results (p-values) if available."""
    res = {}
    s = series.dropna()
    if s.empty:
        return {"adf_pvalue": None, "kpss_pvalue": None}
    try:
        adf_res = adfuller(s, autolag='AIC')
        res["adf_pvalue"] = float(adf_res[1])
    except Exception:
        res["adf_pvalue"] = None
    try:
        kpss_res = kpss(s, nlags='auto')
        res["kpss_pvalue"] = float(kpss_res[1])
    except Exception:
        res["kpss_pvalue"] = None
    return res


def analyze_series_and_recommend_transformers(
    df: pd.DataFrame,
    target_column: str,
    freq: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze the provided dataframe and produce recommendations for target_transform and exog_transform.

    Returns a dict:
        {
          "target_transform": [ {type, transform, params..., reason}, ... ],
          "exog_transform": [ {...}, ... ],
          "meta": { ... profiling info ... }
        }
    """
    recs = {"target_transform": [], "exog_transform": [], "meta": []}
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not present in dataframe")

    series = df[target_column]

    # Profile
    profile = series_summary(series)
    recs["meta"].append({"profile": profile})

    # Frequency detection (if not passed)
    inferred_freq = freq or detect_frequency(df)
    recs["meta"].append({"detected_freq": inferred_freq})

    # Missing values -> Imputer
    if _has_missing_values(series):
        recs["target_transform"].append({
            "type": "imputer",
            "transform": "Imputer",
            "method": "linear",
            "reason": "Présence de valeurs manquantes (interpolation linéaire recommandée pour trous courts)."
        })

    # Outliers -> HampelFilter or rolling z-score
    if _rolling_zscore_outliers(series, window=7, z_thresh=3.0) or _iqr_outliers(series):
        recs["target_transform"].append({
            "type": "outlier_remover",
            "transform": "HampelFilter",
            "window_length": 7,
            "n_sigma": 3,
            "reason": "Anomalies temporelles détectées (HampelFilter recommandé)."
        })

    # Variance instability -> Log or BoxCox
    if _check_variance_instability(series, window=30, pct_threshold=0.5):
        recs["target_transform"].append({
            "type": "variance_stabilizer",
            "transform": "LogTransformer",
            "reason": "Variance non-stationnaire détectée (LogTransformer recommandé)."
        })

    # Trend -> Detrender / HPFilter
    if _detect_trend(series):
        recs["target_transform"].append({
            "type": "detrender",
            "transform": "Detrender",
            "reason": "Composante de tendance détectée (Detrender recommandé)."
        })

    # Stationarity tests
    st_res = _stationarity_tests(series)
    recs["meta"].append({"stationarity": st_res})
    if st_res.get("adf_pvalue") is not None and st_res.get("adf_pvalue") > 0.05:
        # adf fails to reject null -> non stationary
        recs["target_transform"].append({
            "type": "differencing",
            "transform": "Difference",
            "order": 1,
            "reason": "ADF indique non-stationnarité : différenciation à considérer."
        })

    # Seasonality detection (try several candidate periods based on freq)
    candidate_periods = []
    if inferred_freq:
        # some heuristics mapping pandas freq to integer period candidates (daily series -> weekly=7)
        if inferred_freq.startswith("D"):
            candidate_periods = [7, 365]
        elif inferred_freq.startswith("H"):
            candidate_periods = [24, 24*7]
        elif inferred_freq.startswith("T") or ":" in str(inferred_freq) or "min" in str(inferred_freq):
            candidate_periods = [24*60 // int(inferred_freq.replace("T", ""))] if "T" in str(inferred_freq) else [24, 7*24]
            # fallback
            candidate_periods = [24, 168]
        else:
            candidate_periods = [7, 365]
    else:
        # sensible defaults
        candidate_periods = [7, 24, 365]

    strong_seasons = _detect_seasonality_multi(series, candidate_periods)
    for p in strong_seasons:
        recs["exog_transform"].append({
            "type": "fourier",
            "transform": "FourierFeatures",
            "sp": p,
            "k": 3,
            "reason": f"Saisonnalité détectée pour période {p} (FourierFeatures recommandé)."
        })
        # also recommend Deseasonalizer if strong seasonal component
        recs["target_transform"].append({
            "type": "deseasonalize",
            "transform": "Deseasonalizer",
            "sp": p,
            "reason": f"Désaisonnalisation pour la période {p} recommandée."
        })

    # Autocorrelation heuristic: recommend lags if series long enough
    if profile["n"] >= 30:
        recs["exog_transform"].append({
            "type": "lag_features",
            "transform": "Lag",
            "lags": [1, 7] if 7 in candidate_periods else [1, 2, 3],
            "reason": "Autocorrélation potentielle détectée ou données suffisamment longues -> lag features utiles."
        })

    return recs
