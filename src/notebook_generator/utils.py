"""
Utilities used by notebook_generator.
"""
from typing import Optional
import pandas as pd
import numpy as np


def detect_frequency(df: pd.DataFrame, datetime_index_col: Optional[str] = None) -> Optional[str]:
    """
    Try to detect frequency of a time series dataframe.
    Returns a pandas offset alias string (e.g. 'D', 'H', '30T', ...) or None.
    """
    if datetime_index_col:
        s = pd.to_datetime(df[datetime_index_col])
    else:
        # assume df.index is datetime-like
        s = pd.to_datetime(df.index)

    # if explicit freq is set on index, return it
    if hasattr(s, "freq") and s.freq is not None:
        return s.freq.freqstr

    # try to infer
    try:
        inferred = pd.infer_freq(s)
        return inferred
    except Exception:
        return None


def series_summary(series: pd.Series) -> dict:
    """
    Returns a small profile summary of a series.
    """
    res = {}
    res["n"] = int(series.shape[0])
    res["n_missing"] = int(series.isnull().sum())
    res["pct_missing"] = float(series.isnull().mean())
    res["min"] = float(series.min()) if not series.dropna().empty else None
    res["max"] = float(series.max()) if not series.dropna().empty else None
    res["mean"] = float(series.mean()) if not series.dropna().empty else None
    res["std"] = float(series.std()) if not series.dropna().empty else None
    res["iqr"] = None
    if not series.dropna().empty:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        res["iqr"] = float(q3 - q1)
    return res
