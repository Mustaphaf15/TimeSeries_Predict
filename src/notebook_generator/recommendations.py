from typing import Dict, List
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def _has_missing_values(series: pd.Series) -> bool:
    """Vérifie la présence de valeurs manquantes."""
    return series.isnull().sum() > 0

def _has_outliers(series: pd.Series, threshold: float = 1.5) -> bool:
    """Détecte les outliers en utilisant l'écart interquartile (IQR)."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return ((series < lower_bound) | (series > upper_bound)).any()

def _detect_seasonality(series: pd.Series, period: int = 7) -> bool:
    """Détecte une saisonnalité significative."""
    if len(series) < 2 * period:
        return False # Pas assez de données pour la décomposition

    decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
    seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())

    # Un seuil simple pour déterminer si la saisonnalité est "forte"
    return seasonal_strength > 0.5


def analyze_series_and_recommend_transformers(df: pd.DataFrame, target_column: str) -> Dict[str, List[Dict]]:
    """
    Analyse la série temporelle et recommande des transformateurs sktime.
    """
    series = df[target_column]
    recommendations = {
        "target_transform": [],
        "exog_transform": []
    }

    # Règle 1: Imputation si des valeurs manquantes sont présentes
    if _has_missing_values(series):
        recommendations["target_transform"].append({
            'type': 'imputer',
            'transform': 'Imputer',
            'method': 'linear',
            'reason': "Des valeurs manquantes ont été détectées."
        })

    # Règle 2: Détection d'outliers
    if _has_outliers(series):
        recommendations["target_transform"].append({
            'type': 'outlier_remover',
            'transform': 'HampelFilter',
            'window_length': 7,
            'reason': "Des outliers potentiels ont été détectés."
        })

    # Règle 3: Gestion de la saisonnalité
    if _detect_seasonality(series):
        recommendations["exog_transform"].append({
            'type': 'fourier',
            'transform': 'FourierFeatures',
            'sp_list': [7, 365.25],
            'fourier_terms_list': [2, 3],
            'reason': "Une forte saisonnalité a été détectée."
        })

    return recommendations
