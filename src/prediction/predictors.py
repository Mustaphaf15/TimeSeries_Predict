from typing import Any, Dict, List
import pandas as pd
from sktime.forecasting.base import BaseForecaster

def load_trained_model(model_path: str) -> BaseForecaster:
    """
    Charge un modèle entraîné depuis un fichier.
    (Pour l'instant, c'est un placeholder. L'intégration MLflow gérera cela.)
    """
    # Dans une implémentation réelle, on utiliserait joblib ou pickle.
    # import joblib
    # return joblib.load(model_path)
    raise NotImplementedError("Le chargement de modèle n'est pas encore implémenté.")

def prepare_production_data(
    data: pd.DataFrame,
    fitted_pipelines: Dict[str, Any]
) -> pd.DataFrame:
    """
    Prépare les données de production en appliquant les transformations fittées.
    """
    # Cette fonction est un alias pour transform_production pour plus de clarté sémantique.
    from src.preprocessing.transformers import transform_production
    return transform_production(data, fitted_pipelines)

def generate_predictions(
    forecaster: BaseForecaster,
    prepared_data: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """
    Génère les prédictions pour l'horizon futur.
    """
    fh = list(range(1, horizon + 1))

    # prepared_data contient les variables exogènes pour le futur
    X = prepared_data if not prepared_data.empty else None

    predictions = forecaster.predict(fh=fh, X=X)

    # Formatter la sortie
    if isinstance(predictions.index, pd.RangeIndex):
       # Si l'index n'est pas temporel, essayons de le construire
       last_known_date = forecaster.cutoff
       future_dates = pd.date_range(start=last_known_date, periods=horizon + 1, freq=forecaster.freq)[1:]
       predictions.index = future_dates

    return predictions
