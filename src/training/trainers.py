from typing import Any, Dict, List, Tuple
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.fbprophet import Prophet
from src.splitting.splitters import split_data_by_fold

# Note: LSTM is a more complex model and would require a separate module for deep learning.
# For now, we will focus on the sktime-native models.

FORECASTER_MAP = {
    "sarimax": SARIMAX,
    "fbprophet": Prophet,
}

def create_forecaster(model_config: Dict[str, Any], model_name: str) -> BaseForecaster:
    """
    Crée un forecaster sktime à partir de la configuration.
    Gère à la fois les paramètres uniques et les listes de paramètres pour le tuning.
    """
    forecaster_class = FORECASTER_MAP.get(model_name)
    if not forecaster_class:
        raise ValueError(f"Modèle non supporté: {model_name}")

    params = model_config.get('predictors', {}).get(model_name, {})

    final_params = {}
    for key, value in params.items():
        # Si la valeur est une liste de listes (ex: pour le tuning de 'order'), on prend la première sous-liste.
        if isinstance(value, list) and value and isinstance(value[0], list):
            final_params[key] = value[0]
        # Sinon, on utilise la valeur telle quelle (ex: un paramètre 'order' unique qui est une liste).
        else:
            final_params[key] = value

    return forecaster_class(**final_params)

def train_on_fold(
    forecaster: BaseForecaster,
    train_data: pd.DataFrame,
    target_column: str,
    exog_columns: List[str] = None
) -> BaseForecaster:
    """Entraîne un forecaster sur un pli de données."""
    y_train = train_data[[target_column]]
    X_train = train_data[exog_columns] if exog_columns else None

    forecaster.fit(y=y_train, X=X_train)
    return forecaster

def train_and_predict_on_folds(
    forecaster: BaseForecaster,
    data: pd.DataFrame,
    folds: List[Dict[str, Any]],
    target_column: str,
    exog_columns: List[str] = None
) -> Tuple[BaseForecaster, pd.DataFrame]:
    """
    Entraîne et prédit sur plusieurs plis, en retournant les prédictions agrégées.
    """
    all_predictions = []

    for fold in folds:
        train_df, test_df = split_data_by_fold(data, fold)

        y_train = train_df[[target_column]]
        X_train = train_df[exog_columns] if exog_columns else None
        forecaster.fit(y=y_train, X=X_train)

        fh = list(range(1, len(test_df) + 1))
        X_test = test_df[exog_columns] if exog_columns else None

        predictions = forecaster.predict(fh=fh, X=X_test)

        predictions.index = test_df.index
        all_predictions.append(predictions)

    final_forecaster = forecaster
    all_preds_df = pd.concat(all_predictions)

    return final_forecaster, all_preds_df
