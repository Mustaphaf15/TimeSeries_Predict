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
    Si la configuration contient une liste de paramètres pour le tuning,
    seul le premier est utilisé pour l'instanciation.
    """
    forecaster_class = FORECASTER_MAP.get(model_name)
    if not forecaster_class:
        raise ValueError(f"Modèle non supporté: {model_name}")

    params = model_config.get('predictors', {}).get(model_name, {})

    # Handle parameter grids meant for tuning by picking the first option
    final_params = {}
    for key, value in params.items():
        if isinstance(value, list) and value:
            final_params[key] = value[0]
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

        # Entraînement
        y_train = train_df[[target_column]]
        X_train = train_df[exog_columns] if exog_columns else None
        forecaster.fit(y=y_train, X=X_train)

        # Prédiction
        fh = list(range(1, len(test_df) + 1))
        X_test = test_df[exog_columns] if exog_columns else None

        predictions = forecaster.predict(fh=fh, X=X_test)

        # S'assurer que les prédictions ont le bon index
        predictions.index = test_df.index
        all_predictions.append(predictions)

    # Le forecaster final est celui entraîné sur le dernier pli
    final_forecaster = forecaster

    # Concaténer toutes les prédictions
    all_preds_df = pd.concat(all_predictions)

    return final_forecaster, all_preds_df
