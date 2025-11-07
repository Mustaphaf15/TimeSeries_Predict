from typing import Any, Dict, Tuple
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    ForecastingRandomizedSearchCV,
)
from sktime.forecasting.base import BaseForecaster
import pandas as pd

def create_tuner(
    search_method: str,
    forecaster: BaseForecaster,
    param_grid: Dict[str, Any],
    splitter,
    scoring: str,
    n_iter: int = 10,
) -> Any:
    """
    Crée un tuner d'hyperparamètres sktime.
    """
    if search_method == "grid":
        return ForecastingGridSearchCV(
            forecaster=forecaster,
            cv=splitter,
            param_grid=param_grid,
            scoring=scoring,
        )
    elif search_method == "random":
        return ForecastingRandomizedSearchCV(
            forecaster=forecaster,
            cv=splitter,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
        )
    # Optuna would require an additional dependency and a custom wrapper.
    # For now, we focus on the built-in sktime tuners.
    elif search_method == "optuna":
        raise NotImplementedError("Optuna tuning is not yet implemented.")
    else:
        raise ValueError(f"Méthode de recherche non supportée: {search_method}")

def tune_hyperparameters(
    tuner: Any,
    data: pd.DataFrame,
    target_column: str,
    exog_columns: list = None
) -> Tuple[BaseForecaster, Dict[str, Any]]:
    """
    Exécute le tuning et retourne le meilleur forecaster et ses paramètres.
    """
    y = data[[target_column]]
    X = data[exog_columns] if exog_columns else None

    tuner.fit(y, X=X)

    return tuner.best_forecaster_, tuner.best_params_
