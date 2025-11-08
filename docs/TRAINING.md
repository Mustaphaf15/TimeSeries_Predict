# Documentation du Module : `training`

## Fichier : `trainers.py`

### ` def create_forecaster(model_config: Dict[str, Any], model_name: str) -> BaseForecaster `

```
Crée un forecaster sktime à partir de la configuration.
Gère à la fois les paramètres uniques et les listes de paramètres pour le tuning.
```

### ` def train_on_fold(
    forecaster: BaseForecaster,
    train_data: pd.DataFrame,
    target_column: str,
    exog_columns: List[str] = None
) -> BaseForecaster `

```
Entraîne un forecaster sur un pli de données.
```

### ` def train_and_predict_on_folds(
    forecaster: BaseForecaster,
    data: pd.DataFrame,
    folds: List[Dict[str, Any]],
    target_column: str,
    exog_columns: List[str] = None
) -> Tuple[BaseForecaster, pd.DataFrame] `

```
Entraîne et prédit sur plusieurs plis, en retournant les prédictions agrégées.
```
