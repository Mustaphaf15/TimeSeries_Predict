# Documentation du Module : `tuning`

## Fichier : `tuners.py`

### ` def create_tuner(
    search_method: str,
    forecaster: BaseForecaster,
    param_grid: Dict[str, Any],
    splitter,
    scoring: str,
    n_iter: int = 10,
) -> Any `

```
Crée un tuner d'hyperparamètres sktime.
```

### ` def tune_hyperparameters(
    tuner: Any,
    data: pd.DataFrame,
    target_column: str,
    exog_columns: list = None
) -> Tuple[BaseForecaster, Dict[str, Any]] `

```
Exécute le tuning et retourne le meilleur forecaster et ses paramètres.
```
