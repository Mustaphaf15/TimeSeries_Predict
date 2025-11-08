# Documentation du Module : `prediction`

## Fichier : `predictors.py`

### ` def load_trained_model(model_path: str) -> BaseForecaster `

```
Charge un modèle entraîné depuis un fichier.
(Pour l'instant, c'est un placeholder. L'intégration MLflow gérera cela.)
```

### ` def prepare_production_data(
    data: pd.DataFrame,
    fitted_pipelines: Dict[str, Any]
) -> pd.DataFrame `

```
Prépare les données de production en appliquant les transformations fittées.
```

### ` def generate_predictions(
    forecaster: BaseForecaster,
    prepared_data: pd.DataFrame,
    horizon: int
) -> pd.DataFrame `

```
Génère les prédictions pour l'horizon futur.
```
