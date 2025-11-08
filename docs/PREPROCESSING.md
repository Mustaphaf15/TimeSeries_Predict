# Documentation du Module : `preprocessing`

## Fichier : `transformers.py`

### ` def _create_single_pipeline(transformer_configs: List[Dict[str, Any]]) -> TransformerPipeline `

```
Helper to create a single sktime pipeline from a list of configs.
```

### ` def create_transformer_pipelines(
    config: Dict[str, Any],
) -> Dict[str, Any] `

```
Crée des pipelines séparés pour target et exogènes.
```

### ` def encode_categorical_features(
    data: pd.DataFrame,
    cat_columns: List[str]
) -> Tuple[pd.DataFrame, List[str]] `

```
Encode les variables catégorielles et retourne les nouvelles colonnes.
```

### ` def fit_transform_backtest(
    data: pd.DataFrame,
    pipelines: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]] `

```
Fit et transforme les données backtest.
```

### ` def transform_production(
    data: pd.DataFrame,
    fitted_pipelines: Dict[str, Any]
) -> pd.DataFrame `

```
Transforme les données production avec pipelines déjà fittés.
```
