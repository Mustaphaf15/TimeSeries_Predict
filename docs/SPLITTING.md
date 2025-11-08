# Documentation du Module : `splitting`

## Fichier : `splitters.py`

### ` def create_splitter(
    splitter_config: Dict[str, Any],
    forecast_horizon: int,
    data_length: int
) `

```
Crée un splitter sktime à partir de la configuration.
```

### ` def generate_folds(data: pd.DataFrame, splitter) -> List[Dict[str, Any]] `

```
Génère les plis de validation croisée.
```

### ` def get_train_test_indices(fold: Dict[str, Any]) -> Tuple[pd.Index, pd.Index] `

```
Récupère les index d'entraînement et de test pour un pli.
```

### ` def split_data_by_fold(data: pd.DataFrame, fold: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame] `

```
Sépare les données en ensembles d'entraînement et de test pour un pli.
```

### ` def get_folds_summary(folds: List[Dict[str, Any]]) -> pd.DataFrame `

```
Crée un résumé des plis générés.
```

### ` def validate_splitter_config(config: Dict[str, Any], data_length: int, forecast_horizon: int) -> Dict[str, Any] `

```
Valide la configuration du splitter.
```
