# Documentation du Module : `data_loading`

## Fichier : `loaders.py`

### ` def _build_entity_id(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame `

```
Construit la colonne 'entity_id' en concaténant les colonnes d'entité.
```

### ` def load_data_from_csv(config: Dict[str, Any], data_type: str = 'backtest') -> pd.DataFrame `

```
Charge les données depuis un fichier CSV (backtest ou production).
```

### ` def load_data_from_clickhouse(config: Dict[str, Any]) -> pd.DataFrame `

```
Charge les données depuis ClickHouse.
```

### ` def split_backtest_production(
    data: pd.DataFrame,
    production_start_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame] `

```
Sépare les données en backtest et production selon une date de coupure.
```

### ` def load_entity_data(entity_name: str, config_dir: str = "config", entity_config_name: str = None) -> Tuple[pd.DataFrame, pd.DataFrame] `

```
Charge les données backtest et production pour une entité.
```

### ` def join_non_empty(row, sep) `

```
Joins only the non-empty strings in a row.
```
