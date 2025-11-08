# Documentation du Module : `notebook_generator`

## Fichier : `generator.py`

### ` def _safe_extend(cells: List, section_fn, *args, **kwargs) `

```
Call section_fn and extend cells with returned cells.
If section_fn raises, append a markdown cell with the error and continue.
```

### ` def generate_notebook(entity_name: str, output_dir: str = "notebooks", config_dir: str = "config", sections: Optional[List[str]] = None) -> Path `

```
Generate an EDA notebook for a given entity.

Args:
    entity_name: name of the entity in config/entities
    output_dir: where to write the .ipynb
    config_dir: path to the configuration directory
    sections: optional list of sections to include (subset of DEFAULT_SECTION_ORDER)

Returns:
    Path to the generated notebook file.
```

### ` def build_notebooks_for_all_entities(output_dir: str = "notebooks", config_dir: str = "config", active_only: bool = True) -> List[Path] `

```
Build notebooks for all entities found in configuration.

Returns list of generated notebook paths.
```

### ` def main() `

*Aucune docstring fournie.*

## Fichier : `recommendations.py`

### ` def _has_missing_values(series: pd.Series) -> bool `

*Aucune docstring fournie.*

### ` def _iqr_outliers(series: pd.Series, threshold: float = 1.5) -> bool `

*Aucune docstring fournie.*

### ` def _rolling_zscore_outliers(series: pd.Series, window: int = 7, z_thresh: float = 3.0) -> bool `

*Aucune docstring fournie.*

### ` def _detect_trend(series: pd.Series) -> bool `

```
Quick heuristic for trend: check if rolling mean changes systematically,
and run KPSS to detect non-stationarity (trend).
```

### ` def _detect_seasonality_multi(series: pd.Series, candidate_periods: List[int]) -> List[int] `

```
Attempt seasonal_decompose for several candidate periods and return those with strong seasonality.
Use a simple seasonal_strength metric:
    seasonal_strength = var(seasonal) / var(seasonal + resid)
Returns periods that pass threshold 0.35 (tunable).
```

### ` def _check_variance_instability(series: pd.Series, window: int = 30, pct_threshold: float = 0.5) -> bool `

```
Check whether rolling std changes a lot (coefficient of variation of rolling std exceeds threshold)
```

### ` def _stationarity_tests(series: pd.Series) -> Dict[str, Any] `

```
Return ADF + KPSS results (p-values) if available.
```

### ` def analyze_series_and_recommend_transformers(
    df: pd.DataFrame,
    target_column: str,
    freq: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]] `

```
Analyze the provided dataframe and produce recommendations for target_transform and exog_transform.

Returns a dict:
    {
      "target_transform": [ {type, transform, params..., reason}, ... ],
      "exog_transform": [ {...}, ... ],
      "meta": { ... profiling info ... }
    }
```

## Fichier : `sections.py`

### ` def _imports_cell() -> Any `

```
Centralized imports cell for the generated notebook.
```

### ` def create_meta_section(entity_name: str, config: Dict) -> List `

```
Create a technical header with config, generation date, and version.
```

### ` def create_data_loading_section(entity_name: str, target_column: str) -> List `

```
Creates a data loading cell that uses the project's loader.
```

### ` def create_profile_section() -> List `

```
Profile summary: freq, length, missing ratio, simple stats.
```

### ` def create_data_quality_section() -> List `

*Aucune docstring fournie.*

### ` def create_descriptive_eda_section() -> List `

*Aucune docstring fournie.*

### ` def create_seasonality_section() -> List `

*Aucune docstring fournie.*

### ` def create_autocorrelation_section() -> List `

*Aucune docstring fournie.*

### ` def create_anomaly_section() -> List `

*Aucune docstring fournie.*

### ` def create_stationarity_section() -> List `

*Aucune docstring fournie.*

### ` def create_exogenous_analysis_section() -> List `

*Aucune docstring fournie.*

### ` def create_recommendations_section() -> List `

*Aucune docstring fournie.*

### ` def create_summary_section() -> List `

*Aucune docstring fournie.*

## Fichier : `utils.py`

### ` def detect_frequency(df: pd.DataFrame, datetime_index_col: Optional[str] = None) -> Optional[str] `

```
Try to detect frequency of a time series dataframe.
Returns a pandas offset alias string (e.g. 'D', 'H', '30T', ...) or None.
```

### ` def series_summary(series: pd.Series) -> dict `

```
Returns a small profile summary of a series.
```
