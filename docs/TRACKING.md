# Documentation du Module : `tracking`

## Fichier : `artifacts_logger.py`

### ` def log_predictions(df: pd.DataFrame, entity: str, model: str, config: Dict[str, Any]) `

```
Sauvegarde et log les prédictions.
```

### ` def log_model(model_obj: Any, entity: str, model: str, config: Dict[str, Any]) `

```
Sauvegarde et log un modèle.
```

### ` def log_plot(local_path: str, entity: str, model: str, config: Dict[str, Any]) `

```
Log un graphique.
```

### ` def log_config(config_dict: Dict[str, Any], entity: str, model: str, config: Dict[str, Any]) `

```
Sauvegarde et log la configuration.
```

### ` def log_report(local_path: str, entity: str, model: str, config: Dict[str, Any]) `

```
Log un rapport.
```

## Fichier : `mlflow_manager.py`

### ` def init_mlflow(tracking_uri: str) `

```
Configure MLflow et définit le tracking URI.
```

### ` def init_mlflow_experiment(experiment_name: str) `

```
Crée ou sélectionne un experiment MLflow.
```

### ` def start_run(entity_name: str, model_name: str, tags: Dict[str, Any] = {}) `

```
Démarre un run MLflow avec des tags et retourne le context manager.
```

### ` def end_run() `

```
Termine un run MLflow proprement.
```

### ` def configure_autologging(settings_dict: Dict[str, Any]) `

```
Active ou désactive l'autologging de MLflow.
```

### ` def log_params(params_dict: Dict[str, Any]) `

```
Log les paramètres d'un run.
```

### ` def log_metrics(metrics_dict: Dict[str, float], step: int = None) `

```
Log les métriques d'un run.
```

### ` def log_artifact(local_path: str, artifact_path: str = None) `

```
Log un fichier comme un artefact.
```

## Fichier : `output_manager.py`

### ` def build_output_filename(
    entity: str, model: str, category: str, config: Dict[str, Any]
) -> str `

```
Construit un nom de fichier standardisé pour un artefact.
```

### ` def save_local(
    obj: Any, category: str, entity: str, model: str, config: Dict[str, Any]
) -> str `

```
Sauvegarde un artefact en local et retourne son chemin.
```

## Fichier : `system_logger.py`

### ` def log_system_metrics() `

```
Placeholder pour le logging des métriques système.

L'approche recommandée est d'utiliser mlflow.autolog(), qui gère cela
automatiquement pour de nombreux frameworks.

Si un suivi manuel est nécessaire, on pourrait utiliser des bibliothèques
comme 'psutil' pour collecter des informations sur le CPU, la RAM, etc.
et les logger avec mlflow.log_metric().
```
