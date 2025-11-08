import mlflow
from typing import Any, Dict
from datetime import datetime

def init_mlflow(tracking_uri: str):
    """Configure MLflow et définit le tracking URI."""
    mlflow.set_tracking_uri(tracking_uri)

def init_mlflow_experiment(experiment_name: str):
    """Crée ou sélectionne un experiment MLflow."""
    mlflow.set_experiment(experiment_name)

def start_run(entity_name: str, model_name: str, tags: Dict[str, Any] = {}):
    """Démarre un run MLflow avec des tags et retourne le context manager."""
    run_name = f"{entity_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"INFO: Démarrage du run MLflow: {run_name}")
    return mlflow.start_run(run_name=run_name, tags=tags)

def end_run():
    """Termine un run MLflow proprement."""
    mlflow.end_run()
    print("INFO: Run MLflow terminé.")

def configure_autologging(settings_dict: Dict[str, Any]):
    """Active ou désactive l'autologging de MLflow."""
    if settings_dict.get('enabled', False):
        mlflow.autolog(**settings_dict)
        print("INFO: Autologging MLflow activé.")
    else:
        print("INFO: Autologging MLflow désactivé.")

def log_params(params_dict: Dict[str, Any]):
    """Log les paramètres d'un run."""
    mlflow.log_params(params_dict)

def log_metrics(metrics_dict: Dict[str, float], step: int = None):
    """Log les métriques d'un run."""
    mlflow.log_metrics(metrics_dict, step=step)

def log_artifact(local_path: str, artifact_path: str = None):
    """Log un fichier comme un artefact."""
    mlflow.log_artifact(local_path, artifact_path)

# La fonction `log_forecaster` de sktime est plus spécifique, on la garde dans le module de haut niveau.
from datetime import datetime
