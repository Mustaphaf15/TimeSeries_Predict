from typing import Any, Dict, List
import pandas as pd

# Ce module est un placeholder pour l'intégration MLflow.
# Dans une implémentation complète, il utiliserait sktime.utils.mlflow_sktime.

def init_mlflow_experiment(experiment_name: str):
    """Initialise une expérience MLflow."""
    print(f"INFO: Initialisation de l'expérience MLflow: {experiment_name}")

def log_params(params_dict: Dict[str, Any]):
    """Log les paramètres d'un run."""
    print("INFO: Logging des paramètres:", params_dict)

def log_metrics(metrics_dict: Dict[str, float], step: int = None):
    """Log les métriques d'un run."""
    print(f"INFO: Logging des métriques (étape {step if step else 'globale'}):", metrics_dict)

def log_forecaster(forecaster: Any, artifact_path: str):
    """Log un forecaster comme un artifact."""
    print(f"INFO: Logging du forecaster vers: {artifact_path}")

def log_fold_results(fold_index: int, predictions: pd.DataFrame, metrics: Dict[str, float]):
    """Log les résultats d'un pli spécifique."""
    print(f"--- Pli {fold_index} ---")
    print("  Métriques:", metrics)
    # On ne log pas les prédictions ici pour éviter de surcharger la console.

def log_dataset(data: pd.DataFrame, name: str):
    """Log un dataset comme un artifact."""
    print(f"INFO: Logging du dataset '{name}' ({data.shape})")
