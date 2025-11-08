from typing import Any, Dict
import pandas as pd
from . import output_manager
from . import mlflow_manager

def log_predictions(df: pd.DataFrame, entity: str, model: str, config: Dict[str, Any]):
    """Sauvegarde et log les prédictions."""
    local_path = output_manager.save_local(df, 'predictions', entity, model, config)
    mlflow_manager.log_artifact(local_path, 'predictions')

def log_model(model_obj: Any, entity: str, model: str, config: Dict[str, Any]):
    """Sauvegarde et log un modèle."""
    local_path = output_manager.save_local(model_obj, 'models', entity, model, config)
    mlflow_manager.log_artifact(local_path, 'models')

def log_plot(local_path: str, entity: str, model: str, config: Dict[str, Any]):
    """Log un graphique."""
    # Le graphique est déjà sauvegardé localement, on le log directement.
    mlflow_manager.log_artifact(local_path, 'plots')

def log_config(config_dict: Dict[str, Any], entity: str, model: str, config: Dict[str, Any]):
    """Sauvegarde et log la configuration."""
    import yaml

    # Créer un nom de fichier temporaire pour le dictionnaire de configuration
    temp_dir = config.get('output', {}).get('directories', {}).get('base', 'outputs/')
    filepath = f"{temp_dir}/{entity}_{model}_config.yaml"

    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f)

    mlflow_manager.log_artifact(filepath, 'configs')

def log_report(local_path: str, entity: str, model: str, config: Dict[str, Any]):
    """Log un rapport."""
    mlflow_manager.log_artifact(local_path, 'reports')
