from datetime import datetime
import os
import pandas as pd
import joblib
from typing import Any, Dict

def build_output_filename(
    entity: str, model: str, category: str, config: Dict[str, Any]
) -> str:
    """Construit un nom de fichier standardisé pour un artefact."""
    naming_config = config.get('output', {}).get('naming', {})
    timestamp = datetime.now().strftime(naming_config.get('timestamp_format', '%Y%m%d_%H%M%S'))

    parts = []
    if naming_config.get('include_entity_name', True):
        parts.append(entity)
    if naming_config.get('include_model_name', True):
        parts.append(model)
    if naming_config.get('include_timestamp', True):
        parts.append(timestamp)

    base_name = "_".join(parts)

    formats = config.get('output', {}).get('formats', {})
    extension = formats.get(category, 'txt')

    return f"{base_name}.{extension}"

def save_local(
    obj: Any, category: str, entity: str, model: str, config: Dict[str, Any]
) -> str:
    """Sauvegarde un artefact en local et retourne son chemin."""
    directories = config.get('output', {}).get('directories', {})

    # Assurer que le dossier de base existe
    base_dir = directories.get('base', 'outputs/')
    os.makedirs(base_dir, exist_ok=True)

    # Déterminer le sous-dossier et s'assurer qu'il existe
    category_dir = directories.get(category)
    if not category_dir:
        raise ValueError(f"La catégorie de sortie '{category}' n'est pas définie dans la configuration.")
    os.makedirs(category_dir, exist_ok=True)

    # Construire le nom de fichier et le chemin complet
    filename = build_output_filename(entity, model, category, config)
    filepath = os.path.join(category_dir, filename)

    # Sauvegarder l'objet en fonction de son type
    if isinstance(obj, pd.DataFrame):
        file_format = config.get('output', {}).get('formats', {}).get(category, 'csv')
        if file_format == 'csv':
            obj.to_csv(filepath, index=False)
        elif file_format == 'parquet':
            obj.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Format de fichier non supporté pour les DataFrames: {file_format}")
    elif hasattr(obj, 'save'): # Pour les modèles qui ont une méthode save
        obj.save(filepath)
    else: # Sauvegarde générique avec joblib
        joblib.dump(obj, filepath)

    print(f"INFO: Artefact sauvegardé en local: {filepath}")
    return filepath
