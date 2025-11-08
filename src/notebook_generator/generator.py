# src/notebook_generator/generator.py

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook
from typing import List, Dict, Any
import argparse

from src.configuration.config import get_entity_config
from src.notebook_generator.sections import (
    create_intro_section,
    create_data_loading_section,
    create_data_quality_section,
    create_descriptive_eda_section,
    create_seasonality_section,
    create_autocorrelation_section,
    create_anomaly_section,
    create_recommendations_section
)

def generate_notebook(entity_name: str, output_dir: str = "notebooks"):
    """
    Génère un notebook d'analyse exploratoire complet pour une entité donnée.

    Args:
        entity_name (str): Le nom de l'entité à analyser.
        output_dir (str): Le dossier où sauvegarder le notebook généré.
    """
    print(f"Génération du notebook pour l'entité : {entity_name}...")

    # Charger la configuration pour l'entité
    try:
        config = get_entity_config(entity_name)
    except FileNotFoundError:
        print(f"Erreur : Fichier de configuration pour l'entité '{entity_name}' non trouvé.")
        return

    target_column = config.get("data", {}).get("target_column", "nombre")

    # Créer le dossier de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialiser un nouveau notebook
    nb = new_notebook()

    # Construire le notebook section par section
    cells = []
    cells.extend(create_intro_section(entity_name, config))
    cells.extend(create_data_loading_section(entity_name, target_column))
    cells.extend(create_data_quality_section())
    cells.extend(create_descriptive_eda_section())
    cells.extend(create_seasonality_section())
    cells.extend(create_autocorrelation_section())
    cells.extend(create_anomaly_section())
    cells.extend(create_recommendations_section())

    nb['cells'] = cells

    # Sauvegarder le notebook
    output_path = Path(output_dir) / f"eda_report_{entity_name}.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook généré avec succès : {output_path}")

def main():
    """Point d'entrée principal pour la génération de notebooks via CLI."""
    parser = argparse.ArgumentParser(
        description="Générateur de Notebooks d'Analyse Exploratoire (EDA) pour TimeSeries Predict."
    )
    parser.add_argument(
        "entity_name",
        type=str,
        help="Le nom de l'entité pour laquelle générer le notebook (ex: 75_Paris_Sud-Ouest)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="notebooks",
        help="Le dossier où sauvegarder les notebooks générés."
    )
    args = parser.parse_args()

    generate_notebook(args.entity_name, args.output_dir)

if __name__ == "__main__":
    main()
