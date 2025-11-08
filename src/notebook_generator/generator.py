"""
generator.py

Orchestrator to build a complete EDA notebook per entity using sections.py and recommendations.py
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook
from typing import List, Dict, Any, Optional
import argparse
import datetime
import traceback

from src.configuration.config import get_entity_config, list_entities
from src.notebook_generator.sections import (
    create_meta_section,
    create_data_loading_section,
    create_profile_section,
    create_data_quality_section,
    create_descriptive_eda_section,
    create_seasonality_section,
    create_autocorrelation_section,
    create_anomaly_section,
    create_stationarity_section,
    create_exogenous_analysis_section,
    create_recommendations_section,
    create_summary_section,
    _imports_cell,
)
from src.notebook_generator.utils import detect_frequency

DEFAULT_SECTION_ORDER = [
    "meta",
    "data_loading",
    "profile",
    "data_quality",
    "descriptive_eda",
    "seasonality",
    "autocorrelation",
    "anomaly",
    "stationarity",
    "exogenous",
    "recommendations",
    "summary",
]


def _safe_extend(cells: List, section_fn, *args, **kwargs):
    """
    Call section_fn and extend cells with returned cells.
    If section_fn raises, append a markdown cell with the error and continue.
    """
    try:
        new_cells = section_fn(*args, **kwargs)
        cells.extend(new_cells)
    except Exception as e:
        import nbformat
        from nbformat.v4 import new_markdown_cell
        tb = traceback.format_exc()
        msg = f"**Erreur lors de la génération de la section {section_fn.__name__} :**\n\n```\n{str(e)}\n```\n\nTraceback:\n\n```\n{tb}\n```"
        cells.append(new_markdown_cell(msg))


def generate_notebook(entity_name: str, output_dir: str = "notebooks", config_dir: str = "config", sections: Optional[List[str]] = None) -> Path:
    """
    Generate an EDA notebook for a given entity.

    Args:
        entity_name: name of the entity in config/entities
        output_dir: where to write the .ipynb
        config_dir: path to the configuration directory
        sections: optional list of sections to include (subset of DEFAULT_SECTION_ORDER)

    Returns:
        Path to the generated notebook file.
    """
    now = datetime.datetime.utcnow().isoformat()
    print(f"[{now}] Génération du notebook pour l'entité : {entity_name}")

    try:
        config = get_entity_config(entity_name, config_dir=config_dir)
    except FileNotFoundError as e:
        raise

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nb = new_notebook()
    cells: List = []

    # choose order
    use_sections = sections or DEFAULT_SECTION_ORDER

    # Section: meta (includes imports)
    if "meta" in use_sections:
        _safe_extend(cells, create_meta_section, entity_name, config)

    # Data loading
    if "data_loading" in use_sections:
        _safe_extend(cells, create_data_loading_section, entity_name, config.get("data", {}).get("target_column", "target"))

    # profile
    if "profile" in use_sections:
        _safe_extend(cells, create_profile_section)

    # quality
    if "data_quality" in use_sections:
        _safe_extend(cells, create_data_quality_section)

    # descriptive
    if "descriptive_eda" in use_sections:
        _safe_extend(cells, create_descriptive_eda_section)

    # seasonality
    if "seasonality" in use_sections:
        _safe_extend(cells, create_seasonality_section)

    # autocorr
    if "autocorrelation" in use_sections:
        _safe_extend(cells, create_autocorrelation_section)

    # anomaly
    if "anomaly" in use_sections:
        _safe_extend(cells, create_anomaly_section)

    # stationarity
    if "stationarity" in use_sections:
        _safe_extend(cells, create_stationarity_section)

    # exogenous
    if "exogenous" in use_sections:
        _safe_extend(cells, create_exogenous_analysis_section)

    # recommendations
    if "recommendations" in use_sections:
        _safe_extend(cells, create_recommendations_section)

    # summary
    if "summary" in use_sections:
        _safe_extend(cells, create_summary_section)

    nb['cells'] = cells

    out_path = output_dir / f"eda_report_{entity_name}.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook généré : {out_path}")
    return out_path


def build_notebooks_for_all_entities(output_dir: str = "notebooks", config_dir: str = "config", active_only: bool = True) -> List[Path]:
    """
    Build notebooks for all entities found in configuration.

    Returns list of generated notebook paths.
    """
    entities = list_entities(active_only=active_only, config_dir=config_dir)
    generated = []
    for e in entities:
        try:
            p = generate_notebook(e, output_dir=output_dir, config_dir=config_dir)
            generated.append(p)
        except Exception as exc:
            print(f"Erreur pour l'entité {e}: {exc}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Générateur de notebooks EDA par entité")
    parser.add_argument("entity_name", nargs="?", default=None, help="Nom de l'entité (ou none pour toutes)")
    parser.add_argument("--output-dir", default="notebooks", help="Dossier de sortie")
    parser.add_argument("--config-dir", default="config", help="Dossier de configuration")
    parser.add_argument("--all", action="store_true", help="générer pour toutes les entités actives")
    args = parser.parse_args()

    if args.all:
        build_notebooks_for_all_entities(output_dir=args.output_dir, config_dir=args.config_dir, active_only=True)
    elif args.entity_name:
        generate_notebook(args.entity_name, output_dir=args.output_dir, config_dir=args.config_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
