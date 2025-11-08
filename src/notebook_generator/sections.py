"""
sections.py

Provides functions that create notebook cells (markdown & code).
Each function returns a list of nbformat cells.
"""
from typing import Dict, List, Any, Optional
from nbformat.v4 import new_markdown_cell, new_code_cell
import yaml
import textwrap


def _imports_cell() -> Any:
    """Centralized imports cell for the generated notebook."""
    code = textwrap.dedent(
        """
        # Imports courants pour l'EDA
        import warnings
        warnings.filterwarnings('ignore')

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('whitegrid')

        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        # utilitaires locaux
        from src.data_loading.loaders import load_entity_data
        """
    )
    return new_code_cell(code)


def create_meta_section(entity_name: str, config: Dict) -> List:
    """Create a technical header with config, generation date, and version."""
    cfg_yaml = yaml.dump(config, default_flow_style=False, allow_unicode=True)
    md = f"""# Analyse Exploratoire (EDA) — {entity_name}

**But** : produire une analyse standardisée et recommander des transformateurs sktime adaptés.

**Configuration de l'entité (extrait)** :

```yaml
{cfg_yaml}
"""
    return [new_markdown_cell(md), _imports_cell()]

def create_data_loading_section(entity_name: str, target_column: str) -> List:
    """Creates a data loading cell that uses the project's loader."""
    md = "## 1) Chargement des données\n\nNous chargeons les données de backtest via load_entity_data."
    code = textwrap.dedent(f"""
    # Chargement des données
    backtest_df, production_df = load_entity_data("{entity_name}")

    # target column
    target_column = "{target_column}"

    # quick check
    print("Dataset shape:", backtest_df.shape)
    display(backtest_df.head(5))

    # series variable expected by following cells
    series = backtest_df[[target_column]].copy()
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_profile_section() -> List:
    """Profile summary: freq, length, missing ratio, simple stats."""
    md = "## 2) Profil général de la série"
    code = textwrap.dedent("""
    # Profil général
    try:
        freq = pd.infer_freq(series.index)
    except Exception:
        freq = None

    print("Fréquence inférée:", freq)
    print("Longueur série:", len(series))
    print("Taux de valeurs manquantes:", series.isnull().mean())
    display(series.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_data_quality_section() -> List:
    md = "## 3) Qualité des données"
    code = textwrap.dedent("""
    # Valeurs manquantes par période
    print(series.isnull().sum())
    plt.figure(figsize=(14,3))
    sns.heatmap(series.isnull(), cbar=False)
    plt.title("Missing values (heatmap)")
    plt.show()
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_descriptive_eda_section() -> List:
    md = "## 4) Analyse descriptive"
    code = textwrap.dedent("""
    # Statistiques & distribution
    display(series.describe())
    plt.figure(figsize=(12,5))
    sns.histplot(series.dropna().iloc[:,0], kde=True)
    plt.title("Distribution de la cible")
    plt.show()

    # Série temporelle
    plt.figure(figsize=(16,5))
    series.iloc[:,0].plot()
    plt.title("Série temporelle — vue globale")
    plt.show()
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_seasonality_section() -> List:
    md = "## 5) Saisonnalité (décomposition)"
    code = textwrap.dedent("""
    # Try decomposition with a few candidate periods
    candidate_periods = [7, 24, 365]
    for p in candidate_periods:
        try:
            if len(series.dropna()) < max(2*p, 20):
                continue
            print(f"--- Décomposition period={p} ---")
            dec = seasonal_decompose(series.iloc[:,0].dropna(), model='additive', period=p, two_sided=False, extrapolate_trend='freq')
            fig = dec.plot()
            fig.set_size_inches(12,6)
            plt.show()
        except Exception as e:
            print("Skipped period", p, ":", e)
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_autocorrelation_section() -> List:
    md = "## 6) Autocorrélation"
    code = textwrap.dedent("""
    plt.figure(figsize=(14,6))
    plot_acf(series.dropna().iloc[:,0], lags=40)
    plt.title("ACF")
    plt.show()

    plt.figure(figsize=(14,6))
    plot_pacf(series.dropna().iloc[:,0], lags=40)
    plt.title("PACF")
    plt.show()
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_anomaly_section() -> List:
    md = "## 7) Détection d'anomalies (plusieurs méthodes)"
    code = textwrap.dedent("""
    # IQR based
    s = series.iloc[:,0]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
    anomalies_iqr = series[(s < lb) | (s > ub)]
    print("Anomalies (IQR):", len(anomalies_iqr))

    # Rolling z-score (local)
    roll_mean = s.rolling(window=7, min_periods=1).mean()
    roll_std = s.rolling(window=7, min_periods=1).std().replace(0, np.nan)
    z = (s - roll_mean) / roll_std
    anomalies_z = series[z.abs() > 3]
    print("Anomalies (rolling z>3):", len(anomalies_z))

    # Visualize
    plt.figure(figsize=(16,5))
    plt.plot(series.index, s, label='serie')
    plt.scatter(anomalies_iqr.index, anomalies_iqr.iloc[:,0], color='red', label='IQR anomalies')
    plt.scatter(anomalies_z.index, anomalies_z.iloc[:,0], color='orange', label='rolling-z anomalies', alpha=0.7)
    plt.legend()
    plt.show()
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_stationarity_section() -> List:
    md = "## 8) Stationnarité (ADF / KPSS)"
    code = textwrap.dedent("""
    from statsmodels.tsa.stattools import adfuller, kpss
    s = series.dropna().iloc[:,0]
    try:
        adf_res = adfuller(s)
        print("ADF p-value:", adf_res[1])
    except Exception as e:
        print("ADF failed:", e)
    try:
        kpss_res = kpss(s, nlags='auto')
        print("KPSS p-value:", kpss_res[1])
    except Exception as e:
        print("KPSS failed:", e)
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_exogenous_analysis_section() -> List:
    md = "## 9) Analyse des exogènes (si présents)"
    code = textwrap.dedent("""
    # Si des colonnes exogènes sont présentes dans backtest_df, afficher corrélations rapides.
    exog_cols = [c for c in backtest_df.columns if c != target_column]
    print("Exog columns detected:", exog_cols)
    if exog_cols:
        display(backtest_df[exog_cols].describe().T)
        # correlation with target
        corr = backtest_df[[target_column] + exog_cols].corr()[target_column].drop(target_column)
        print("Corrélation exog -> target:")
        display(corr.sort_values(ascending=False))
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_recommendations_section() -> List:
    md = "## 10) Recommandations de preprocessing (règles automatiques)"
    code = textwrap.dedent("""
    from src.notebook_generator.recommendations import analyze_series_and_recommend_transformers
    import yaml

    recs = analyze_series_and_recommend_transformers(backtest_df, target_column)
    print("=== Meta / Profil ===")
    display(recs.get('meta', []))

    print("\\n=== Recommandations pour target_transform ===")
    for r in recs.get('target_transform', []):
        reason = r.pop('reason', None)
        print(f"- {r.get('transform')} : {reason}")
        display(r)

    print("\\n=== Recommandations pour exog_transform ===")
    for r in recs.get('exog_transform', []):
        reason = r.pop('reason', None)
        print(f"- {r.get('transform')} : {reason}")
        display(r)

    # Build a suggested YAML pipeline (simple mapping)
    suggested = {'preprocessing': {'target_transform': [], 'exog_transform': []}}
    for r in recs.get('target_transform', []):
        suggested['preprocessing']['target_transform'].append({r.get('type', 'custom'): r})
    for r in recs.get('exog_transform', []):
        suggested['preprocessing']['exog_transform'].append({r.get('type', 'custom'): r})
    print("\\n=== Pipeline YAML suggéré ===")
    print(yaml.dump(suggested, allow_unicode=True, sort_keys=False))
    """)
    return [new_markdown_cell(md), new_code_cell(code)]

def create_summary_section() -> List:
    md = "## 11) Synthèse & Actions recommandées\n\nRésumé automatique des points clés et prochaines étapes pour le data scientist."
    code = textwrap.dedent("""
    # Exemple d'actions (basé sur les recommandations affichées)
    print("1) Vérifier les valeurs manquantes et décider d'une stratégie d'imputation.")
    print("2) Examiner les anomalies détectées (IQR + rolling z-score).")
    print("3) Appliquer les transformateurs recommandés (Log/Detrender/Deseasonalizer) et tester sur un fold.")
    print("4) Générer le YAML final et lancer un backtest.")
    """)
    return [new_markdown_cell(md), new_code_cell(code)]
