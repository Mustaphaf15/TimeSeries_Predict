import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from typing import Dict, List

def create_intro_section(entity_name: str, config: Dict) -> List:
    """Crée la section d'introduction du notebook."""
    content = f"""
# Analyse Exploratoire (EDA) pour l'entité : {entity_name}

Ce notebook présente une analyse exploratoire complète de la série temporelle pour l'entité `{entity_name}`.

**Objectif** : Comprendre les caractéristiques de la série (tendance, saisonnalité, anomalies) afin de préparer le terrain pour la modélisation et de recommander un pipeline de prétraitement `sktime`.

**Période d'analyse** : de `{config.get('date', {}).get('start_backtest', 'N/A')}` à la dernière date disponible.
    """
    return [new_markdown_cell(content)]

def create_data_loading_section(entity_name: str, target_column: str) -> List:
    """Crée la section de chargement des données."""
    markdown = "## 1. Chargement des données\n\nChargeons les données de backtest pour notre entité."
    code = f"""
from src.data_loading.loaders import load_entity_data
import warnings
warnings.filterwarnings('ignore')

backtest_df, _ = load_entity_data("{entity_name}")
target_column = "{target_column}"
series = backtest_df[[target_column]]

print("Données chargées :")
series.info()
series.head()
    """
    return [new_markdown_cell(markdown), new_code_cell(code)]

def create_data_quality_section() -> List:
    """Crée la section d'analyse de la qualité des données."""
    markdown = "## 2. Qualité des Données"
    code = """
import matplotlib.pyplot as plt
import seaborn as sns

# Vérification des valeurs manquantes
missing_values = series.isnull().sum()
print(f"Valeurs manquantes :\\n{missing_values}")

# Visualisation des valeurs manquantes
plt.figure(figsize=(12, 6))
sns.heatmap(series.isnull(), cbar=False, cmap='viridis')
plt.title("Visualisation des valeurs manquantes")
plt.show()
    """
    return [new_markdown_cell(markdown), new_code_cell(code)]

def create_descriptive_eda_section() -> List:
    """Crée la section de l'EDA descriptive."""
    markdown = "## 3. Analyse Descriptive (EDA)"
    code = """
# Statistiques descriptives
print("Statistiques descriptives :")
print(series.describe())

# Distribution de la variable cible
plt.figure(figsize=(12, 6))
sns.histplot(series[target_column], kde=True)
plt.title("Distribution de la variable cible")
plt.show()

# Evolution temporelle
plt.figure(figsize=(18, 8))
series[target_column].plot()
plt.title("Évolution temporelle de la série")
plt.ylabel(target_column)
plt.xlabel("Date")
plt.grid(True)
plt.show()
    """
    return [new_markdown_cell(markdown), new_code_cell(code)]

def create_recommendations_section() -> List:
    """Crée la section des recommandations de preprocessing."""
    markdown = "## 7. Recommandations de Preprocessing"
    code = """
from src.notebook_generator.recommendations import analyze_series_and_recommend_transformers as recommend_transformers

recommendations = recommend_transformers(series, target_column)
print("Basé sur l'analyse, voici quelques recommandations pour le preprocessing :")
for group, recos in recommendations.items():
    if recos:
        print(f"\\n### Recommandations pour `{group}`:")
        for reco in recos:
            # Créer une copie pour ne pas modifier l'original
            reco_copy = reco.copy()
            reason = reco_copy.pop('reason', 'N/A')
            transform_type = reco_copy.pop('type', 'N/A')
            transform_name = reco_copy.pop('transform', 'N/A')

            # Formatter la configuration YAML pour l'affichage
            config_str = yaml.dump({transform_type: reco_copy}, indent=2, allow_unicode=True)

            print(f"- **{transform_name}** ({transform_type}): {reason}")
            print(f"  Configuration suggérée :\\n```yaml\\n{config_str}\\n```")

"""
    return [new_markdown_cell(markdown), new_code_cell(code)]

def create_seasonality_section() -> List:
    """Crée la section d'analyse de la saisonnalité."""
    markdown = "## 4. Analyse de la Saisonnalité"
    code = """
from statsmodels.tsa.seasonal import seasonal_decompose

# Décomposition STL (Seasonal-Trend-Loess)
decomposition = seasonal_decompose(series[target_column].dropna(), model='additive', period=7)
fig = decomposition.plot()
fig.set_size_inches(14, 8)
plt.show()
    """
    return [new_markdown_cell(markdown), new_code_cell(code)]

def create_autocorrelation_section() -> List:
    """Crée la section d'analyse de l'autocorrélation."""
    markdown = "## 5. Analyse de l'Autocorrélation"
    code = """
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
plot_acf(series[target_column].dropna(), ax=ax1, lags=40)
plot_pacf(series[target_column].dropna(), ax=ax2, lags=40)
plt.show()
    """
    return [new_markdown_cell(markdown), new_code_cell(code)]

def create_anomaly_section() -> List:
    """Crée la section de détection d'anomalies."""
    markdown = "## 6. Détection d'Anomalies"
    code = """
Q1 = series[target_column].quantile(0.25)
Q3 = series[target_column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies = series[(series[target_column] < lower_bound) | (series[target_column] > upper_bound)]
print(f"Nombre d'anomalies détectées (basé sur l'IQR) : {len(anomalies)}")
print("Anomalies :")
print(anomalies)

plt.figure(figsize=(18, 8))
plt.plot(series.index, series[target_column], label='Série originale')
plt.scatter(anomalies.index, anomalies[target_column], color='red', label='Anomalies')
plt.title("Détection d'anomalies")
plt.legend()
plt.show()
    """
    return [new_markdown_cell(markdown), new_code_cell(code)]
