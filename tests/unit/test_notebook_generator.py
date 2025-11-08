# tests/unit/test_notebook_generator.py

import sys
from pathlib import Path
import pytest
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import pandas as pd
import numpy as np

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.notebook_generator.recommendations import analyze_series_and_recommend_transformers
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
from src.notebook_generator.generator import generate_notebook

@pytest.fixture
def sample_df():
    """Fixture for a sample time series DataFrame."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq='D'))
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum() + 50
    })
    return data.set_index('date')

def test_recommend_imputer(sample_df):
    """Test that Imputer is recommended when there are missing values."""
    df_with_nan = sample_df.copy()
    df_with_nan.iloc[5, 0] = np.nan
    recos = analyze_series_and_recommend_transformers(df_with_nan, 'value')
    assert any(r['transform'] == 'Imputer' for r in recos['target_transform'])

def test_recommend_hampel_filter(sample_df):
    """Test that HampelFilter is recommended when there are outliers."""
    df_with_outlier = sample_df.copy()
    df_with_outlier.iloc[10, 0] = 200  # Add a significant outlier
    recos = analyze_series_and_recommend_transformers(df_with_outlier, 'value')
    assert any(r['transform'] == 'HampelFilter' for r in recos['target_transform'])

def test_recommend_fourier_features():
    """Test that FourierFeatures is recommended for seasonality."""
    dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=400, freq='D'))
    # Create a series with very strong and clear weekly seasonality
    num_weeks = 400 / 7
    values = np.sin(np.linspace(0, num_weeks * 2 * np.pi, 400)) * 15 + np.random.randn(400) * 2
    df = pd.DataFrame({'date': dates, 'value': values}).set_index('date')
    recos = analyze_series_and_recommend_transformers(df, 'value')
    assert any(r['transform'] == 'FourierFeatures' for r in recos['exog_transform'])


def test_create_intro_section():
    """Test the intro section creation."""
    cells = create_intro_section("test_entity", {"date": {"start_backtest": "2023-01-01"}})
    assert len(cells) == 1
    assert cells[0].cell_type == 'markdown'
    assert "test_entity" in cells[0].source

def test_generate_notebook(tmp_path):
    """Test the full notebook generation process."""
    entity_name = "test_entity_A"
    output_dir = tmp_path

    try:
        generate_notebook(entity_name, output_dir)
        output_file = output_dir / f"eda_report_{entity_name}.ipynb"
        assert output_file.exists()
    except FileNotFoundError:
        pytest.skip(f"Skipping full notebook generation test: config file for '{entity_name}' not found.")
    except Exception as e:
        pytest.fail(f"generate_notebook failed with an unexpected exception: {e}")
