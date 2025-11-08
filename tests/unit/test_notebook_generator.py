# tests/unit/test_notebook_generator.py

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.notebook_generator.utils import detect_frequency, series_summary
from src.notebook_generator.recommendations import (
    analyze_series_and_recommend_transformers,
    _detect_trend,
    _check_variance_instability,
    _detect_seasonality_multi
)
from src.notebook_generator.generator import generate_notebook

@pytest.fixture
def sample_df():
    """Fixture for a sample time series DataFrame with daily frequency."""
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=400, freq='D'))
    data = pd.DataFrame({
        'value': np.random.randn(400).cumsum() + 50
    }, index=dates)
    return data

# --- Tests for utils.py ---

def test_detect_frequency(sample_df):
    assert detect_frequency(sample_df) == 'D'
    df_no_freq = sample_df.reset_index().rename(columns={'index': 'date'})
    assert detect_frequency(df_no_freq, datetime_index_col='date') == 'D'

def test_series_summary(sample_df):
    summary = series_summary(sample_df['value'])
    assert summary['n'] == 400
    assert summary['n_missing'] == 0
    assert 'mean' in summary
    assert 'std' in summary
    assert 'iqr' in summary

# --- Tests for recommendations.py ---

def test_recommend_imputer(sample_df):
    df_with_nan = sample_df.copy()
    df_with_nan.iloc[10, 0] = np.nan
    recos = analyze_series_and_recommend_transformers(df_with_nan, 'value')
    assert any(r['transform'] == 'Imputer' for r in recos['target_transform'])

def test_recommend_outlier_remover(sample_df):
    df_with_outlier = sample_df.copy()
    df_with_outlier.iloc[20, 0] = 500  # Large outlier
    recos = analyze_series_and_recommend_transformers(df_with_outlier, 'value')
    assert any(r['transform'] == 'HampelFilter' for r in recos['target_transform'])

def test_recommend_variance_stabilizer():
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=200, freq='D'))
    # Exponential growth to create variance instability
    values = np.exp(np.linspace(0, 5, 200)) + np.random.randn(200)
    df = pd.DataFrame({'value': values}, index=dates)
    assert _check_variance_instability(df['value'])
    recos = analyze_series_and_recommend_transformers(df, 'value')
    assert any(r['transform'] == 'LogTransformer' for r in recos['target_transform'])

def test_recommend_detrender():
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=200, freq='D'))
    # Linear trend
    values = np.linspace(0, 100, 200) + np.random.randn(200)
    df = pd.DataFrame({'value': values}, index=dates)
    assert _detect_trend(df['value'])
    recos = analyze_series_and_recommend_transformers(df, 'value')
    assert any(r['transform'] == 'Detrender' for r in recos['target_transform'])

def test_recommend_seasonality_transformers():
    dates = pd.to_datetime(pd.date_range(start="2020-01-01", periods=400, freq='D'))
    # Strong weekly seasonality
    num_weeks = 400 / 7
    values = np.sin(np.linspace(0, num_weeks * 2 * np.pi, 400)) * 20 + np.random.randn(400)
    df = pd.DataFrame({'value': values}, index=dates)

    assert 7 in _detect_seasonality_multi(df['value'], [7, 365])

    recos = analyze_series_and_recommend_transformers(df, 'value', freq='D')
    assert any(r['transform'] == 'FourierFeatures' and r['sp'] == 7 for r in recos['exog_transform'])
    assert any(r['transform'] == 'Deseasonalizer' and r['sp'] == 7 for r in recos['target_transform'])

# --- Test for generator.py ---

def test_generate_notebook_resilience(tmp_path):
    """
    Test that the generator can run using a specific config directory.
    """
    entity_name = "test_entity_A"
    output_dir = tmp_path
    config_dir = "tests/fixtures/config"  # Point to the test configs

    try:
        generate_notebook(entity_name, output_dir=output_dir, config_dir=config_dir)
        output_file = output_dir / f"eda_report_{entity_name}.ipynb"
        assert output_file.exists()

        content = output_file.read_text()
        assert "Analyse Exploratoire" in content

    except Exception as e:
        pytest.fail(f"generate_notebook failed unexpectedly: {e}")
