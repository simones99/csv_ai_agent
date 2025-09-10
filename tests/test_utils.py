"""
Test utilities for CSV AI Agent.

This module contains tests for utility functions:
- memory_monitor
- data_optimizer
- validation

Ensures that helper modules work correctly.
"""
import pytest
import pandas as pd

from utils.memory_monitor import memory_monitor
from utils.data_optimizer import optimizer
from utils.validation import quick_validate_csv, validate_dataframe
from config import config

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'a': range(10),
        'b': [None if i%2==0 else i for i in range(10)],
        'c': ['x','y','z']*3 + ['x']
    })


def test_quick_validate_csv(tmp_path, sample_df):
    # Create temporary CSV file
    test_file = tmp_path / "test.csv"
    sample_df.to_csv(test_file, index=False)

    # Should be valid
    valid, message = quick_validate_csv(str(test_file))
    assert valid
    assert "valid" in message.lower()

    # Test too large file
    old_limit = config.MAX_FILE_SIZE_MB
    config.MAX_FILE_SIZE_MB = 0.000001  # Riduci drasticamente il limite
    # Crea un file molto grande
    big_file = tmp_path / "big.csv"
    with open(big_file, "w") as f:
        f.write("col1\n" + ("x\n" * 20000))  # ~120KB
    valid, message = quick_validate_csv(str(big_file))
    assert not valid
    assert "too large" in message.lower()
    # Reset limit
    config.MAX_FILE_SIZE_MB = old_limit


def test_validate_dataframe(sample_df):
    # Validate well-formed DataFrame
    result = validate_dataframe(sample_df)
    assert result['valid']
    assert isinstance(result['stats'], dict)
    assert 'shape' in result['stats']

    # Introduce high nulls to trigger issues
    bad_df = sample_df.copy()
    bad_df['b'] = [None]*len(bad_df)  # Tutti i valori nulli
    result2 = validate_dataframe(bad_df)
    assert not result2['valid'] or result2['issues']


def test_data_optimizer(sample_df):
    # Test optimizing a small DataFrame
    df_opt, stats = optimizer.optimize_dataframe(sample_df)
    assert df_opt.shape == sample_df.shape
    assert 'reduction_percent' in stats
    assert isinstance(stats['optimization_log'], list)


def test_memory_monitor(sample_df):
    # Test dataframe memory check
    size_mb = sample_df.memory_usage(deep=True).sum()/(1024**2)
    check = memory_monitor.check_memory_for_dataframe(size_mb)
    assert isinstance(check, dict)
    assert 'can_load' in check

