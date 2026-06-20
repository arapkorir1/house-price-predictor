
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.data import drop_irrelevant_cols, handle_missing


def test_drop_cols_removes_specified():
    df = pd.DataFrame({'Id': [1, 2], 'SalePrice': [100, 200], 'area': [50, 60]})
    result = drop_irrelevant_cols(df, cols=['Id'])
    assert 'Id' not in result.columns
    assert 'SalePrice' in result.columns


def test_handle_missing_leaves_no_nulls_numeric():
    df = pd.DataFrame({'a': [1.0, None, 3.0], 'b': [None, 2.0, 3.0]})
    result = handle_missing(df, strategy='median')
    assert result[['a', 'b']].isnull().sum().sum() == 0


def test_handle_missing_fills_categorical():
    df = pd.DataFrame({'city': ['London', None, 'Paris']})
    result = handle_missing(df)
    assert result['city'].isnull().sum() == 0
    assert 'Unknown' in result['city'].values
