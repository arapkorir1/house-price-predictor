# tests/test_predict.py
import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from src.predict import predict_single, predict_batch, get_model_info


def make_empty_input():
    """Minimal input — all features missing, filled with 0."""
    return {}


def make_good_house():
    """Above-average house features."""
    return {
        "OverallQual":   8,
        "GrLivArea":     2000,
        "GarageCars":    2,
        "TotalBsmtSF":   1000,
        "FullBath":      2,
        "YearBuilt":     2005,
        "TotalSF":       3000,
        "Quality_x_Area": 16000,
    }


def make_poor_house():
    """Below-average house features."""
    return {
        "OverallQual":   4,
        "GrLivArea":     900,
        "GarageCars":    0,
        "TotalBsmtSF":   400,
        "FullBath":      1,
        "YearBuilt":     1960,
        "TotalSF":       1300,
        "Quality_x_Area": 3600,
    }


# ── Model info tests ──────────────────────────────────────────────────────────
def test_model_info_returns_dict():
    info = get_model_info()
    assert isinstance(info, dict)


def test_model_info_has_required_keys():
    info = get_model_info()
    assert 'model_class' in info
    assert 'status' in info
    assert info['status'] == 'ready'


# ── Single prediction tests ───────────────────────────────────────────────────
def test_predict_single_returns_dict():
    result = predict_single(make_empty_input())
    assert isinstance(result, dict)


def test_predict_single_has_price():
    result = predict_single(make_good_house())
    assert 'predicted_price' in result
    assert result['predicted_price'] > 0


def test_predict_single_price_is_positive():
    result = predict_single(make_good_house())
    assert result['predicted_price'] > 0


def test_predict_single_price_in_realistic_range():
    """House prices in Ames Iowa should be between $50k and $1M."""
    result = predict_single(make_good_house())
    price  = result['predicted_price']
    assert 50_000 < price < 1_000_000, f"Price ${price:,.0f} outside realistic range"


def test_predict_single_good_house_more_than_poor():
    """Better house should predict higher price."""
    good_price = predict_single(make_good_house())['predicted_price']
    poor_price = predict_single(make_poor_house())['predicted_price']
    assert good_price > poor_price, (
        f"Good house (${good_price:,.0f}) should cost more "
        f"than poor house (${poor_price:,.0f})")


def test_predict_single_formatted_price_is_string():
    result = predict_single(make_good_house())
    assert isinstance(result['predicted_price_formatted'], str)
    assert '$' in result['predicted_price_formatted']


# ── Batch prediction tests ────────────────────────────────────────────────────
def test_predict_batch_returns_dataframe():
    import pandas as pd
    data   = [make_good_house(), make_poor_house()]
    result = predict_batch(data)
    assert isinstance(result, pd.DataFrame)


def test_predict_batch_correct_row_count():
    import pandas as pd
    data   = [make_good_house(), make_poor_house(), make_empty_input()]
    result = predict_batch(data)
    assert len(result) == 3


def test_predict_batch_no_missing_predictions():
    data   = [make_good_house(), make_poor_house()]
    result = predict_batch(data)
    assert result['predicted_price'].isnull().sum() == 0


def test_predict_batch_all_prices_positive():
    data   = [make_good_house(), make_poor_house(), make_empty_input()]
    result = predict_batch(data)
    assert (result['predicted_price'] > 0).all()
