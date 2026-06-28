
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from config import MODELS_PATH, PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load artefacts once at startup ────────────────────────────────────────────
# Loading at import time = loaded once when server starts,
# not on every request. Critical for performance.

_model    = None
_scaler   = None
_metadata = None
_features = None


def _load_artefacts():
    global _model, _scaler, _metadata, _features

    model_path  = MODELS_PATH / 'final_model.pkl'
    meta_path   = MODELS_PATH / 'final_model_metrics.json'
    scaler_path = MODELS_PATH / 'scaler.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    _model = joblib.load(model_path)
    logger.info(f"Model loaded: {_model.__class__.__name__}")

    if scaler_path.exists():
        _scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded")

    if meta_path.exists():
        with open(meta_path) as f:
            _metadata = json.load(f)
        _features = _metadata.get('features', [])
        logger.info(f"Metadata loaded — {len(_features)} features expected")


_load_artefacts()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(data: dict | pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw input for the model.
    Must mirror notebook 02 exactly — any difference = wrong predictions.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Add any missing expected features with 0
    if _features:
        for col in _features:
            if col not in df.columns:
                df[col] = 0
                logger.warning(f"Missing feature filled with 0: {col}")
        df = df[_features]   # enforce correct column order

    # Apply scaler
    if _scaler is not None:
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = _scaler.transform(df[numeric_cols])

    return df


# ── Core prediction functions ─────────────────────────────────────────────────
def predict_single(data: dict) -> dict:
    """Predict price for one house. Returns price in dollars."""
    df       = preprocess(data)
    log_pred = _model.predict(df)[0]
    price    = float(np.expm1(log_pred))   # reverse log transform

    return {
        'predicted_price': round(price, 2),
        'predicted_price_formatted': f'${price:,.0f}',
        'model': _model.__class__.__name__,
        'log_prediction': float(log_pred)
    }


def predict_batch(data: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Predict prices for multiple houses."""
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()

    original_df       = df.copy()
    processed         = preprocess(df)
    log_preds         = _model.predict(processed)
    prices            = np.expm1(log_preds)

    original_df['predicted_price']           = prices.round(2)
    original_df['predicted_price_formatted'] = [f'${p:,.0f}' for p in prices]

    logger.info(f"Batch prediction: {len(df)} houses")
    return original_df


def get_model_info() -> dict:
    """Return model metadata — used by health check and /info endpoint."""
    return {
        'model_class':  _model.__class__.__name__,
        'task':         'regression',
        'target':       'SalePrice (USD)',
        'n_features':   len(_features) if _features else 'unknown',
        'RMSE':         _metadata.get('RMSE') if _metadata else None,
        'MAE':          _metadata.get('MAE')  if _metadata else None,
        'MAPE':         _metadata.get('MAPE') if _metadata else None,
        'R2':           _metadata.get('R2')   if _metadata else None,
        'status':       'ready'
    }
