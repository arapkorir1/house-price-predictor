import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    LabelEncoder, OrdinalEncoder)
import logging

logger = logging.getLogger(__name__)


# ─── Numeric Transformations ──────────────────────────────────────────────────

def log_transform(df, cols):
    """Apply log1p to right-skewed columns. Adds new columns, keeps originals."""
    for col in cols:
        df[f'log_{col}'] = np.log1p(df[col])
        logger.info(f"Log-transformed: {col} → log_{col}")
    return df


def scale_features(X_train, X_test, method='standard', cols=None):
    """
    Fit scaler on train, transform both train and test.
    CRITICAL: Always fit on train only to avoid data leakage.

    method: 'standard' (zero mean, unit variance)
             'minmax' (0-1 range)
    """
    cols = cols or X_train.select_dtypes(include='number').columns.tolist()

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    X_train[cols] = scaler.fit_transform(X_train[cols])  # fit + transform
    X_test[cols] = scaler.transform(X_test[cols])         # transform only

    logger.info(f"Scaled {len(cols)} columns with {method} scaler")
    return X_train, X_test, scaler  # return scaler so you can save it


def create_interaction(df, col1, col2, operation='multiply'):
    """Create interaction features between two numeric columns."""
    name = f'{col1}_{operation}_{col2}'
    if operation == 'multiply':
        df[name] = df[col1] * df[col2]
    elif operation == 'divide':
        df[name] = df[col1] / (df[col2] + 1e-8)  # avoid div by zero
    elif operation == 'add':
        df[name] = df[col1] + df[col2]
    return df


def bin_numeric(df, col, bins, labels, new_col=None):
    """Convert a continuous column into discrete bins."""
    new_col = new_col or f'{col}_binned'
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    return df


# ─── Categorical Encoding ─────────────────────────────────────────────────────

def one_hot_encode(df, cols, drop_first=True):
    """
    One-hot encode categorical columns.
    Use for nominal categories (no order): color, city, brand.
    drop_first avoids multicollinearity.
    """
    df = pd.get_dummies(df, columns=cols, drop_first=drop_first)
    logger.info(f"One-hot encoded: {cols}")
    return df


def label_encode(df, cols):
    """
    Assign integer codes to categories.
    Use ONLY for tree-based models (RF, XGBoost) — not linear models.
    """
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info(f"Label encoded: {cols}")
    return df


def ordinal_encode(df, cols, category_order):
    """
    Encode with meaningful order.
    Use when order matters: low < medium < high.

    category_order example:
    [['low', 'medium', 'high']]  — one list per column
    """
    enc = OrdinalEncoder(categories=category_order)
    df[cols] = enc.fit_transform(df[cols])
    logger.info(f"Ordinal encoded: {cols}")
    return df


def target_encode(df, col, target, smoothing=10):
    """
    Replace category with mean of target for that category.
    Very powerful for high-cardinality categoricals.
    smoothing blends category mean with global mean for rare categories.
    WARNING: Must only be fit on training data.
    """
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(['mean', 'count'])
    stats['smoothed'] = (
        (stats['count'] * stats['mean'] + smoothing * global_mean)
        / (stats['count'] + smoothing)
    )
    df[f'{col}_target_enc'] = df[col].map(stats['smoothed'])
    return df


def handle_high_cardinality(df, col, threshold=50, replacement='Other'):
    """
    Replace rare categories with 'Other'.
    Use before encoding high-cardinality columns (zip code, product ID).
    """
    top_cats = df[col].value_counts().nlargest(threshold).index
    df[col] = df[col].where(df[col].isin(top_cats), replacement)
    logger.info(f"Capped {col} to top {threshold} categories")
    return df


# ─── Master Feature Pipeline ──────────────────────────────────────────────────

def build_features(df):
    """
    Apply full feature engineering pipeline.
    """
    from config import TARGET


    df = log_transform(df, cols=['income', 'price'])
    df = handle_high_cardinality(df, col='city', threshold=30)
    df = one_hot_encode(df, cols=['city', 'category'])
    return df
