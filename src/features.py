import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    LabelEncoder, OrdinalEncoder)

logger = logging.getLogger(__name__)


# ─── Numeric Transformations ──────────────────────────────────────────────────
def log_transform(df, cols):
    """Apply log1p to skewed columns. Adds new columns, keeps originals."""
    for col in cols:
        df[f'log_{col}'] = np.log1p(df[col])
    logger.info(f"Log-transformed: {cols}")
    return df


def scale_features(X_train, X_test, method='standard', cols=None):
    """Fit scaler on train, transform both. Never fit on test."""
    cols = cols or X_train.select_dtypes(include='number').columns.tolist()
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols]  = scaler.transform(X_test[cols])
    logger.info(f"Scaled {len(cols)} columns with {method} scaler")
    return X_train, X_test, scaler


def create_interaction(df, col1, col2, operation='multiply'):
    name = f'{col1}_{operation}_{col2}'
    if operation == 'multiply':
        df[name] = df[col1] * df[col2]
    elif operation == 'divide':
        df[name] = df[col1] / (df[col2] + 1e-8)
    elif operation == 'add':
        df[name] = df[col1] + df[col2]
    return df


# ─── Categorical Encoding ─────────────────────────────────────────────────────
def one_hot_encode(df, cols, drop_first=True):
    """Nominal categories with no order: color, city, brand."""
    df = pd.get_dummies(df, columns=cols, drop_first=drop_first)
    logger.info(f"One-hot encoded: {cols}")
    return df


def label_encode(df, cols):
    """Use ONLY for tree-based models, not linear models."""
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info(f"Label encoded: {cols}")
    return df


def target_encode(df, col, target, smoothing=10):
    """Replace category with smoothed mean of target. Fit on train only."""
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(['mean', 'count'])
    stats['smoothed'] = ((stats['count'] * stats['mean'] + smoothing * global_mean)
                          / (stats['count'] + smoothing))
    df[f'{col}_target_enc'] = df[col].map(stats['smoothed'])
    return df


def handle_high_cardinality(df, col, threshold=50, replacement='Other'):
    """Cap a high-cardinality categorical to its top N values."""
    top_cats = df[col].value_counts().nlargest(threshold).index
    df[col] = df[col].where(df[col].isin(top_cats), replacement)
    logger.info(f"Capped {col} to top {threshold} categories")
    return df


# ─── Master Feature Pipeline ──────────────────────────────────────────────────
def build_features(df):
    """Customize per project — example pipeline below."""
    return df
