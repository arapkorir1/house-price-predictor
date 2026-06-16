

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from config import (RAW_DATA_PATH, PROCESSED_DATA_PATH,
                    RANDOM_SEED, TEST_SIZE, TARGET, DROP_COLS)

# Set up logging — much better than print() for tracking what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_raw_data(path=None):
    """Load raw data from disk. Returns unmodified DataFrame."""
    path = path or RAW_DATA_PATH
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def load_processed(filename):
    """Load a previously saved processed file."""
    path = PROCESSED_DATA_PATH / filename
    logger.info(f"Loading processed data: {path}")
    return pd.read_csv(path)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_schema(df, expected_cols):
    """
    Check that expected columns exist.
    Raises ValueError if any are missing.
    Call this right after loading raw data.
    """
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Schema validation passed.")
    return True


def summarize(df):
    """
    Print a quick health check of the DataFrame.
    Use at the start of any notebook to verify the data loaded correctly.
    """
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values (%):\n"
          f"{(df.isnull().mean() * 100).sort_values(ascending=False)}")
    print(f"\nDuplicates: {df.duplicated().sum()}")


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def drop_irrelevant_cols(df, cols=None):
    """Drop ID columns, leakage columns, or anything specified in config."""
    cols = cols or DROP_COLS
    existing = [c for c in cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")
    return df


def handle_missing(df, strategy='median', fill_value=None):
    """
    Impute missing values.
    strategy: 'median', 'mean', 'mode', 'constant'
    fill_value: used only when strategy='constant'
    """
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    if strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    elif strategy == 'constant':
        df = df.fillna(fill_value)

    # Categorical missing → 'Unknown' is usually the right default
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    logger.info(f"Missing values handled with strategy='{strategy}'")
    return df


def remove_outliers_iqr(df, cols, multiplier=1.5):
    """
    Remove rows where values fall outside IQR bounds.
    Use carefully — only for clear data errors, not natural outliers.
    """
    initial_len = len(df)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = initial_len - len(df)
    logger.info(f"Removed {removed:,} outlier rows ({removed/initial_len:.1%})")
    return df.reset_index(drop=True)


def fix_dtypes(df, int_cols=None, float_cols=None, cat_cols=None):
    """Explicitly cast columns to correct types."""
    if int_cols:
        df[int_cols] = df[int_cols].astype(int)
    if float_cols:
        df[float_cols] = df[float_cols].astype(float)
    if cat_cols:
        df[cat_cols] = df[cat_cols].astype('category')
    return df


# ─── Splitting ────────────────────────────────────────────────────────────────

def split_features_target(df, target=None):
    """Separate features (X) and target (y)."""
    target = target or TARGET
    X = df.drop(columns=[target])
    y = df[target]
    logger.info(f"X: {X.shape}, y: {y.shape}")
    return X, y


def split_train_test(X, y, test_size=None, random_state=None):
    """Standard train/test split."""
    test_size = test_size or TEST_SIZE
    random_state = random_state or RANDOM_SEED
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ─── Saving ───────────────────────────────────────────────────────────────────

def save_processed(df, filename):
    """Save a processed DataFrame to the processed folder."""
    path = PROCESSED_DATA_PATH / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path} ({df.shape[0]:,} rows)")


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def clean_data(df):
    """
    Run the full cleaning pipeline in one call.
    This is what your notebook calls — not the individual steps.
    """
    df = drop_irrelevant_cols(df)
    df = handle_missing(df, strategy='median')
    df = fix_dtypes(df)
    return df
