import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from config import (TRAIN_FILE, PROCESSED_DATA_PATH,
                     RANDOM_SEED, TEST_SIZE, TARGET, DROP_COLS)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Loading ──────────────────────────────────────────────────────────────────
def load_raw_data(path=None):
    """Load raw data from disk. Returns unmodified DataFrame."""
    path = path or TRAIN_FILE
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
def summarize(df):
    """Quick health check of a DataFrame."""
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    print(f"\nMissing values (%):\n{missing[missing > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum()}")


# ─── Cleaning ─────────────────────────────────────────────────────────────────
def drop_irrelevant_cols(df, cols=None):
    """Drop ID columns or anything specified in config."""
    cols = cols or DROP_COLS
    existing = [c for c in cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info(f"Dropped columns: {existing}")
    return df


def handle_missing(df, strategy='median'):
    """Impute missing values. strategy: 'median', 'mean', 'mode'."""
    numeric_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    if strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df[cat_cols] = df[cat_cols].fillna('Unknown')
    logger.info(f"Missing values handled with strategy='{strategy}'")
    return df


def remove_outliers_iqr(df, cols, multiplier=1.5):
    """Remove rows outside IQR bounds for given columns."""
    initial_len = len(df)
    for col in cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    removed = initial_len - len(df)
    logger.info(f"Removed {removed:,} outlier rows")
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
    target = target or TARGET
    X = df.drop(columns=[target])
    y = df[target]
    logger.info(f"X: {X.shape}, y: {y.shape}")
    return X, y


def split_train_test(X, y, test_size=None, random_state=None):
    test_size = test_size or TEST_SIZE
    random_state = random_state or RANDOM_SEED
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ─── Saving ───────────────────────────────────────────────────────────────────
def save_processed(df, filename):
    path = PROCESSED_DATA_PATH / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path} ({df.shape[0]:,} rows)")


# ─── Master Pipeline ──────────────────────────────────────────────────────────
def clean_data(df):
    """Run the full basic cleaning pipeline in one call."""
    df = drop_irrelevant_cols(df)
    df = handle_missing(df, strategy='median')
    return df
