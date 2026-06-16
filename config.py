from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT                 = Path(__file__).parent
RAW_DATA_PATH        = ROOT / 'data' / 'raw'
PROCESSED_DATA_PATH  = ROOT / 'data' / 'processed'
EXTERNAL_DATA_PATH   = ROOT / 'data' / 'external'
MODELS_PATH          = ROOT / 'models'
FIGURES_PATH         = ROOT / 'reports' / 'figures'
SUBMISSIONS_PATH     = ROOT / 'data' / 'external'

# ── Data ───────────────────────────────────────────────────────────────────────
TARGET       = 'SalePrice'
ID_COL       = 'Id'
DROP_COLS    = ['Id']
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

# ── Model ──────────────────────────────────────────────────────────────────────
CV_FOLDS = 5
SCORING  = 'neg_root_mean_squared_error'
