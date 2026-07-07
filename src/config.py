import sys
from pathlib import Path


def _find_root() -> Path:
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DATA = ROOT / "data/raw/Telco_customer_churn.xlsx"
PROCESSED_DIR = ROOT / "data/processed"
REPORTS_DIR = ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "eda_report.html"
CLEAN_PATH = PROCESSED_DIR / "telco_clean.parquet"
ML_PATH = PROCESSED_DIR / "telco_ml.parquet"
ML_SCALED_PATH = PROCESSED_DIR / "telco_ml_scaled.parquet"

MODELS_DIR = ROOT / "models"
TARGET = "Churn Value"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "AvgMonthlySpend",
]
