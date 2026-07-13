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

# --- Paths ---
RAW_DATA = ROOT / "data/raw/Telco_customer_churn.xlsx"
PROCESSED_DIR = ROOT / "data/processed"
REPORTS_DIR = ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "eda_report.html"
CLEAN_PATH = PROCESSED_DIR / "telco_clean.parquet"
MODELS_DIR = ROOT / "models"

# --- Model constants ---
TARGET = "Churn Value"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTUNA_TRIALS = 30

# --- Feature engineering ---
SERVICE_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


# --- Columns for scaling ---
NUMERIC_SCALE_COLS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "AvgMonthlySpend",
    "NumServices",
]

# --- Encoding mappings ---
BINARY_ENCODING_MAP = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0,
}

ORDINAL_ENCODINGS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
}

# --- Binary columns (string columns with 2 values that need Yes/No mapping) ---
BINARY_COLS = [
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Paperless Billing",
    "Gender",
]

# --- Nominal columns for OHE (after binary encoding, only multi-category strings remain) ---
NOMINAL_COLS = [
    "Internet Service",
    "Payment Method",
]

