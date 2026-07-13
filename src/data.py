import pandas as pd

from src.config import CLEAN_PATH, RAW_DATA
from src.preprocessing import clean_raw_data


def load_clean_data() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_PATH)
    print(f"Clean data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def load_and_clean_raw_data() -> pd.DataFrame:
    df = pd.read_excel(RAW_DATA)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    df = clean_raw_data(df)
    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} cols")
    return df