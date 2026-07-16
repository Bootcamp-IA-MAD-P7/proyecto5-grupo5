import numpy as np
import pandas as pd

LEAKAGE_COLS = [
    "Churn Score",
    "Churn Reason",
    "Churn Category",
    "Customer Status",
    "CLTV",
]

GEO_COLS = ["Lat Long", "Latitude", "Longitude", "Zip Code", "City"]

INTERNET_SERVICE_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["CustomerID"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df


def remove_geographic_columns(df: pd.DataFrame) -> pd.DataFrame:
    geo_cols = [c for c in GEO_COLS if c in df.columns]
    if geo_cols:
        df = df.drop(columns=geo_cols)
    return df


def remove_redundant_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Churn Label" in df.columns:
        df = df.drop(columns=["Churn Label"])
    return df


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    n_before = df["Total Charges"].isna().sum()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    return df


def drop_missing_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Total Charges"])
    return df


def standardize_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in INTERNET_SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")
    if "Multiple Lines" in df.columns:
        df["Multiple Lines"] = df["Multiple Lines"].replace("No phone service", "No")
    return df


def convert_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    if "Senior Citizen" in df.columns:
        df["Senior Citizen"] = (
            df["Senior Citizen"].map({"Yes": 1, "No": 0}).astype(int)
        )
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = remove_leakage_columns(df)
    df = remove_identifier_columns(df)
    df = remove_constant_columns(df)
    df = remove_geographic_columns(df)
    df = remove_redundant_target_columns(df)
    df = convert_total_charges(df)
    df = drop_missing_total_charges(df)
    df = standardize_service_columns(df)
    df = convert_senior_citizen(df)
    return df