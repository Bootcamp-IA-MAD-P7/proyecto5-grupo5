import numpy as np
import pandas as pd

SERVICE_COLS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]
PROTECTION_COLS = ["Online Security", "Online Backup", "Device Protection"]
STREAMING_COLS = ["Streaming TV", "Streaming Movies"]
TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]


def add_num_services(df):
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    return df


def add_protection_services(df):
    df["ProtectionServices"] = (df[PROTECTION_COLS] == "Yes").sum(axis=1)
    return df


def add_streaming_services(df):
    df["StreamingServices"] = (df[STREAMING_COLS] == "Yes").sum(axis=1)
    return df


def add_tenure_group(df):
    df["TenureGroup"] = pd.cut(
        df["Tenure Months"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )
    return df


def add_avg_monthly_spend(df):
    df["AvgMonthlySpend"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"] / df["Tenure Months"],
        0,
    )
    return df


def add_has_internet_service(df):
    df["HasInternetService"] = (df["Internet Service"] != "No").astype(int)
    return df


def add_all_features(df):
    df = add_num_services(df)
    df = add_protection_services(df)
    df = add_streaming_services(df)
    df = add_tenure_group(df)
    df = add_avg_monthly_spend(df)
    df = add_has_internet_service(df)
    return df
