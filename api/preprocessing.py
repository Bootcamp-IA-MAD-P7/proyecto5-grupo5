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
STREAMING_COLS = ["Streaming TV", "Streaming Movies"]
TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]
BINARY_MAPPINGS = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
ORDINAL_ENCODINGS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "TenureGroup": TENURE_LABELS,
}
OHE_COLS = ["Payment Method", "Internet Service"]
OHE_DROP_FIRST = True
OHE_CATEGORIES = {
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
    "Internet Service": ["DSL", "Fiber optic", "No"],
}


def add_features(df):
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["StreamingServices"] = (df[STREAMING_COLS] == "Yes").sum(axis=1)
    df["TenureGroup"] = pd.cut(
        df["Tenure Months"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )
    df["HasInternetService"] = (df["Internet Service"] != "No").astype(int)
    df["AvgMonthlySpend"] = np.where(
        df["Tenure Months"] > 0,
        df["Total Charges"] / df["Tenure Months"],
        0,
    )
    return df


def encode_binary(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        unique_vals = set(df[col].unique())
        mapping = {k: v for k, v in BINARY_MAPPINGS.items() if k in unique_vals}
        if mapping:
            df[col] = df[col].map(mapping).astype(int)
    return df


def encode_ordinal(df):
    for col, order in ORDINAL_ENCODINGS.items():
        if col in df.columns:
            if df[col].dtype.name == "category":
                df[col] = df[col].cat.codes
            else:
                mapping = {label: idx for idx, label in enumerate(order)}
                df[col] = df[col].map(mapping).astype(int)
    return df


def encode_ohe(df):
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, dtype=int)
    drop_cols = []
    for col, cats in OHE_CATEGORIES.items():
        ref = cats[0]
        ohe_ref = f"{col}_{ref}"
        if ohe_ref in df.columns:
            drop_cols.append(ohe_ref)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def preprocess(df_raw):
    df = df_raw.copy()
    df = add_features(df)
    df = encode_binary(df)
    df = encode_ordinal(df)
    df = encode_ohe(df)
    expected_cols = [
        "Gender", "Senior Citizen", "Partner", "Dependents",
        "Tenure Months", "Phone Service", "Multiple Lines",
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
        "Contract", "Paperless Billing",
        "Monthly Charges", "Total Charges",
        "NumServices", "StreamingServices", "TenureGroup",
        "HasInternetService",
        "Internet Service_Fiber optic",
        "Payment Method_Credit card (automatic)",
        "Payment Method_Electronic check",
        "Payment Method_Mailed check",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]