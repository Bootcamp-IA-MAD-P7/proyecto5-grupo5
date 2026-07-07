import pandas as pd

BINARY_MAPPINGS = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0,
}

ORDINAL_ENCODINGS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
}


def apply_binary_encoding(df, binary_cols):
    for col in binary_cols:
        unique_vals = set(df[col].unique())
        mapping = {k: v for k, v in BINARY_MAPPINGS.items() if k in unique_vals}
        df[col] = df[col].map(mapping).astype(int)
    return df


def apply_ordinal_encoding(df, ordinal_mapping=None):
    if ordinal_mapping is None:
        ordinal_mapping = ORDINAL_ENCODINGS
    for col, order in ordinal_mapping.items():
        if col in df.columns:
            mapping = {label: idx for idx, label in enumerate(order)}
            df[col] = df[col].map(mapping).astype(int)
    return df


def classify_columns(df):
    categorical_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
    nominal_cols = [c for c in categorical_cols if df[c].nunique() > 2]
    return categorical_cols, binary_cols, nominal_cols


def remove_collinear_ohe(df, base_col, ohe_prefix):
    ohe_cols = [c for c in df.columns if c.startswith(ohe_prefix)]
    for col in ohe_cols:
        if col in df.columns and df[col].nunique() == 2:
            corr_val = df[col].corr(df[base_col])
            if abs(corr_val) > 0.99:
                df.drop(columns=[col], inplace=True)
                print(f"Dropped '{col}' (r={corr_val:.2f} with {base_col})")
    return df
