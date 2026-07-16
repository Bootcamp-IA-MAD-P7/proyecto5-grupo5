import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.config import SERVICE_COLS


def add_num_services(df):
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
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


def add_selected_features(df):
    df = add_num_services(df)
    df = add_avg_monthly_spend(df)
    df = add_has_internet_service(df)
    return df


def create_selected_feature_transformer() -> FunctionTransformer:
    return FunctionTransformer(add_selected_features, validate=False)