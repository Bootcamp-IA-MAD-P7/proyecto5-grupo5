import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_numerical_feature(data, col, target_col):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(data[col], kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {col}")
    axes[0].set_xlabel(col)

    sns.boxplot(y=data[col], ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}")

    churn_labels = data[target_col].map({0: "No", 1: "Yes"})
    sns.boxplot(x=churn_labels, y=data[col], ax=axes[2])
    axes[2].set_title(f"{col} by Churn")
    axes[2].set_xlabel("Churn")
    axes[2].set_ylabel(col)

    plt.tight_layout()
    plt.show()

    print(
        f"{col}: mean={data[col].mean():.2f}, median={data[col].median():.2f}, "
        f"std={data[col].std():.2f}, min={data[col].min():.2f}, max={data[col].max():.2f}"
    )


def plot_categorical_feature(data, col, target_col):
    print(f"\n--- {col} ---")
    freq = data[col].value_counts()
    print(freq)
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    order = data[col].value_counts().index
    sns.countplot(data=data, x=col, order=order, ax=axes[0])
    axes[0].set_title(f"Countplot of {col}")
    axes[0].tick_params(axis="x", rotation=45)

    target_label = data[target_col].map({0: "No", 1: "Yes"})
    crosstab = pd.crosstab(data[col], target_label, normalize="index") * 100
    crosstab.plot(kind="bar", stacked=True, ax=axes[1], color=["#3498db", "#e74c3c"])
    axes[1].set_title(f"{col} vs Churn (Stacked %)")
    axes[1].set_ylabel("Percentage")
    axes[1].legend(title="Churn", loc="upper right")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
