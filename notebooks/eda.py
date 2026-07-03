# %% [markdown]
# # Exploratory Data Analysis Report

# %%
from pathlib import Path

import pandas as pd
import sweetviz as sv

# %% [markdown]
# ## Configuration

# %%
DATASET_PATH = Path("../data/raw/Telco_customer_churn.xlsx")
REPORTS_DIR = Path("../reports")
REPORT_PATH = REPORTS_DIR / "eda_report.html"

# %% [markdown]
# ## Load dataset

# %%
df = pd.read_excel(DATASET_PATH)

df.head()
# %%
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce",
)

df["Total Charges"].isna().sum()
df = df.dropna(subset=["Total Charges"])
# %% [markdown]
# ## Generate report

# %%
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

report = sv.analyze(df)

report.show_html(
    filepath=str(REPORT_PATH),
    open_browser=False,
)

print(f"Report generated: {REPORT_PATH.resolve()}")
