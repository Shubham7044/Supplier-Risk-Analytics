# src/preprocess.py
import pandas as pd
import warnings

# Silence pandas date format warnings
warnings.filterwarnings("ignore", message="Could not infer format")

# Load data
df = pd.read_csv("data/supply_chain.csv")

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("/", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

print("✅ Cleaned columns:\n", df.columns.tolist())

# Convert date columns (safe parsing for mixed formats)
date_cols = [
    "po_sent_to_vendor_date",
    "scheduled_delivery_date",
    "delivered_to_client_date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Drop rows only if all critical dates are missing
df = df.dropna(subset=date_cols, how="all")

# Feature engineering
df["delivery_delay_days"] = (
    df["delivered_to_client_date"] - df["scheduled_delivery_date"]
).dt.days

df["fulfillment_time_days"] = (
    df["delivered_to_client_date"] - df["po_sent_to_vendor_date"]
).dt.days

# Late delivery flag
df["late_flag"] = (df["delivery_delay_days"] > 0).astype(int)

# Ensure numeric columns are numeric
num_cols = ["freight_cost_usd", "line_item_value"]
for col in num_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "")
        .replace("nan", None)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing numeric KPIs with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Drop rows with missing vendor
df = df.dropna(subset=["vendor"])

# Aggregate to supplier-level KPIs
supplier_metrics = df.groupby("vendor", as_index=False).agg({
    "delivery_delay_days": "mean",
    "fulfillment_time_days": "mean",
    "late_flag": "mean",
    "freight_cost_usd": "mean",
    "line_item_value": "mean"
})

supplier_metrics.rename(columns={
    "delivery_delay_days": "avg_delivery_delay_days",
    "fulfillment_time_days": "avg_fulfillment_time_days",
    "late_flag": "late_delivery_rate",
    "freight_cost_usd": "avg_freight_cost",
    "line_item_value": "avg_order_value"
}, inplace=True)

# Business-driven risk labeling
def label_risk(row):
    if row["late_delivery_rate"] > 0.5 or row["avg_delivery_delay_days"] > 5:
        return "High"
    elif row["late_delivery_rate"] > 0.25 or row["avg_delivery_delay_days"] > 2:
        return "Medium"
    else:
        return "Low"

supplier_metrics["risk_label"] = supplier_metrics.apply(label_risk, axis=1)

# Save processed features
supplier_metrics.to_csv("data/supplier_features.csv", index=False)

print("\n🎯 Preprocessing complete!")
print("Rows:", len(supplier_metrics))
print(supplier_metrics.head())
