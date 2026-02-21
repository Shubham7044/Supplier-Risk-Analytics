# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

# Load features
df = pd.read_csv("data/supplier_features.csv")

print("Class distribution before fix:")
print(df["risk_label"].value_counts())

# Merge ultra-rare classes
vc = df["risk_label"].value_counts()
rare_classes = vc[vc < 2].index.tolist()

if rare_classes:
    print("⚠️ Rare classes found:", rare_classes)
    df["risk_label"] = df["risk_label"].replace(rare_classes, "Medium")
    print("Class distribution after fix:")
    print(df["risk_label"].value_counts())

X = df[
    [
        "avg_delivery_delay_days",
        "avg_fulfillment_time_days",
        "late_delivery_rate",
        "avg_freight_cost",
        "avg_order_value"
    ]
]

y = df["risk_label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model with class imbalance handling
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance
fi = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n🔍 Feature Importance (Most important → least important):")
print(fi)

# Save model
Path("models").mkdir(exist_ok=True)
joblib.dump((model, le), "models/supplier_risk_model.pkl")
print("\n✅ Model saved to models/supplier_risk_model.pkl")
