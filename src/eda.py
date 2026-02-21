# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/supplier_features.csv")

print(df.head())
print(df.describe())

sns.countplot(x="risk_label", data=df)
plt.title("Supplier Risk Distribution")
plt.show()
