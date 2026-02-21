```markdown
# 📊 Customer Churn Prediction — Machine Learning Project

## 🚚 Supplier Risk Analytics

**Production-Style Machine Learning for Supply Chain Risk Assessment**

A real-world inspired machine learning project that analyzes supplier performance data to predict supplier risk categories (`Low`, `Medium`, `High`) based on delivery delays, fulfillment efficiency, late delivery rates, and cost-related KPIs.

This project demonstrates how raw operational supply chain data can be transformed into actionable supplier risk insights to support procurement and logistics decision-making.

---

## 🎯 Project Objective

- Build a machine learning model to classify suppliers into risk categories using historical delivery and logistics performance data.
- Enable procurement teams to proactively identify high-risk suppliers and take preventive actions before disruptions occur.
- Showcase an end-to-end ML workflow: preprocessing → feature engineering → model training → evaluation → prediction.

---

## 🧠 Business Problem

In enterprise supply chains, supplier performance directly impacts delivery timelines, inventory planning, and customer satisfaction. However, raw supply chain data often contains challenges such as:

- Inconsistent timestamps
- Delayed shipments
- Cost variability
- Operational noise

This project offers a data-driven approach to supplier risk management, empowering teams to shift from reactive firefighting to proactive risk mitigation.

---

## 📊 Dataset

The pipeline uses transactional supply chain data aggregated into supplier-level KPIs.

| Dataset                 | Description                         |
|-------------------------|-----------------------------------|
| `data/supply_chain.csv` | Order-level logistics records      |
| `data/supplier_features.csv` | Supplier-level engineered features |

### Key Features

| Feature               | Description                                  |
|-----------------------|----------------------------------------------|
| `avg_delivery_delay_days`    | Average days of delivery delay               |
| `avg_fulfillment_time_days`  | Average number of days to fulfill an order  |
| `late_delivery_rate`          | Proportion of late deliveries                 |
| `avg_freight_cost`            | Average freight cost per order                |
| `avg_order_value`             | Average monetary value of orders              |
| `risk_label`                 | Risk category label (`Low`, `Medium`, `High`)|

---

## ⚙️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-learn (Random Forest)
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

---

## 🔄 End-to-End Workflow

### 1️⃣ Preprocessing & Feature Engineering

python src/preprocess.py
```

**Sample Output:**

```
🎯 Preprocessing complete!
Rows: 73
```

**Process Details:**

- Cleans column names
- Parses mixed date formats
- Engineers delivery delay and fulfillment KPIs
- Aggregates order-level data to supplier-level metrics
- Assigns business-rule-based risk labels

---

### 2️⃣ Exploratory Data Analysis (EDA)

```bash
python src/eda.py
```

**Sample Insights:**

```
🚦 Supplier Risk Prediction
Avg Delay (days): 3.0
Fulfillment Time (days): 12.0
Late Delivery Rate: 0.4
Avg Freight Cost: 800.0
Avg Order Value: 15000.0

👉 Predicted Risk Category: Medium
```

Visualizes supplier risk distribution and key feature patterns to understand data imbalance and performance trends.

---

### 3️⃣ Model Training & Evaluation

```bash
python src/train_model.py
```

**Sample Output:**

```
📊 Classification Report:
accuracy: 1.00

🔍 Feature Importance (Most important → least important):
- avg_delivery_delay_days
- late_delivery_rate
- avg_order_value
- avg_fulfillment_time_days
- avg_freight_cost
```

**⚠️ Note:**  
The dataset is small and imbalanced (most suppliers are Low risk), which can lead to optimistic evaluation results. In production, more diverse historical data would be needed for robust generalization.

---

## 📈 Key Insights

- Delivery delays and late delivery rate are the strongest drivers of supplier risk.
- Fulfillment time and order value patterns also influence risk classification.
- The model provides interpretable feature importance to support business decisions.

---

## 💼 Business Value

- Proactive identification of medium/high-risk suppliers
- Improved supplier governance and SLA management
- Data-driven procurement planning
- Reduced operational disruptions
- Strong foundation for enterprise supplier risk monitoring systems

---

## 🚀 Future Improvements

- Integrate SHAP for enhanced model explainability
- Deploy as REST API using FastAPI
- Build dashboards with Power BI or Streamlit for procurement teams
- Introduce cross-validation and advanced models (e.g., XGBoost)
- Implement continuous retraining and monitoring pipelines

---

## 🧪 How to Run Locally

```bash
git clone <your-repo-url>
cd Supplier-Risk-Analytics

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing and training pipeline
python src/preprocess.py
python src/train_model.py
python src/predict.py
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request with a clear description of the changes.

Please ensure your code adheres to the existing style and includes tests for new features where applicable.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## ⭐ Why This Project Matters

This project demonstrates:

- Real-world supply chain analytics
- Production-style machine learning pipeline design
- Feature engineering from operational data
- Translating ML outputs into actionable business insights
- Practical AI application in logistics and procurement domains

---
```
