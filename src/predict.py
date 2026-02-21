# src/predict.py
import joblib
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Predict supplier risk")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--fulfillment", type=float, default=12.0)
    parser.add_argument("--late_rate", type=float, default=0.4)
    parser.add_argument("--freight_cost", type=float, default=800.0)
    parser.add_argument("--order_value", type=float, default=15000.0)

    args = parser.parse_args()

    model, le = joblib.load("models/supplier_risk_model.pkl")

    X_new = pd.DataFrame([{
        "avg_delivery_delay_days": args.delay,
        "avg_fulfillment_time_days": args.fulfillment,
        "late_delivery_rate": args.late_rate,
        "avg_freight_cost": args.freight_cost,
        "avg_order_value": args.order_value
    }])

    pred = model.predict(X_new)
    risk = le.inverse_transform(pred)[0]

    print("\n🚦 Supplier Risk Prediction")
    print(f"Avg Delay (days): {args.delay}")
    print(f"Fulfillment Time (days): {args.fulfillment}")
    print(f"Late Delivery Rate: {args.late_rate}")
    print(f"Avg Freight Cost: {args.freight_cost}")
    print(f"Avg Order Value: {args.order_value}")
    print(f"\n👉 Predicted Risk Category: {risk}")

if __name__ == "__main__":
    main()
