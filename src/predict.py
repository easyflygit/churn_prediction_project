import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"


def predict(input_data: pd.DataFrame):
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(input_data)[:, 1]
    return proba


if __name__ == "__main__":
    sample = pd.DataFrame([{
        "Age": 35,
        "Gender": "Male",
        "Tenure": 20,
        "Usage Frequency": 1,
        "Support Calls": 20,
        "Payment Delay": 20,
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Total Spend": 100,
        "Last Interaction": 10
    }])

    print("Churn probability:", predict(sample))