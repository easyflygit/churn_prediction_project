import json
import joblib
import pandas as pd

MODEL_DIR = "/Users/imac/Desktop/python_work/ml_churn_project/churn_project/models"


# -------------------------
# Risk segmentation
# -------------------------
def get_risk_segment(churn_proba: float) -> str:
    if churn_proba >= 0.7:
        return "High Risk"
    elif churn_proba >= 0.4:
        return "Medium Risk"
    elif churn_proba < 0.3:
        return "Low Risk"
    else:
        return "Critical Risk"


# -------------------------
# Prediction
# -------------------------
def predict_single_client(client_data: dict):
    """
    client_data - словарь с признаками клиента
    """

    # загрузка модели и threshold
    pipeline = joblib.load(f"{MODEL_DIR}/logistic_regression.pkl")

    with open(f"{MODEL_DIR}/lr_threshold.json") as f:
        threshold_data = json.load(f)
        threshold = threshold_data["threshold"]

    # преобразуем в DataFrame
    X = pd.DataFrame([client_data])

    # вероятность churn
    churn_proba = pipeline.predict_proba(X)[:, 1][0]

    # класс по threshold
    churn_pred = int(churn_proba >= threshold)

    # risk_segment
    risk_segment = get_risk_segment(churn_proba)

    return {
        "churn_probability": round(float(churn_proba), 3),
        "churn_prediction": churn_pred,
        "risk_segment": risk_segment,
        "decision_threshold": round(float(threshold), 2),
    }


if __name__ == "__main__":
    example_client = {
        "Age": 42,
        "Gender": "Male",
        "Tenure": 12,
        "Usage Frequency": 3,
        "Support Calls": 5,
        "Payment Delay": 20,
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Total Spend": 320,
        "Last Interaction": 14
    }

    result = predict_single_client(example_client)
    print(result)