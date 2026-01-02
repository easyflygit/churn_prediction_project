import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/raw/customer_churn_dataset-testing-master.csv"
MODEL_PATH = "models/churn_model.pkl"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    categorical_features = ["Gender", "Subscription Type", "Contract Length"]
    numeric_features = [
        "Age", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay",
        "Total Spend", "Last Interaction"
    ]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)


if __name__ == "__main__":
    train()


# data = joblib.load("models/churn_model.pkl")
# print(data)