import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# признаки
CATEGORICAL_FEATURES = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]


def load_data(path="data/raw/customer_churn_dataset-testing-master.csv"):
    """
    Загружает датасет
    """
    df = pd.read_csv(path)
    y = df["Churn"]
    X = df.drop(columns=["Churn", "CustomerID"])
    return X, y


def get_preprocessor():
    """
    Возвращает ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
             CATEGORICAL_FEATURES)
        ]
    )