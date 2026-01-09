import os
import json
import joblib
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocess import get_preprocessor, load_data
from evaluate import find_best_threshold, print_metrics, evaluate_classification, plot_roc_curve

MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

GROUP_MAPPING = {
    "Payment Behavior": ["Payment Delay", "Total Spend"],
    "Customer Support": ["Support Calls"],
    "Engagement": ["Usage Frequency", "Last Interaction", "Tenure"],
    "Demographics": ["Age", "Gender"],
    "Subscription": ["Subscription Type", "Contract Length"],
}


# =========================
# LOGISTIC REGRESSION
# =========================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n=== Logistic Regression ===")

    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix by default:\n", confusion_matrix(y_test, y_pred))

    thresholds = find_best_threshold(y_test, y_proba)
    best_threshold = max(
        thresholds,
        key=lambda x: x["recall"] - (1 - x["precision"])
    )

    y_pred_best = (y_proba >= best_threshold["threshold"]).astype(int)
    y_pred_045 = (y_proba >= 0.45).astype(int)

    print(f"Confusion Matrix with best threshold -> "
          f"{best_threshold['threshold']:.3f}:\n", confusion_matrix(y_test, y_pred_best))
    print(f"Confusion Matrix with threshold -> 0.45 "
          f"{best_threshold['threshold']:.3f}:\n", confusion_matrix(y_test, y_pred_045))

    print(f"Best threshold (LR): {best_threshold['threshold']:.3f}")

    # оценка модели через evaluate.py
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    joblib.dump(pipeline, f"{MODELS_DIR}/logistic_regression.pkl")
    with open(f"{MODELS_DIR}/lr_threshold.json", "w") as f:
        json.dump(best_threshold, f, indent=2)

    return pipeline, y_proba

# =========================
# MAIN
# =========================


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Logistic regression (baseline)
    train_logistic_regression(
        X_train, X_test, y_train, y_test
    )


#
if __name__ == "__main__":
    main()