import os
import json
import joblib
import numpy as np
import pandas as pd

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
# RANDOM FOREST (INSIGHTS)
# =========================


def train_random_forest(X_train, y_train, preprocessor):
    print("\n=== Random Forest (Feature Importance) ===")

    X_train_processed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train_processed, y_train)

    importances = rf.feature_importances_
    rf_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)

    rf_importance.to_csv(
        f"{ARTIFACTS_DIR}/rf_feature_importance.csv", index=False
    )

    # ---- group importance ----
    group_rows = []

    for group, raw_features in GROUP_MAPPING.items():
        mask = rf_importance["feature"].apply(
            lambda x: any(raw in x for raw in raw_features)
        )
        group_importance = rf_importance.loc[mask, "importance"].sum()
        group_rows.append(
            {"group": group, "importance": group_importance}
        )

    group_df = (
        pd.DataFrame(group_rows)
        .sort_values(by="importance", ascending=False)
    )

    group_df.to_csv(
        f"{ARTIFACTS_DIR}/rf_group_importance.csv", index=False
    )

    print(rf_importance.head(10))
    print("\nSaved RF feature & group importance")

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

    # Random Forest (insights only)
    preprocessor = get_preprocessor()
    train_random_forest(X_train, y_train, preprocessor)


#
if __name__ == "__main__":
    main()