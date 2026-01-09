import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,\
    roc_curve, precision_score, recall_score


def evaluate_classification(
        y_true,
        y_pred,
        y_proba,
        threshold=0.5
):
    """
    Печатает основные метрики классификации
    """
    print(f"\nThreshold: {threshold}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")

    return roc_auc


def plot_roc_curve(y_true, y_proba):
    """
    Рисует ROC-кривую
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_best_threshold(y_true, y_proba, thresholds=None):
    """
    Подбор threshold по precision / recall
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall
        })

    return results


def print_metrics(y_true, y_pred, title):
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
