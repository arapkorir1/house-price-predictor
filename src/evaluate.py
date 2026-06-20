
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)
from config import FIGURES_PATH
import logging

logger = logging.getLogger(__name__)


def regression_report(y_true, y_pred, label='Model'):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'R²':   r2_score(y_true, y_pred),
    }
    print(f"\n{'='*40}\n  {label} — Regression Metrics\n{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<8} {v:.4f}")
    return metrics


def classification_full_report(y_true, y_pred, y_proba=None, label='Model'):
    print(f"\n{'='*40}\n  {label} — Classification Metrics\n{'='*40}")
    print(classification_report(y_true, y_pred))
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"  ROC-AUC: {auc:.4f}")
        return auc


def plot_confusion_matrix(y_true, y_pred, labels=None, label='Model', save=False):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'{label} — Confusion Matrix')
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_PATH / f'{label}_confusion_matrix.png',
                     dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, label='Model', save=False):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()
    plt.figure(figsize=(8, max(4, top_n // 2)))
    importances.plot(kind='barh')
    plt.title(f'{label} — Top {top_n} Feature Importances')
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_PATH / f'{label}_feature_importance.png',
                     dpi=150, bbox_inches='tight')
    plt.show()
