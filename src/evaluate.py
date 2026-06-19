import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from config import FIGURES_PATH

def regression_report(y_true, y_pred, label='Model'):
    """Print and return all regression metrics."""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'R²':   r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    print(f"\n{'='*40}")
    print(f"  {label} — Regression Metrics")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<8} {v:.4f}")
    return metrics


def plot_residuals(y_true, y_pred, label='Model', save=False):
    """
    Two essential regression diagnostic plots:
    1. Predicted vs Actual (should be on the diagonal)
    2. Residuals vs Predicted (should be random scatter around 0)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, y_true, alpha=0.4, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set(xlabel='Predicted', ylabel='Actual',
                title=f'{label} — Predicted vs Actual')

    axes[1].scatter(y_pred, residuals, alpha=0.4, edgecolors='k', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set(xlabel='Predicted', ylabel='Residual',
                title=f'{label} — Residuals')

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_PATH / f'{label}_residuals.png',
                    dpi=150, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results_df, metric='mean_score', save=False):
    """
    Visualize CV results from compare_models().
    results_df must have columns: model, mean_score, std_score
    """
    results_df = results_df.sort_values(metric)
    plt.figure(figsize=(8, 5))
    plt.barh(results_df['model'], results_df[metric],
             xerr=results_df['std_score'],
             color='steelblue', alpha=0.8, capsize=4)
    plt.xlabel(metric)
    plt.title('Model Comparison — Cross-Validation Scores')
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_PATH / 'model_comparison.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
