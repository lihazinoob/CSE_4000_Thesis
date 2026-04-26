from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_roc_curve(
        roc_df: pd.DataFrame,
        roc_auc: float,
        threshold_info: Dict[str, float],
        title: str = "ROC Curve",
        save_path: Path = None,
) -> None:
    fig = plt.figure(figsize=(7, 7))
    plt.plot(
        roc_df['fpr'],
        roc_df['tpr'],
        linewidth=2,
        label=f'ROC (AUC = {roc_auc:.4f})'
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='gray',
        linewidth=1.5,
        label='Random Baseline'
    )
    plt.scatter(
        threshold_info['best_threshold_fpr'],
        threshold_info['best_threshold_tpr'],
        color='#d62728',
        s=70,
        label=f"Best Threshold (d={threshold_info['best_distance_threshold']:.4f})",
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
        score_df: pd.DataFrame,
        distance_threshold: float,
        title: str = "Confusion matrix",
        save_path: Path = None,
) -> None:
    predicted_labels = (score_df['distance'].to_numpy() <= distance_threshold).astype(int)
    true_labels = score_df['label'].to_numpy()
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])

    fig, ax = plt.subplots(figsize=(6, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=['Genuine', 'Forgery']
    )
    display.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)
