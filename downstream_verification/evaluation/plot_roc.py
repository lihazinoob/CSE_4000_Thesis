from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLOTS_DIR = _PROJECT_ROOT / 'artifacts' / 'plots'
_PLOT_NAME = 'bh_sig_hindi_roc_curve.png'


def plot_and_save_roc_curve(
    roc_df: pd.DataFrame,
    roc_auc: float,
    threshold_info: Dict[str, float],
) -> None:
    """
    Plot the ROC curve with the best operating threshold marked and save to artifacts/plots/.

    Args:
        roc_df:          DataFrame with columns fpr, tpr, threshold.
        roc_auc:         Scalar AUC value.
        threshold_info:  Output of compute_threshold_diagnostics().
    """
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot(
        roc_df['fpr'], roc_df['tpr'],
        linewidth=2,
        label=f'ROC (AUC = {roc_auc:.4f})',
    )
    plt.plot(
        [0, 1], [0, 1],
        linestyle='--', color='gray', linewidth=1.5,
        label='Random Baseline',
    )
    plt.scatter(
        threshold_info['best_threshold_fpr'],
        threshold_info['best_threshold_tpr'],
        color='#d62728', s=70, zorder=3,
        label=f"Best Threshold (d = {threshold_info['best_distance_threshold']:.4f})",
    )
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 − FRR)')
    plt.title('ROC Curve on Unseen Test Writers — BHSig Hindi')
    plt.grid(alpha=0.25)
    plt.legend(loc='lower right')
    plt.tight_layout()

    save_path = _PLOTS_DIR / _PLOT_NAME
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'ROC plot saved → {save_path}')
