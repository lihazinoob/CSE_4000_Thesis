from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLOTS_DIR = _PROJECT_ROOT / 'artifacts' / 'plots'
_PLOT_NAME = 'bh_sig_hindi_train_validation_plot.png'


def plot_training_history(history_df: pd.DataFrame) -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(history_df['epoch'], history_df['train_loss'], marker='o', linewidth=2, label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], marker='o', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dual-Triplet Loss')
    plt.title('Train vs Fixed-Validation Loss — BHSig Hindi')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_path = _PLOTS_DIR / _PLOT_NAME
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Plot saved → {save_path}')
