from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CSV_DIR = _PROJECT_ROOT / 'artifacts' / 'csv'
_PLOTS_DIR = _PROJECT_ROOT / 'artifacts' / 'plots'
_PLOT_NAME = 'bh_sig_hindi_ssl_pretraining_loss_curve.png'


def plot_pretraining_loss_curve() -> None:
    epoch_df = pd.read_csv(_CSV_DIR / 'ssl_pretraining_epoch_summary.csv')
    step_df = pd.read_csv(_CSV_DIR / 'ssl_pretraining_step_summary.csv')

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: per-step loss with intra-epoch running average ---
    ax = axes[0]
    ax.plot(
        step_df['global_step'],
        step_df['batch_recons_loss'],
        linewidth=0.6,
        alpha=0.35,
        color='steelblue',
        label='Batch Loss',
    )
    ax.plot(
        step_df['global_step'],
        step_df['running_avg_recons_loss'],
        linewidth=1.4,
        color='#d62728',
        label='Running Avg Loss',
    )
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Foreground-Weighted MSE Loss')
    ax.set_title('SSL Pretraining — Per-Step Loss (BHSig Hindi)')
    ax.grid(alpha=0.25)
    ax.legend()

    # --- Right: per-epoch average loss with LR on twin axis ---
    ax2 = axes[1]
    color_loss = 'steelblue'
    color_lr = '#2ca02c'

    ln1 = ax2.plot(
        epoch_df['epoch'],
        epoch_df['epoch_avg_recons_loss'],
        marker='o',
        markersize=4,
        linewidth=2,
        color=color_loss,
        label='Avg Reconstruction Loss',
    )
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg Foreground-Weighted MSE Loss', color=color_loss)
    ax2.tick_params(axis='y', labelcolor=color_loss)

    ax3 = ax2.twinx()
    ln2 = ax3.plot(
        epoch_df['epoch'],
        epoch_df['epoch_end_lr'],
        linestyle='--',
        linewidth=1.5,
        color=color_lr,
        label='Learning Rate',
    )
    ax3.set_ylabel('Learning Rate', color=color_lr)
    ax3.tick_params(axis='y', labelcolor=color_lr)

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.set_title('SSL Pretraining — Per-Epoch Loss & LR (BHSig Hindi)')
    ax2.grid(alpha=0.25)

    plt.tight_layout()

    save_path = _PLOTS_DIR / _PLOT_NAME
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Plot saved → {save_path}')


if __name__ == '__main__':
    plot_pretraining_loss_curve()
