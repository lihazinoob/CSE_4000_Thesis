import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_fg_bg_loss(
    csv_entries: list[tuple[str, str]],
    output_dir: str,
    prefix: str = "mendeley",
) -> None:
    """
    csv_entries: list of (csv_path, label) e.g. [("path/w1.csv", "FG Weight = 1"), ...]
    """
    dfs = [(pd.read_csv(path), label) for path, label in csv_entries]

    _plot_loss(
        dfs,
        col="epoch_avg_fg_mse",
        ylabel="Avg FG MSE Loss",
        title="Training Foreground MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_training_fg_loss.png",
    )
    _plot_loss(
        dfs,
        col="epoch_avg_bg_mse",
        ylabel="Avg BG MSE Loss",
        title="Training Background MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_training_bg_loss.png",
    )


def plot_eval_fg_bg_loss(
    csv_entries: list[tuple[str, str]],
    output_dir: str,
    prefix: str = "mendeley",
) -> None:
    """
    csv_entries: list of (csv_path, label) e.g. [("path/w1.csv", "FG Weight = 1"), ...]
    """
    dfs = [(pd.read_csv(path), label) for path, label in csv_entries]

    _plot_loss(
        dfs,
        col="eval_mean_fg_mse",
        ylabel="Eval Mean FG MSE Loss",
        title="Evaluation Foreground MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_eval_fg_loss.png",
    )
    _plot_loss(
        dfs,
        col="eval_mean_bg_mse",
        ylabel="Eval Mean BG MSE Loss",
        title="Evaluation Background MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_eval_bg_loss.png",
    )


def _plot_loss(
    dfs: list[tuple[pd.DataFrame, str]],
    col: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    style = dict(linewidth=1.8, marker="o", markersize=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (df, label), color in zip(dfs, _COLORS):
        ax.plot(df["epoch"], df[col], label=label, color=color, **style)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\n(Mendeley Dataset — SSL Pretraining)", fontsize=13)
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts_new", "csv")

    csv_entries = [
        (os.path.join(base, "mendeley_ssl_pretraining_fg_weight_1_epoch_summary.csv"),  "FG Weight = 1"),
        (os.path.join(base, "mendeley_ssl_pretraining_fg_weight_5_epoch_summary.csv"),  "FG Weight = 5"),
        (os.path.join(base, "mendeley_ssl_pretraining_fg_weight_10_epoch_summary.csv"), "FG Weight = 10"),
        (os.path.join(base, "mendeley_ssl_pretraining_fg_weight_15_epoch_summary.csv"), "FG Weight = 15"),
        (os.path.join(base, "mendeley_ssl_pretraining_fg_weight_20_epoch_summary.csv"), "FG Weight = 20"),
    ]

    plots = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts_new", "plots")
    plot_fg_bg_loss(csv_entries=csv_entries, output_dir=plots)
    plot_eval_fg_bg_loss(csv_entries=csv_entries, output_dir=plots)
