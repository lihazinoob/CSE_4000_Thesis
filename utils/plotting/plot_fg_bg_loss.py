import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_fg_bg_loss(
    csv_weight_1: str,
    csv_weight_5: str,
    output_dir: str,
    prefix: str = "mendeley",
) -> None:
    df1 = pd.read_csv(csv_weight_1)
    df5 = pd.read_csv(csv_weight_5)

    _plot_loss(
        df1, df5,
        col="epoch_avg_fg_mse",
        ylabel="Avg FG MSE Loss",
        title="Training Foreground MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_training_fg_loss.png",
    )
    _plot_loss(
        df1, df5,
        col="epoch_avg_bg_mse",
        ylabel="Avg BG MSE Loss",
        title="Training Background MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_training_bg_loss.png",
    )


def _plot_loss(
    df1: pd.DataFrame,
    df5: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    style = dict(linewidth=1.8, marker="o", markersize=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df1["epoch"], df1[col], label="FG Weight = 1", color="#1f77b4", **style)
    ax.plot(df5["epoch"], df5[col], label="FG Weight = 5", color="#ff7f0e", **style)
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


def plot_eval_fg_bg_loss(
    csv_weight_1: str,
    csv_weight_5: str,
    output_dir: str,
    prefix: str = "mendeley",
) -> None:
    df1 = pd.read_csv(csv_weight_1)
    df5 = pd.read_csv(csv_weight_5)

    _plot_loss(
        df1, df5,
        col="eval_mean_fg_mse",
        ylabel="Eval Mean FG MSE Loss",
        title="Evaluation Foreground MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_eval_fg_loss.png",
    )
    _plot_loss(
        df1, df5,
        col="eval_mean_bg_mse",
        ylabel="Eval Mean BG MSE Loss",
        title="Evaluation Background MSE Loss per Epoch",
        save_path=f"{output_dir}/{prefix}_eval_bg_loss.png",
    )


if __name__ == "__main__":
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts_new")
    csv_w1 = os.path.join(base, "csv", "mendeley_ssl_pretraining_fg_weight_1_epoch_summary.csv")
    csv_w5 = os.path.join(base, "csv", "mendeley_ssl_pretraining_fg_weight_5_epoch_summary.csv")
    plots  = os.path.join(base, "plots")

    plot_fg_bg_loss(csv_weight_1=csv_w1, csv_weight_5=csv_w5, output_dir=plots)
    plot_eval_fg_bg_loss(csv_weight_1=csv_w1, csv_weight_5=csv_w5, output_dir=plots)
