import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize_map(feature_2d: torch.Tensor) -> np.ndarray:
    feature_2d = feature_2d.detach().cpu()
    feature_2d = feature_2d - feature_2d.min()
    feature_2d = feature_2d / (feature_2d.max() + 1e-8)
    return feature_2d.numpy()


def summarize_feature_map(feature_map: torch.Tensor, mode: str = "abs_mean") -> np.ndarray:
    if mode == "mean":
        summary = feature_map.mean(dim=1)
    elif mode == "abs_mean":
        summary = feature_map.abs().mean(dim=1)
    elif mode == "max":
        summary = feature_map.max(dim=1).values
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return normalize_map(summary.squeeze(0))


def plot_feature_grid(
        feature_map: torch.Tensor,
        max_channels: int = 32,
        cols: int = 8,
        cmap: str = "inferno",
        figsize: Tuple[int, int] = (16, 44),
        title: str = "Feature Maps",
        save_path: Path = None,
):
    fmap = feature_map.squeeze(0)
    num_channels = min(fmap.shape[0], max_channels)
    rows = math.ceil(num_channels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    image = None
    for i in range(num_channels):
        channel_map = normalize_map(fmap[i])
        image = axes[i].imshow(channel_map, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[i].set_title(f"Ch {i}", fontsize=9)
        axes[i].axis("off")

    for i in range(num_channels, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.tolist(),
            fraction=0.02,
            pad=0.02,
            label="Normalized Activation"
        )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_summary_and_overlay(
        input_tensor: torch.Tensor,
        feature_map: torch.Tensor,
        summary_mode: str = "abs_mean",
        heatmap_cmap: str = "inferno",
        alpha: float = 0.4,
        figsize: Tuple[int, int] = (12, 4),
        title_prefix: str = "Feature Map",
        save_path: Path = None,
):
    input_image = input_tensor.squeeze().detach().cpu().numpy()
    summary_map = summarize_feature_map(feature_map, mode=summary_mode)

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    summary_image = plt.imshow(summary_map, cmap=heatmap_cmap, vmin=0.0, vmax=1.0)
    plt.title(f"{title_prefix} Summary")
    plt.axis("off")
    plt.colorbar(summary_image, fraction=0.046, pad=0.04, label="Normalized Activation")



    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
