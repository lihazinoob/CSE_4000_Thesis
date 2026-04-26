from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def build_prototype_matrix(
        writer_to_prototype: Dict[int, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    writer_ids = np.array(sorted(writer_to_prototype.keys()), dtype=np.int32)
    prototype_matrix = []

    for writer_id in writer_ids:
        prototype_tensor = writer_to_prototype[int(writer_id)]
        prototype_matrix.append(prototype_tensor.detach().cpu().numpy())

    return writer_ids, np.stack(prototype_matrix, axis=0)


def compute_pca_projection(
        feature_matrix: np.ndarray,
        n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2-dimensional.")
    if feature_matrix.shape[0] < n_components:
        raise ValueError("Number of samples must be at least n_components.")

    pca = PCA(n_components=n_components)
    projected_coordinates = pca.fit_transform(feature_matrix)
    explained_variance_ratio = pca.explained_variance_ratio_
    return projected_coordinates, explained_variance_ratio


def plot_writer_prototypes_pca(
        writer_to_prototype: Dict[int, torch.Tensor],
        title: str = "PCA of Writer Prototypes",
        figsize=(10, 8),
        annotate_points: bool = True,
        save_path: Path = None,
):
    if not writer_to_prototype:
        raise ValueError("writer_to_prototype is empty.")

    writer_ids, prototype_matrix = build_prototype_matrix(writer_to_prototype)
    projected_coordinates, explained_variance_ratio = compute_pca_projection(
        prototype_matrix,
        n_components=2
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        projected_coordinates[:, 0],
        projected_coordinates[:, 1],
        s=80,
        c="darkred",
        edgecolors="black",
        alpha=0.85
    )

    if annotate_points:
        for index, writer_id in enumerate(writer_ids):
            ax.annotate(
                str(int(writer_id)),
                (projected_coordinates[index, 0], projected_coordinates[index, 1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9
            )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.2f}% variance)")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return writer_ids, projected_coordinates, explained_variance_ratio
