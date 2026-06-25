from typing import Dict, List

import torch
import torch.nn.functional as F


def build_writer_prototype(reference_embeddings: List[torch.Tensor]) -> torch.Tensor:
    """
    Average a list of reference embeddings and L2-normalize the result.

    Args:
        reference_embeddings: Embeddings of the reference genuine signatures (CPU tensors).

    Returns:
        L2-normalized prototype tensor of shape [embedding_dim].
    """
    if not reference_embeddings:
        raise ValueError('reference_embeddings is empty.')
    stacked = torch.stack(reference_embeddings, dim=0)      # [N, embedding_dim]
    mean_embedding = stacked.mean(dim=0)                    # [embedding_dim]
    return F.normalize(mean_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)


def build_all_writer_prototypes(
    writer_to_reference_paths: Dict[int, List[str]],
    path_to_embedding: Dict[str, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    Build one prototype per writer from their reference genuine paths.

    Args:
        writer_to_reference_paths: Maps writer_id → list of reference genuine image paths.
        path_to_embedding:         Pre-computed path → embedding dict.

    Returns:
        Dict mapping writer_id → L2-normalized prototype tensor.
    """
    return {
        writer_id: build_writer_prototype([path_to_embedding[p] for p in paths])
        for writer_id, paths in writer_to_reference_paths.items()
    }
