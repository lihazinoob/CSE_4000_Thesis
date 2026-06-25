import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from downstream_verification.evaluation.prototype import build_writer_prototype


def collect_prototype_scores(
    test_inventory_df: pd.DataFrame,
    path_to_embedding: Dict[str, torch.Tensor],
    num_reference_genuine: int = 5,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each test writer:
        - Sample num_reference_genuine reference genuines → build prototype.
        - Score all remaining genuines and all forgeries against the prototype.

    Args:
        test_inventory_df:    Inventory DataFrame for test writers only.
        path_to_embedding:    Output of encode_all_signatures().
        num_reference_genuine: Number of genuine signatures used to form the prototype.
        seed:                  RNG seed for reproducible reference sampling.

    Returns:
        score_df:    One row per query. Columns:
                         writer_id, query_path, query_type ('genuine'|'forgery'),
                         label (1|0), distance, score (-distance).
        protocol_df: One row per writer documenting the reference/query split.
    """
    rng = random.Random(seed)

    writer_groups: Dict[int, pd.DataFrame] = {
        int(writer_id): group.reset_index(drop=True)
        for writer_id, group in test_inventory_df.groupby('writer_id')
    }

    score_rows: List[dict] = []
    protocol_rows: List[dict] = []

    for writer_id in sorted(writer_groups.keys()):
        writer_df = writer_groups[writer_id]

        genuine_paths = sorted(
            writer_df.loc[writer_df['signature_type'] == 'G', 'image_path'].astype(str).tolist()
        )
        forgery_paths = sorted(
            writer_df.loc[writer_df['signature_type'] == 'F', 'image_path'].astype(str).tolist()
        )

        if len(genuine_paths) <= num_reference_genuine:
            raise ValueError(
                f'Writer {writer_id} has only {len(genuine_paths)} genuine signatures; '
                f'need more than {num_reference_genuine} for reference + query split.'
            )

        reference_paths = sorted(rng.sample(genuine_paths, num_reference_genuine))
        reference_set = set(reference_paths)
        genuine_query_paths = [p for p in genuine_paths if p not in reference_set]

        prototype = build_writer_prototype(
            [path_to_embedding[p] for p in reference_paths]
        )

        protocol_rows.append({
            'writer_id': writer_id,
            'num_reference_genuine': len(reference_paths),
            'num_genuine_queries': len(genuine_query_paths),
            'num_forgery_queries': len(forgery_paths),
        })

        for query_path in genuine_query_paths:
            distance = F.pairwise_distance(
                path_to_embedding[query_path].unsqueeze(0),
                prototype.unsqueeze(0),
            ).item()
            score_rows.append({
                'writer_id': writer_id,
                'query_path': query_path,
                'query_type': 'genuine',
                'label': 1,
                'distance': distance,
                'score': -distance,
            })

        for query_path in forgery_paths:
            distance = F.pairwise_distance(
                path_to_embedding[query_path].unsqueeze(0),
                prototype.unsqueeze(0),
            ).item()
            score_rows.append({
                'writer_id': writer_id,
                'query_path': query_path,
                'query_type': 'forgery',
                'label': 0,
                'distance': distance,
                'score': -distance,
            })

    return pd.DataFrame(score_rows), pd.DataFrame(protocol_rows)
