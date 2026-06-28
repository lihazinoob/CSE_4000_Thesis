from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from downstream_verification.evaluation.encode_signatures import encode_all_signatures
from downstream_verification.evaluation.metrics import (
    compute_per_writer_metrics,
    compute_roc_metrics,
    compute_threshold_diagnostics,
)
from downstream_verification.evaluation.score import collect_prototype_scores

VAL_REFERENCE_SEEDS: List[int] = [101, 202, 303, 404, 505]
_NUM_REFERENCE_GENUINE: int = 5


def run_prototype_validation(
    model: nn.Module,
    val_inventory_df: pd.DataFrame,
    target_size: Tuple[int, int],
    device: torch.device,
    num_reference_genuine: int = _NUM_REFERENCE_GENUINE,
    seeds: List[int] = VAL_REFERENCE_SEEDS,
) -> Dict[str, float]:
    """
    Prototype-based validation that mirrors the test-time evaluation protocol exactly.

    Encodes all validation signatures once (shared across seeds), then for each seed
    runs the full prototype-scoring pipeline and accumulates global pooled AUC, EER,
    and per-writer AUC. Returns the 5-seed averages used for checkpoint selection and
    diagnostics. Seeds are distinct from the triplet-val seed (314) and the test-time
    seed (2026).
    """
    all_val_paths = val_inventory_df['image_path'].astype(str).tolist()
    path_to_embedding = encode_all_signatures(
        model=model,
        image_paths=all_val_paths,
        target_size=target_size,
        device=device,
    )

    per_seed_global_aucs: List[float] = []
    per_seed_eers: List[float] = []
    per_seed_per_writer_aucs: List[float] = []

    for seed in seeds:
        score_df, _ = collect_prototype_scores(
            test_inventory_df=val_inventory_df,
            path_to_embedding=path_to_embedding,
            num_reference_genuine=num_reference_genuine,
            seed=seed,
        )

        roc_df, global_auc = compute_roc_metrics(score_df)
        per_seed_global_aucs.append(global_auc)

        threshold_info = compute_threshold_diagnostics(roc_df)
        per_seed_eers.append(threshold_info['eer_estimate'])

        per_writer_df = compute_per_writer_metrics(score_df)
        per_seed_per_writer_aucs.append(float(per_writer_df['auc'].mean()))

    return {
        'val_proto_auc_mean': float(np.mean(per_seed_global_aucs)),
        'val_proto_auc_std': float(np.std(per_seed_global_aucs)),
        'val_proto_per_writer_auc_mean': float(np.mean(per_seed_per_writer_aucs)),
        'val_proto_eer_mean': float(np.mean(per_seed_eers)),
    }
