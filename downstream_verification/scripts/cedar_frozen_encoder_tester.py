"""
Ablation sweep: prototype-based verification on CEDAR using the Mendeley-pretrained
frozen encoder ONLY — no projector head, no downstream fine-tuning on any data.

The encoder's 256-dim pooled output is L2-normalized directly and used as the
signature embedding. This measures what the SSL representation alone can achieve
before any metric-learning adaptation.

This script sweeps ALL 10 pretrained checkpoints (foreground weights {1,5,10,15,20}
× epochs {10,50}) on the CEDAR *test* writers and reports one row per checkpoint.
This is an EXPLORATORY comparison on the test set — no validation-based model
selection is performed and no single "best model" claim is made here.

Each checkpoint is scored over K fixed reference-sampling seeds (the same seeds for
every checkpoint, so the comparison is paired) and we report mean ± std of ROC-AUC /
EER. Setting K=1 reproduces the original single-draw behavior.

Known limitation (intentionally NOT corrected here): the encoder uses BatchNorm whose
running statistics, accumulated during SSL, are dominated by 16×16-patch activations
(256 patches vs 1 full image per step). At probe time the encoder runs in eval() on
full images using those patch-derived stats. This is a documented caveat for the
write-up, not a bug in this script.

Outputs (artifacts_new/csv/):
  mendeley_frozenEncoderOnly_cedar_sweep.csv
      one row per checkpoint: fg_weight, epoch, roc_auc_mean/std, eer_mean/std, ...
  mendeley_frozenEncoderOnly_cedar_per_writer__epoch{E}_weight{W}.csv
      per-writer breakdown for each checkpoint (seed[0] draw), one file per checkpoint.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from downstream_verification.datasets.create_cedar_inventory import build_cedar_inventory
from downstream_verification.evaluation.encode_signatures import encode_all_signatures
from downstream_verification.evaluation.metrics import (
    compute_per_writer_metrics,
    compute_roc_metrics,
    compute_threshold_diagnostics,
)
from downstream_verification.evaluation.score import collect_prototype_scores
from ssl_pretraining.models.Encoder import Encoder

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_JSON_PATH = _PROJECT_ROOT / 'artifacts_new' / 'json' / 'cedar_downstream_mendeley_ssl.json'
_CEDAR_DATA_DIR = _PROJECT_ROOT / 'data' / 'all' / 'CEDAR'
_ENCODER_DIR = _PROJECT_ROOT / 'artifacts_new' / 'models' / 'encoder'
_CSV_DIR = _PROJECT_ROOT / 'artifacts_new' / 'csv'

TARGET_IMAGE_SIZE: Tuple[int, int] = (256, 256)
NUM_REFERENCE_GENUINE: int = 5
# Same seeds applied to every checkpoint -> paired comparison. K=1 -> original behavior.
PROTOTYPE_EVAL_SEEDS: Tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
_NORM_TYPE: str = 'batch'

# Sweep grid.
_EPOCHS: Tuple[int, ...] = (10, 50)
_FG_WEIGHTS: Tuple[int, ...] = (1, 5, 10, 15, 20)


class _FrozenEncoderModel(nn.Module):
    """
    Wraps the pretrained encoder for inference-only use.

    Forward pass: image [B,1,256,256] → encoder(pool=True) → [B,256]
                  → L2-normalize → [B,256] unit-norm embedding.

    No projector head. This is the raw SSL representation.
    """

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.encoder = Encoder(norm_type=_NORM_TYPE)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=True)
        print(f'Loaded encoder from: {checkpoint_path}')
        print(f'  Missing keys:     {len(missing)}')
        print(f'  Unexpected keys:  {len(unexpected)}')
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, pool=True)          # [B, 256]
        return F.normalize(features, p=2, dim=1)        # [B, 256], unit norm


def _checkpoint_grid() -> List[Tuple[int, int, Path]]:
    """Return (epoch, fg_weight, checkpoint_path) for every checkpoint in the sweep."""
    grid: List[Tuple[int, int, Path]] = []
    for epoch in _EPOCHS:
        for weight in _FG_WEIGHTS:
            path = _ENCODER_DIR / f'mendeley_epoch{epoch}_weight_{weight}.pt'
            if not path.exists():
                raise FileNotFoundError(f'Expected checkpoint not found: {path}')
            grid.append((epoch, weight, path))
    return grid


def _evaluate_checkpoint(
    epoch: int,
    weight: int,
    checkpoint_path: Path,
    test_inventory_df: pd.DataFrame,
    device: torch.device,
) -> Tuple[dict, pd.DataFrame]:
    """
    Encode all test signatures with this checkpoint, then score over every seed.

    Returns:
        summary_row:        dict of mean/std metrics across PROTOTYPE_EVAL_SEEDS.
        per_writer_df:      per-writer breakdown for the FIRST seed (for inspection).
    """
    model = _FrozenEncoderModel(checkpoint_path=str(checkpoint_path)).to(device)

    all_paths = test_inventory_df['image_path'].astype(str).tolist()
    path_to_embedding = encode_all_signatures(
        model=model,
        image_paths=all_paths,
        target_size=TARGET_IMAGE_SIZE,
        device=device,
    )

    roc_aucs: List[float] = []
    eers: List[float] = []
    per_writer_auc_means: List[float] = []
    first_seed_per_writer_df: pd.DataFrame = pd.DataFrame()

    for seed_index, seed in enumerate(PROTOTYPE_EVAL_SEEDS):
        score_df, _ = collect_prototype_scores(
            test_inventory_df=test_inventory_df,
            path_to_embedding=path_to_embedding,
            num_reference_genuine=NUM_REFERENCE_GENUINE,
            seed=seed,
        )
        roc_full_df, roc_auc = compute_roc_metrics(score_df)
        threshold_info = compute_threshold_diagnostics(roc_full_df)
        per_writer_df = compute_per_writer_metrics(score_df)

        roc_aucs.append(roc_auc)
        eers.append(threshold_info['eer_estimate'])
        per_writer_auc_means.append(float(per_writer_df['auc'].mean()))

        if seed_index == 0:
            first_seed_per_writer_df = per_writer_df

    auc_series = pd.Series(roc_aucs)
    eer_series = pd.Series(eers)
    pw_series = pd.Series(per_writer_auc_means)

    summary_row = {
        'encoder_checkpoint': checkpoint_path.name,
        'epoch': epoch,
        'fg_weight': weight,
        'num_seeds': len(PROTOTYPE_EVAL_SEEDS),
        'roc_auc_mean': round(float(auc_series.mean()), 6),
        'roc_auc_std': round(float(auc_series.std(ddof=0)), 6),
        'eer_mean': round(float(eer_series.mean()), 6),
        'eer_std': round(float(eer_series.std(ddof=0)), 6),
        'per_writer_auc_mean': round(float(pw_series.mean()), 6),
        'per_writer_auc_std': round(float(pw_series.std(ddof=0)), 6),
    }

    print(
        f'  -> epoch {epoch:>2} | fg_weight {weight:>2} | '
        f'ROC-AUC {summary_row["roc_auc_mean"]:.4f} ± {summary_row["roc_auc_std"]:.4f} | '
        f'EER {summary_row["eer_mean"]:.4f} ± {summary_row["eer_std"]:.4f}'
    )
    return summary_row, first_seed_per_writer_df


def run() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load writer split ─────────────────────────────────────────────────────
    with open(_SPLIT_JSON_PATH, 'r') as f:
        splits = json.load(f)

    test_writer_ids = splits['downstream_verification_testing']
    print(f'Test writers: {len(test_writer_ids)} -> {sorted(test_writer_ids, key=int)}')

    # ── Build CEDAR test inventory (shared across all checkpoints) ─────────────
    test_inventory_df = build_cedar_inventory(test_writer_ids, _CEDAR_DATA_DIR)
    n_genuine = (test_inventory_df['signature_type'] == 'G').sum()
    n_forgery = (test_inventory_df['signature_type'] == 'F').sum()
    print(
        f'Test inventory: {len(test_inventory_df)} images | '
        f'G={n_genuine} | F={n_forgery}'
    )
    print(f'Reference genuines per writer: {NUM_REFERENCE_GENUINE}')
    print(f'Prototype eval seeds (paired across checkpoints): {PROTOTYPE_EVAL_SEEDS}')

    _CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ── Sweep every checkpoint ────────────────────────────────────────────────
    grid = _checkpoint_grid()
    print(f'\nSweeping {len(grid)} checkpoints...\n' + '=' * 60)

    summary_rows: List[dict] = []
    for epoch, weight, checkpoint_path in grid:
        summary_row, per_writer_df = _evaluate_checkpoint(
            epoch=epoch,
            weight=weight,
            checkpoint_path=checkpoint_path,
            test_inventory_df=test_inventory_df,
            device=device,
        )
        summary_rows.append(summary_row)

        per_writer_path = (
            _CSV_DIR
            / f'mendeley_frozenEncoderOnly_cedar_per_writer__epoch{epoch}_weight{weight}.csv'
        )
        per_writer_df.to_csv(per_writer_path, index=False)

    # ── Combined comparison table ─────────────────────────────────────────────
    sweep_df = pd.DataFrame(summary_rows).sort_values(
        by=['epoch', 'fg_weight']
    ).reset_index(drop=True)

    sweep_path = _CSV_DIR / 'mendeley_frozenEncoderOnly_cedar_sweep.csv'
    sweep_df.to_csv(sweep_path, index=False)

    print('\n' + '=' * 60)
    print('CEDAR FROZEN ENCODER SWEEP — SUMMARY (exploratory, test set)')
    print('=' * 60)
    print(sweep_df.to_string(index=False))

    best_row = sweep_df.loc[sweep_df['roc_auc_mean'].idxmax()]
    print(
        f'\nHighest ROC-AUC (exploratory, NOT a selection claim): '
        f'{best_row["encoder_checkpoint"]} -> {best_row["roc_auc_mean"]:.4f}'
    )
    print(f'\nSweep table saved -> {sweep_path}')


if __name__ == '__main__':
    run()
