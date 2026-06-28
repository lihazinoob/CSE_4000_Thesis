import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from downstream_verification.datasets.create_inventory import build_downstream_inventory
from downstream_verification.evaluation.encode_signatures import encode_all_signatures
from downstream_verification.evaluation.metrics import (
    compute_per_writer_metrics,
    compute_roc_metrics,
    compute_threshold_diagnostics,
)
from downstream_verification.evaluation.plot_roc import plot_and_save_roc_curve
from downstream_verification.evaluation.score import collect_prototype_scores
from downstream_verification.models.Embedding_Model import DownstreamSignatureEmbeddingModel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_JSON_PATH = _PROJECT_ROOT / 'artifacts' / 'json' / 'bh_sig_hindi_writer_split.json'
_DATA_DIR = _PROJECT_ROOT / 'data' / 'all' / 'BHSig260_Hindi'
_ENCODER_CHECKPOINT_PATH = (
    _PROJECT_ROOT / 'artifacts' / 'saved_models' / 'encoder_model' / 'encoder_epoch50.pt'
)
_BEST_MODEL_PATH = (
    _PROJECT_ROOT / 'artifacts' / 'saved_models' / 'downstream_verification'
    / 'best_model_bh_sig_hindi_verification.pt'
)
_CSV_DIR = _PROJECT_ROOT / 'artifacts' / 'csv'

# ── Must match trainer.py exactly ─────────────────────────────────────────────
TARGET_IMAGE_SIZE: Tuple[int, int] = (256, 256)
TRAINABLE_ENCODER_STAGES: Tuple[str, ...] = ()
PROJECTOR_HIDDEN_DIM: int = 256
EMBEDDING_DIM: int = 256
NORM_TYPE: str = 'batch'

# ── Prototype evaluation protocol ─────────────────────────────────────────────
NUM_REFERENCE_GENUINE: int = 5
PROTOTYPE_EVAL_SEED: int = 2026


def _load_best_model(device: torch.device) -> nn.Module:
    model = DownstreamSignatureEmbeddingModel(
        pretrained_ssl_checkpoint_path=str(_ENCODER_CHECKPOINT_PATH),
        trainable_encoder_stages=TRAINABLE_ENCODER_STAGES,
        projector_hidden_dim=PROJECTOR_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        norm_type=NORM_TYPE,
    )
    checkpoint = torch.load(_BEST_MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'Loaded downstream model from: {_BEST_MODEL_PATH}')
    print(f"  Checkpoint epoch:    {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val AUC:        {checkpoint.get('best_val_auc', 'unknown')}")
    return model


def test() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load split and build test inventory ───────────────────────────────────
    with open(_SPLIT_JSON_PATH, 'r') as f:
        splits = json.load(f)

    test_writer_ids: List[str] = splits['downstream_verification_testing']
    print(f'Test writers: {len(test_writer_ids)}')

    test_inventory_df = build_downstream_inventory(test_writer_ids, _DATA_DIR)
    print(
        f'Test inventory: {len(test_inventory_df)} images | '
        f'G={(test_inventory_df["signature_type"] == "G").sum()} | '
        f'F={(test_inventory_df["signature_type"] == "F").sum()}'
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model = _load_best_model(device)

    # ── Encode all test signatures in one batched pass ────────────────────────
    print('Encoding all test signatures...')
    all_test_paths = test_inventory_df['image_path'].astype(str).tolist()
    path_to_embedding = encode_all_signatures(
        model=model,
        image_paths=all_test_paths,
        target_size=TARGET_IMAGE_SIZE,
        device=device,
    )
    print(f'Encoded {len(path_to_embedding)} signatures.')

    # ── Prototype-based scoring ───────────────────────────────────────────────
    print(f'Building prototypes with {NUM_REFERENCE_GENUINE} reference genuines per writer...')
    score_df, protocol_df = collect_prototype_scores(
        test_inventory_df=test_inventory_df,
        path_to_embedding=path_to_embedding,
        num_reference_genuine=NUM_REFERENCE_GENUINE,
        seed=PROTOTYPE_EVAL_SEED,
    )
    print(
        f'Scored {len(score_df)} queries | '
        f'genuine: {(score_df["label"] == 1).sum()} | '
        f'forgery: {(score_df["label"] == 0).sum()}'
    )

    # ── Overall metrics ───────────────────────────────────────────────────────
    roc_df, roc_auc = compute_roc_metrics(score_df)
    threshold_info = compute_threshold_diagnostics(roc_df)

    print('\n' + '=' * 50)
    print('OVERALL TEST RESULTS')
    print('=' * 50)
    print(f'  ROC AUC:              {roc_auc:.6f}')
    print(f'  EER estimate:         {threshold_info["eer_estimate"]:.6f}')
    print(f'  EER distance thresh:  {threshold_info["eer_distance_threshold"]:.6f}')
    print(f'  Best distance thresh: {threshold_info["best_distance_threshold"]:.6f}')
    print(f'  Best threshold FPR:   {threshold_info["best_threshold_fpr"]:.6f}')
    print(f'  Best threshold TPR:   {threshold_info["best_threshold_tpr"]:.6f}')

    # ── Per-writer metrics ────────────────────────────────────────────────────
    per_writer_df = compute_per_writer_metrics(score_df)
    print('\nPer-writer AUC / EER:')
    print(per_writer_df.to_string(index=False))

    # ── Save results ──────────────────────────────────────────────────────────
    _CSV_DIR.mkdir(parents=True, exist_ok=True)

    score_df.to_csv(
        _CSV_DIR / 'bh_sig_hindi_test_scores.csv', index=False
    )
    protocol_df.to_csv(
        _CSV_DIR / 'bh_sig_hindi_test_protocol.csv', index=False
    )
    per_writer_df.to_csv(
        _CSV_DIR / 'bh_sig_hindi_test_per_writer_metrics.csv', index=False
    )

    overall_summary = {
        'roc_auc': round(roc_auc, 6),
        **{k: round(v, 6) for k, v in threshold_info.items()},
        'num_test_writers': test_inventory_df['writer_id'].nunique(),
        'num_reference_genuine': NUM_REFERENCE_GENUINE,
        'total_genuine_queries': int((score_df['label'] == 1).sum()),
        'total_forgery_queries': int((score_df['label'] == 0).sum()),
    }
    import pandas as pd
    pd.DataFrame([overall_summary]).to_csv(
        _CSV_DIR / 'bh_sig_hindi_test_overall_summary.csv', index=False
    )

    print('\nCSV files saved:')
    print(f'  bh_sig_hindi_test_scores.csv')
    print(f'  bh_sig_hindi_test_protocol.csv')
    print(f'  bh_sig_hindi_test_per_writer_metrics.csv')
    print(f'  bh_sig_hindi_test_overall_summary.csv')

    # ── ROC plot ──────────────────────────────────────────────────────────────
    plot_and_save_roc_curve(roc_df, roc_auc, threshold_info)


if __name__ == '__main__':
    test()
