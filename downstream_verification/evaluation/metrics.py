from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import auc, roc_curve


def compute_roc_metrics(score_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Compute the ROC curve and AUC from a score DataFrame.

    Args:
        score_df: DataFrame with columns 'label' (1|0) and 'score' (higher = more genuine).

    Returns:
        roc_df:  DataFrame with columns fpr, tpr, threshold.
        roc_auc: Scalar AUC value.
    """
    fpr, tpr, thresholds = roc_curve(
        score_df['label'].to_numpy(),
        score_df['score'].to_numpy(),
    )
    roc_auc = float(auc(fpr, tpr))
    return pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}), roc_auc


def compute_threshold_diagnostics(roc_df: pd.DataFrame) -> Dict[str, float]:
    """
    Derive EER and best operating threshold from the ROC DataFrame.

    EER is estimated at the point where FPR ≈ FNR (1 - TPR).
    Best threshold maximises Youden's J = TPR - FPR.

    Returns dict with keys:
        eer_estimate, eer_distance_threshold,
        best_distance_threshold, best_threshold_fpr, best_threshold_tpr.
    """
    df = roc_df.copy()
    df['fnr'] = 1.0 - df['tpr']
    df['youden_j'] = df['tpr'] - df['fpr']
    df['eer_gap'] = (df['fpr'] - df['fnr']).abs()

    best_row = df.loc[df['youden_j'].idxmax()]
    eer_row = df.loc[df['eer_gap'].idxmin()]

    return {
        'eer_estimate': float((eer_row['fpr'] + eer_row['fnr']) / 2.0),
        'eer_distance_threshold': float(-eer_row['threshold']),
        'best_distance_threshold': float(-best_row['threshold']),
        'best_threshold_fpr': float(best_row['fpr']),
        'best_threshold_tpr': float(best_row['tpr']),
    }


def compute_per_writer_metrics(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AUC and EER independently for each test writer.

    Writers with only one class present (no forgeries or no genuine queries)
    are skipped with a warning.

    Returns:
        DataFrame with columns: writer_id, auc, eer_estimate.
    """
    rows = []
    for writer_id, writer_df in score_df.groupby('writer_id'):
        if writer_df['label'].nunique() < 2:
            print(f'Warning: writer {writer_id} has only one label class — skipping AUC.')
            continue
        roc_df, writer_auc = compute_roc_metrics(writer_df)
        diagnostics = compute_threshold_diagnostics(roc_df)
        rows.append({
            'writer_id': writer_id,
            'auc': round(writer_auc, 6),
            'eer_estimate': round(diagnostics['eer_estimate'], 6),
        })
    return pd.DataFrame(rows)
