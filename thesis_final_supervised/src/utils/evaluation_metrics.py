from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_curve

from thesis_final_supervised.src.utils.evaluation_embeddings import encode_signature_paths


def collect_prototype_score_rows(
        writer_to_protocol: Dict[int, Dict[str, List[str]]],
        writer_to_prototype: Dict[int, torch.Tensor],
        path_to_embedding: Dict[str, torch.Tensor],
) -> pd.DataFrame:
    score_rows = []

    for writer_id in sorted(writer_to_protocol):
        prototype_embedding = writer_to_prototype[writer_id]
        genuine_query_paths = writer_to_protocol[writer_id]['genuine_query_paths']
        forgery_query_paths = writer_to_protocol[writer_id]['forgery_query_paths']

        for query_path in genuine_query_paths:
            distance = F.pairwise_distance(
                path_to_embedding[query_path].unsqueeze(0),
                prototype_embedding.unsqueeze(0)
            ).item()
            score_rows.append({
                'writer_id': writer_id,
                'query_path': query_path,
                'query_type': 'genuine',
                'label': 1,
                'distance': distance,
                'score': -distance,
            })

        for query_path in forgery_query_paths:
            distance = F.pairwise_distance(
                path_to_embedding[query_path].unsqueeze(0),
                prototype_embedding.unsqueeze(0)
            ).item()
            score_rows.append({
                'writer_id': writer_id,
                'query_path': query_path,
                'query_type': 'forgery',
                'label': 0,
                'distance': distance,
                'score': -distance,
            })

    return pd.DataFrame(score_rows)


def build_protocol_summary_df(
        writer_to_protocol: Dict[int, Dict[str, List[str]]]
) -> pd.DataFrame:
    protocol_rows = []

    for writer_id in sorted(writer_to_protocol):
        protocol_rows.append({
            'writer_id': writer_id,
            'num_reference_genuine': len(writer_to_protocol[writer_id]['reference_genuine_paths']),
            'num_genuine_queries': len(writer_to_protocol[writer_id]['genuine_query_paths']),
            'num_forgery_queries': len(writer_to_protocol[writer_id]['forgery_query_paths']),
            'reference_paths': writer_to_protocol[writer_id]['reference_genuine_paths'],
        })

    return pd.DataFrame(protocol_rows)


@torch.no_grad()
def collect_prototype_scores(
        model,
        writer_to_protocol: Dict[int, Dict[str, List[str]]],
        writer_to_prototype: Dict[int, torch.Tensor],
        device: torch.device,
        target_size=(512, 512),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_query_paths = []
    for writer_id in sorted(writer_to_protocol):
        all_query_paths.extend(writer_to_protocol[writer_id]['genuine_query_paths'])
        all_query_paths.extend(writer_to_protocol[writer_id]['forgery_query_paths'])

    path_to_embedding = encode_signature_paths(
        model=model,
        image_paths=sorted(all_query_paths),
        device=device,
        target_size=target_size,
    )

    score_df = collect_prototype_score_rows(
        writer_to_protocol=writer_to_protocol,
        writer_to_prototype=writer_to_prototype,
        path_to_embedding=path_to_embedding,
    )
    protocol_df = build_protocol_summary_df(writer_to_protocol)
    return score_df, protocol_df


def compute_roc_metrics(score_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    fpr, tpr, thresholds = roc_curve(
        score_df['label'].to_numpy(),
        score_df['score'].to_numpy()
    )
    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds,
    })
    return roc_df, float(auc(fpr, tpr))


def compute_threshold_diagnostics(roc_df: pd.DataFrame) -> Dict[str, float]:
    working_df = roc_df.copy()
    working_df['fnr'] = 1.0 - working_df['tpr']
    working_df['youden_j'] = working_df['tpr'] - working_df['fpr']
    working_df['eer_gap'] = (working_df['fpr'] - working_df['fnr']).abs()

    best_youden_row = working_df.loc[working_df['youden_j'].idxmax()]
    eer_row = working_df.loc[working_df['eer_gap'].idxmin()]

    return {
        'best_score_threshold': float(best_youden_row['threshold']),
        'best_distance_threshold': float(-best_youden_row['threshold']),
        'best_threshold_fpr': float(best_youden_row['fpr']),
        'best_threshold_tpr': float(best_youden_row['tpr']),
        'eer_estimate': float((eer_row['fpr'] + eer_row['fnr']) / 2.0),
        'eer_distance_threshold': float(-eer_row['threshold']),
    }


def apply_distance_threshold(
        score_df: pd.DataFrame,
        distance_threshold: float
) -> pd.DataFrame:
    evaluated_df = score_df.copy()
    evaluated_df['predicted_label'] = (
        evaluated_df['distance'] <= distance_threshold
    ).astype(int)
    return evaluated_df


def compute_fixed_threshold_metrics(
        score_df: pd.DataFrame,
        distance_threshold: float
) -> Dict[str, float]:
    evaluated_df = apply_distance_threshold(
        score_df=score_df,
        distance_threshold=distance_threshold,
    )
    labels = evaluated_df['label'].to_numpy()
    predictions = evaluated_df['predicted_label'].to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    accuracy = float((tp + tn) / len(evaluated_df)) if len(evaluated_df) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    error_rate = float((fp + fn) / len(evaluated_df)) if len(evaluated_df) > 0 else 0.0
    f1_score = float(
        (2 * precision * recall) / (precision + recall)
    ) if (precision + recall) > 0 else 0.0

    return {
        'distance_threshold': float(distance_threshold),
        'accuracy': accuracy,
        'tpr_recall_sensitivity': recall,
        'far': far,
        'frr': frr,
        'fpr': far,
        'fnr': frr,
        'precision': precision,
        'specificity_tnr': specificity,
        'tnr': specificity,
        'error_rate': error_rate,
        'err': error_rate,
        'f1_score': f1_score,
        'recall': recall,
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
        'num_samples': int(len(evaluated_df)),
    }


def select_validation_threshold(
        validation_score_df: pd.DataFrame
) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    validation_roc_df, validation_roc_auc = compute_roc_metrics(validation_score_df)
    threshold_info = compute_threshold_diagnostics(validation_roc_df)
    return (
        validation_roc_df,
        validation_roc_auc,
        threshold_info,
    )
