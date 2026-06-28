from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from downstream_verification.datasets.DynamicDualTripletDataset import DynamicDualTripletDataset
from downstream_verification.evaluation.val_protocol import run_prototype_validation
from downstream_verification.loss.dual_triplet_loss import DualTripletLoss
from downstream_verification.utils.run_one_epoch import run_one_epoch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CSV_DIR = _PROJECT_ROOT / 'artifacts' / 'csv'
_DOWNSTREAM_MODEL_DIR = _PROJECT_ROOT / 'artifacts' / 'saved_models' / 'downstream_verification'

_BEST_MODEL_NAME = 'best_model_bh_sig_hindi_verification.pt'
_PERIODIC_SAVE_FREQUENCY = 5


def _get_model_state_dict(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def train_and_validate_model(
    model: nn.Module,
    train_dataset: DynamicDualTripletDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_inventory_df: pd.DataFrame,
    target_image_size: Tuple[int, int],
    loss_function: DualTripletLoss,
    optimizer,
    epochs: int,
    device: torch.device,
) -> pd.DataFrame:

    _CSV_DIR.mkdir(parents=True, exist_ok=True)
    _DOWNSTREAM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_auc = 0.0

    for epoch in range(epochs):
        current_epoch = epoch + 1
        print(f'===== Epoch {current_epoch}/{epochs} =====')

        train_dataset.set_epoch(epoch)

        train_metrics = run_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            description='Train',
        )
        # Triplet validation — kept as a training-health diagnostic, not used for selection
        val_metrics = run_one_epoch(
            model=model,
            data_loader=val_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
            description='Val Triplet',
        )

        print('Running prototype validation...')
        proto_metrics = run_prototype_validation(
            model=model,
            val_inventory_df=val_inventory_df,
            target_size=target_image_size,
            device=device,
        )

        epoch_record = {
            'epoch': current_epoch,
            'train_loss': train_metrics['loss'],
            'train_intra_loss': train_metrics['intra_loss'],
            'train_inter_loss': train_metrics['inter_loss'],
            'train_intra_ranking_accuracy': train_metrics['intra_ranking_accuracy'],
            'train_inter_ranking_accuracy': train_metrics['inter_ranking_accuracy'],
            'val_triplet_loss': val_metrics['loss'],
            'val_intra_loss': val_metrics['intra_loss'],
            'val_inter_loss': val_metrics['inter_loss'],
            'val_intra_ranking_accuracy': val_metrics['intra_ranking_accuracy'],
            'val_inter_ranking_accuracy': val_metrics['inter_ranking_accuracy'],
            'train_positive_distance_mean': train_metrics['positive_distance_mean'],
            'train_negative_intra_distance_mean': train_metrics['negative_intra_distance_mean'],
            'train_negative_inter_distance_mean': train_metrics['negative_inter_distance_mean'],
            'val_positive_distance_mean': val_metrics['positive_distance_mean'],
            'val_negative_intra_distance_mean': val_metrics['negative_intra_distance_mean'],
            'val_negative_inter_distance_mean': val_metrics['negative_inter_distance_mean'],
            'val_proto_auc_mean': proto_metrics['val_proto_auc_mean'],
            'val_proto_auc_std': proto_metrics['val_proto_auc_std'],
            'val_proto_per_writer_auc_mean': proto_metrics['val_proto_per_writer_auc_mean'],
            'val_proto_eer_mean': proto_metrics['val_proto_eer_mean'],
        }
        history.append(epoch_record)
        print(epoch_record)

        # Checkpoint selection: save on val_proto_auc_mean improvement
        if proto_metrics['val_proto_auc_mean'] > best_val_auc:
            best_val_auc = proto_metrics['val_proto_auc_mean']
            best_model_path = _DOWNSTREAM_MODEL_DIR / _BEST_MODEL_NAME
            torch.save(
                {
                    'epoch': current_epoch,
                    'model_state_dict': _get_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_auc': best_val_auc,
                    'history': history,
                },
                best_model_path,
            )
            print(f'Best model saved → {best_model_path}')

        # Periodic checkpoint every 5 epochs (unchanged behaviour)
        if current_epoch % _PERIODIC_SAVE_FREQUENCY == 0:
            periodic_path = _DOWNSTREAM_MODEL_DIR / f'bh_sig_hindi_epoch{current_epoch}.pt'
            torch.save(
                {
                    'epoch': current_epoch,
                    'model_state_dict': _get_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_triplet_loss': val_metrics['loss'],
                    'val_proto_auc_mean': proto_metrics['val_proto_auc_mean'],
                    'history': history,
                },
                periodic_path,
            )
            print(f'Periodic checkpoint saved → {periodic_path}')

        # Flush CSV after every epoch so a mid-training crash loses no history
        history_df = pd.DataFrame(history)
        history_df.to_csv(
            _CSV_DIR / 'bh_sig_hindi_downstream_verification_history.csv',
            index=False,
        )

    return pd.DataFrame(history)
