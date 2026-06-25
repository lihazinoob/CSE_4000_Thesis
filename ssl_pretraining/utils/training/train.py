from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_pretraining.learningratescheduler.cosineLR import LinearWarmupCosineAnnealingLR
from ssl_pretraining.loss.lossfunction import foreground_weighted_mse_loss

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CSV_DIR = _PROJECT_ROOT / 'artifacts' / 'csv'
_SSL_MODEL_DIR = _PROJECT_ROOT / 'artifacts'/ 'saved_models' / 'ssl_model'
_ENCODER_DIR = _PROJECT_ROOT / 'artifacts'/ 'saved_models' / 'encoder_model'


def _save_checkpoint(ssl_model: nn.Module, epoch: int) -> None:
    model_core = ssl_model.module if isinstance(ssl_model, torch.nn.DataParallel) else ssl_model

    _SSL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model_core.state_dict(),
        _SSL_MODEL_DIR / f'self_supervised_model_epoch{epoch}.pt'
    )
    torch.save(
        model_core.encoder.state_dict(),
        _ENCODER_DIR / f'encoder_epoch{epoch}.pt'
    )
    print(f'  [Checkpoint] Saved model and encoder at epoch {epoch}')


def train_self_supervised_model(
    ssl_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    number_of_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    foreground_weight: float,
    background_weight: float,
    save_frequency: int = 5,
) -> None:

    optimizer = SGD(
        ssl_model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, number_of_epochs * steps_per_epoch)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    step_logs = []
    epoch_logs = []
    global_step = 0

    _CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Epochs: {number_of_epochs}')
    print(f'Steps per epoch: {steps_per_epoch}')
    print(f'Warmup steps: {warmup_steps}')
    print(f'Total steps: {total_steps}')
    print(f'Learning rate: {learning_rate}')
    print('=' * 60)

    for epoch in range(number_of_epochs):
        ssl_model.train()
        epoch_loss = 0
        num_batches = 0
        epoch_start_lr = optimizer.param_groups[0]['lr']
        epoch_foreground_ratio_sum = 0.0
        epoch_active_patch_sum = 0.0
        epoch_patch_weight_max_sum = 0.0

        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f'Epoch {epoch + 1}/{number_of_epochs}',
            leave=True
        )

        for batch_index, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            image = batch['image'].to(device, non_blocking=True)
            patches = batch['patches'].to(device, non_blocking=True)

            reconstructed_image, patch_attn_maps, foreground_ratio, foreground_prior, patch_scores, patch_weights = ssl_model(
                image,
                patches
            )

            reconstruction_loss = foreground_weighted_mse_loss(
                target=image,
                prediction=reconstructed_image,
                foreground_weight=foreground_weight,
                background_weight=background_weight,
            )

            reconstruction_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(reconstruction_loss.item())
            num_batches += 1
            global_step += 1

            avg_loss = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']

            mean_foreground_ratio = float(foreground_ratio.mean().item())
            mean_active_patches = float(patch_scores.sum(dim=1).float().mean().item())
            mean_patch_weight_max = float(patch_weights.max(dim=1).values.mean().item())

            epoch_foreground_ratio_sum += mean_foreground_ratio
            epoch_active_patch_sum += mean_active_patches
            epoch_patch_weight_max_sum += mean_patch_weight_max

            step_logs.append({
                'epoch': epoch + 1,
                'epoch_step': batch_index + 1,
                'global_step': global_step,
                'batch_recons_loss': float(reconstruction_loss.item()),
                'running_avg_recons_loss': avg_loss,
                'lr': current_lr,
                'batch_size': int(image.shape[0]),
                'mean_foreground_ratio': mean_foreground_ratio,
                'mean_active_patches': mean_active_patches,
                'mean_patch_weight_max': mean_patch_weight_max,
            })

            progress_bar.set_postfix({
                'loss': f'{reconstruction_loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.6f}',
            })

        epoch_avg_loss = epoch_loss / max(1, num_batches)
        epoch_end_lr = optimizer.param_groups[0]['lr']
        epoch_logs.append({
            'epoch': epoch + 1,
            'epoch_avg_recons_loss': epoch_avg_loss,
            'epoch_start_lr': epoch_start_lr,
            'epoch_end_lr': epoch_end_lr,
            'num_batches': num_batches,
            'global_step_end': global_step,
            'epoch_mean_foreground_ratio': epoch_foreground_ratio_sum / max(1, num_batches),
            'epoch_mean_active_patches': epoch_active_patch_sum / max(1, num_batches),
            'epoch_mean_patch_weight_max': epoch_patch_weight_max_sum / max(1, num_batches),
        })

        current_epoch = epoch + 1
        if current_epoch % save_frequency == 0:
            _save_checkpoint(ssl_model, current_epoch)

        pd.DataFrame(step_logs).to_csv(_CSV_DIR / 'ssl_pretraining_step_summary.csv', index=False)
        pd.DataFrame(epoch_logs).to_csv(_CSV_DIR / 'ssl_pretraining_epoch_summary.csv', index=False)

    # Save final checkpoint if the last epoch wasn't already a save_frequency multiple
    if number_of_epochs % save_frequency != 0:
        _save_checkpoint(ssl_model, number_of_epochs)
