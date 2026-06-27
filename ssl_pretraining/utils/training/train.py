from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_pretraining.learningratescheduler.cosineLR import LinearWarmupCosineAnnealingLR
from ssl_pretraining.loss.lossfunction import foreground_weighted_mse_loss, compute_fg_bg_mse
from ssl_pretraining.utils.training.eval import evaluate_on_held_out

_PROJECT_ROOT  = Path(__file__).resolve().parents[3]
_CSV_DIR       = _PROJECT_ROOT / 'artifacts' / 'csv'
_SSL_MODEL_DIR = _PROJECT_ROOT / 'artifacts' / 'saved_models' / 'ssl_model'
_ENCODER_DIR   = _PROJECT_ROOT / 'artifacts' / 'saved_models' / 'encoder_model'


def _save_checkpoint(ssl_model: nn.Module, epoch: int, run_name: str) -> None:
    model_core = ssl_model.module if isinstance(ssl_model, torch.nn.DataParallel) else ssl_model

    ssl_model_dir = _SSL_MODEL_DIR / run_name
    encoder_dir   = _ENCODER_DIR   / run_name

    ssl_model_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model_core.state_dict(),
        ssl_model_dir / f'mendeley_epoch{epoch}.pt',
    )
    torch.save(
        model_core.encoder.state_dict(),
        encoder_dir / f'mendeley_epoch{epoch}.pt',
    )
    print(f'  [Checkpoint] Saved model and encoder at epoch {epoch}  [{run_name}]')


def train_self_supervised_model(
    ssl_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    number_of_epochs: int,
    warmup_epochs: int,
    learning_rate: float,
    foreground_weight: float,
    background_weight: float,
    run_name: str,
    save_frequency: int = 5,
    eval_loader: Optional[DataLoader] = None,
) -> None:

    optimizer = SGD(
        ssl_model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )

    steps_per_epoch = len(train_loader)
    warmup_steps    = max(1, warmup_epochs * steps_per_epoch)
    total_steps     = max(warmup_steps + 1, number_of_epochs * steps_per_epoch)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Per-run CSV paths — each weight value writes to its own files.
    step_csv  = _CSV_DIR / f'mendeley_ssl_pretraining_{run_name}_step_summary.csv'
    epoch_csv = _CSV_DIR / f'mendeley_ssl_pretraining_{run_name}_epoch_summary.csv'

    step_logs  = []
    epoch_logs = []
    global_step = 0

    _CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Run             : {run_name}')
    print(f'Foreground wt   : {foreground_weight}')
    print(f'Epochs          : {number_of_epochs}')
    print(f'Steps per epoch : {steps_per_epoch}')
    print(f'Warmup steps    : {warmup_steps}')
    print(f'Total steps     : {total_steps}')
    print(f'Learning rate   : {learning_rate}')
    print(f'Eval loader     : {"yes" if eval_loader is not None else "none"}')
    print('=' * 60)

    for epoch in range(number_of_epochs):
        ssl_model.train()
        epoch_loss       = 0.0
        epoch_fg_mse_sum = 0.0
        epoch_bg_mse_sum = 0.0
        num_batches      = 0
        epoch_start_lr   = optimizer.param_groups[0]['lr']

        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f'[{run_name}] Epoch {epoch + 1}/{number_of_epochs}',
            leave=True,
        )

        for batch_index, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)

            image   = batch['image'].to(device, non_blocking=True)
            patches = batch['patches'].to(device, non_blocking=True)

            reconstructed_image, patch_attn_maps = ssl_model(image, patches)

            reconstruction_loss = foreground_weighted_mse_loss(
                target=image,
                prediction=reconstructed_image,
                foreground_weight=foreground_weight,
                background_weight=background_weight,
            )

            reconstruction_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                batch_fg_mse, batch_bg_mse = compute_fg_bg_mse(image, reconstructed_image)

            epoch_loss       += float(reconstruction_loss.item())
            epoch_fg_mse_sum += batch_fg_mse
            epoch_bg_mse_sum += batch_bg_mse
            num_batches      += 1
            global_step      += 1

            avg_loss   = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']

            step_logs.append({
                'run_name':                 run_name,
                'foreground_weight':        foreground_weight,
                'epoch':                    epoch + 1,
                'epoch_step':               batch_index + 1,
                'global_step':              global_step,
                'batch_recons_loss':        float(reconstruction_loss.item()),
                'running_avg_recons_loss':  avg_loss,
                'lr':                       current_lr,
                'batch_size':               int(image.shape[0]),
                'batch_fg_mse':             batch_fg_mse,
                'batch_bg_mse':             batch_bg_mse,
            })

            progress_bar.set_postfix({
                'loss':     f'{reconstruction_loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr':       f'{current_lr:.6f}',
            })

        # ---- epoch-end training metrics --------------------------------
        epoch_avg_loss = epoch_loss / max(1, num_batches)
        epoch_end_lr   = optimizer.param_groups[0]['lr']

        epoch_log = {
            'run_name':                run_name,
            'foreground_weight':       foreground_weight,
            'epoch':                   epoch + 1,
            'epoch_avg_recons_loss':   epoch_avg_loss,
            'epoch_start_lr':          epoch_start_lr,
            'epoch_end_lr':            epoch_end_lr,
            'num_batches':             num_batches,
            'global_step_end':         global_step,
            'epoch_avg_fg_mse':        epoch_fg_mse_sum / max(1, num_batches),
            'epoch_avg_bg_mse':        epoch_bg_mse_sum / max(1, num_batches),
        }

        # ---- held-out evaluation (per-image, then averaged) ------------
        if eval_loader is not None:
            print(f'  [Eval] Running held-out evaluation after epoch {epoch + 1} ...')
            eval_metrics = evaluate_on_held_out(
                ssl_model,
                eval_loader,
                device,
                epoch=epoch + 1,
                run_name=run_name,
            )
            epoch_log.update(eval_metrics)
            print(
                f'  [Eval] fg_mse={eval_metrics["eval_mean_fg_mse"]:.6f}  '
                f'bg_mse={eval_metrics["eval_mean_bg_mse"]:.6f}  '
                f'ratio={eval_metrics["eval_bg_fg_ratio"]:.3f}  '
                f'ssim={eval_metrics["eval_mean_ssim"]:.4f} '
                f'(±{eval_metrics["eval_std_ssim"]:.4f})'
            )
            ssl_model.train()

        epoch_logs.append(epoch_log)

        current_epoch = epoch + 1
        if current_epoch % save_frequency == 0:
            _save_checkpoint(ssl_model, current_epoch, run_name)

        pd.DataFrame(step_logs).to_csv(step_csv,  index=False)
        pd.DataFrame(epoch_logs).to_csv(epoch_csv, index=False)

    if number_of_epochs % save_frequency != 0:
        _save_checkpoint(ssl_model, number_of_epochs, run_name)
