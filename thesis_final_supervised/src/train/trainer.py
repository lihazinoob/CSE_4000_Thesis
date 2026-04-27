import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict

from thesis_final_supervised.src.train.loss import DualTripletLoss, compute_triplet_distance_statistics


def get_optimizer_param_groups(model: nn.Module, weight_decay: float):
    decay_parameters = []
    no_decay_parameters = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or parameter_name.endswith('.bias'):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return [
        {'params': decay_parameters, 'weight_decay': weight_decay},
        {'params': no_decay_parameters, 'weight_decay': 0.0},
    ]


def get_model_state_dict(model: nn.Module):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def compute_batch_embeddings(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device):
    anchor = batch['anchor'].to(device)
    positive = batch['positive'].to(device)
    negative_intra = batch['negative_intra'].to(device)
    negative_inter = batch['negative_inter'].to(device)

    return {
        'anchor_embedding': model(anchor),
        'positive_embedding': model(positive),
        'negative_intra_embedding': model(negative_intra),
        'negative_inter_embedding': model(negative_inter),
    }


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: DualTripletLoss,
    device: torch.device,
    optimizer=None,
    description: str = 'Eval',
):
    is_training = optimizer is not None
    model.train(mode=is_training)
    running = {
        'loss': 0.0,
        'intra_loss': 0.0,
        'inter_loss': 0.0,
        'positive_distance_mean': 0.0,
        'negative_intra_distance_mean': 0.0,
        'negative_inter_distance_mean': 0.0,
        'intra_ranking_accuracy': 0.0,
        'inter_ranking_accuracy': 0.0,
    }
    num_batches = 0

    context_manager = torch.enable_grad() if is_training else torch.no_grad()
    with context_manager:
        progress_bar = tqdm(data_loader, desc=description, ncols=100, mininterval=10)
        for batch in progress_bar:
            if is_training:
                optimizer.zero_grad()

            embeddings = compute_batch_embeddings(model, batch, device)
            loss_outputs = loss_function(**embeddings)
            stats = compute_triplet_distance_statistics(**embeddings)

            if is_training:
                loss_outputs['loss'].backward()
                optimizer.step()

            for key in ('loss', 'intra_loss', 'inter_loss'):
                running[key] += float(loss_outputs[key].item())
            for key in ('positive_distance_mean', 'negative_intra_distance_mean', 'negative_inter_distance_mean', 'intra_ranking_accuracy', 'inter_ranking_accuracy'):
                running[key] += float(stats[key].item())
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f"{running['loss'] / num_batches:.4f}",
                'intra_acc': f"{running['intra_ranking_accuracy'] / num_batches:.4f}",
                'inter_acc': f"{running['inter_ranking_accuracy'] / num_batches:.4f}",
            })

    return {key: value / max(1, num_batches) for key, value in running.items()}


def train_model(
    model: nn.Module,
    train_dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: DualTripletLoss,
    optimizer,
    epochs: int,
    device: torch.device,
    best_model_path: Path,
    history_csv_path: Path,
) -> pd.DataFrame:
    history = []
    best_val_loss = float('inf')
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)

        print(f'===== Epoch {epoch + 1}/{epochs} =====')
        train_metrics = run_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            description='Train',
        )
        val_metrics = run_one_epoch(
            model=model,
            data_loader=val_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
            description='Validation',
        )

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_intra_loss': train_metrics['intra_loss'],
            'train_inter_loss': train_metrics['inter_loss'],
            'train_intra_ranking_accuracy': train_metrics['intra_ranking_accuracy'],
            'train_inter_ranking_accuracy': train_metrics['inter_ranking_accuracy'],
            'val_loss': val_metrics['loss'],
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
        }
        history.append(epoch_record)
        print(epoch_record)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': get_model_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
            }, best_model_path)
            print(f'Saved new best model to {best_model_path}')

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    return history_df
