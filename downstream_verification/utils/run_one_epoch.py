from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


from tqdm import tqdm

from downstream_verification.loss.dual_triplet_loss import DualTripletLoss


#  A helper function to compute the embeddings for the quadruple
def compute_batch_embeddings(
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device
):

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


def compute_triplet_distance_statistics(
    anchor_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_intra_embedding: torch.Tensor,
    negative_inter_embedding: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
    negative_intra_distance = F.pairwise_distance(anchor_embedding, negative_intra_embedding)
    negative_inter_distance = F.pairwise_distance(anchor_embedding, negative_inter_embedding)

    return {
        'positive_distance_mean': positive_distance.mean(),
        'negative_intra_distance_mean': negative_intra_distance.mean(),
        'negative_inter_distance_mean': negative_inter_distance.mean(),
        'intra_ranking_accuracy': (positive_distance < negative_intra_distance).float().mean(),
        'inter_ranking_accuracy': (positive_distance < negative_inter_distance).float().mean(),
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

    #  Set the model in training or inference mode based on the optimizer parameter
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

