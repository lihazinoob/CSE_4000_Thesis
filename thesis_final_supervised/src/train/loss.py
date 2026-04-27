import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DualTripletLoss(nn.Module):
    def __init__(
        self,
        intra_margin: float = 0.2,
        inter_margin: float = 0.2,
        inter_loss_weight: float = 1.0,
        distance_p: float = 2.0,
    ):
        super().__init__()
        self.intra_triplet = nn.TripletMarginLoss(margin=intra_margin, p=distance_p)
        self.inter_triplet = nn.TripletMarginLoss(margin=inter_margin, p=distance_p)
        self.inter_loss_weight = inter_loss_weight

    def forward(
        self,
        anchor_embedding: torch.Tensor,
        positive_embedding: torch.Tensor,
        negative_intra_embedding: torch.Tensor,
        negative_inter_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        intra_loss = self.intra_triplet(anchor_embedding, positive_embedding, negative_intra_embedding)
        inter_loss = self.inter_triplet(anchor_embedding, positive_embedding, negative_inter_embedding)
        total_loss = intra_loss + (self.inter_loss_weight * inter_loss)
        return {
            'loss': total_loss,
            'intra_loss': intra_loss,
            'inter_loss': inter_loss,
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
