import torch

def foreground_weighted_mse_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    foreground_weight: float,
    background_weight: float,
) -> torch.Tensor:

    # Extract the mask (both the foreground and background)
    # Inputs are normalized to [-1, 1]. Foreground ink pixels are positive after binary inversion.
    foreground_mask = (target > 0).float()
    background_mask = 1.0 - foreground_mask

    weight_map = (foreground_weight * foreground_mask) + (background_weight * background_mask)
    squared_error = (target - prediction) ** 2

    return (weight_map * squared_error).sum() / (weight_map.sum() + 1e-8)
