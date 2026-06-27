from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for training scripts
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SSIM_MAP_DIR = _PROJECT_ROOT / 'artifacts' / 'ssim_maps'

# Number of SSIM maps saved per evaluation call (first N images in eval order)
NUM_MAPS_TO_SAVE = 5


def _save_ssim_map(
    ssim_map: np.ndarray,
    epoch: int,
    image_idx: int,
    run_name: str,
) -> None:
    """Save one SSIM spatial map as a coloured PNG under artifacts/ssim_maps/{run_name}/."""
    save_dir = _SSIM_MAP_DIR / run_name / f'epoch_{epoch:03d}'
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(ssim_map, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'SSIM map — {run_name} | epoch {epoch} | image {image_idx}', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(
        save_dir / f'mendeley_image_{image_idx:04d}_ssim_map.png',
        dpi=120,
        bbox_inches='tight',
    )
    plt.close(fig)


def evaluate_on_held_out(
    ssl_model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    epoch: Optional[int] = None,
    num_maps_to_save: int = NUM_MAPS_TO_SAVE,
    run_name: str = 'default',
) -> Dict[str, float]:
    """
    Run the model in eval mode over the held-out writer set and compute
    per-image diagnostics, then average across all images.

    Metrics
    -------
    eval_mean_fg_mse   – mean per-image foreground MSE  (ink pixels only)
    eval_mean_bg_mse   – mean per-image background MSE  (blank pixels only)
    eval_bg_fg_ratio   – bg_mse / fg_mse  (1.0 = perfectly balanced)
    eval_mean_ssim     – mean per-image SSIM score
    eval_std_ssim      – std  per-image SSIM score

    All metrics are computed per-image first, then averaged, so every
    signature contributes equally regardless of ink density.

    SSIM map saving
    ---------------
    When `epoch` is provided, the full per-pixel SSIM map for the first
    `num_maps_to_save` images is saved to:
        artifacts/ssim_maps/{run_name}/epoch_{epoch:03d}/image_XXXX_ssim_map.png

    Each run_name gets its own subdirectory, so maps from different
    weight values never overwrite each other.
    """
    ssl_model.eval()

    all_fg_mse: list = []
    all_bg_mse: list = []
    all_ssim:   list = []
    images_processed = 0

    with torch.no_grad():
        for batch in eval_loader:
            images  = batch['image'].to(device, non_blocking=True)    # (N, 1, H, W) in [-1,1]
            patches = batch['patches'].to(device, non_blocking=True)

            reconstructed, _ = ssl_model(images, patches)             # (N, 1, H, W) in [-1,1]

            # ---- per-image FG / BG MSE ----------------------------------
            sq_err = (images - reconstructed) ** 2                    # (N, 1, H, W)

            for i in range(images.shape[0]):
                fg_mask = (images[i] > 0)   # ink pixels — consistent with loss definition
                bg_mask = ~fg_mask

                fg_mse = sq_err[i][fg_mask].mean().item() if fg_mask.any() else 0.0
                bg_mse = sq_err[i][bg_mask].mean().item() if bg_mask.any() else 0.0

                all_fg_mse.append(fg_mse)
                all_bg_mse.append(bg_mse)

            # ---- per-image SSIM -----------------------------------------
            # Rescale from Tanh's [-1, 1] to [0, 1] before SSIM
            target_01 = ((images + 1) / 2).clamp(0.0, 1.0).cpu().numpy()     # (N, 1, H, W)
            pred_01   = ((reconstructed + 1) / 2).clamp(0.0, 1.0).cpu().numpy()

            for i in range(images.shape[0]):
                t = target_01[i, 0]   # (H, W)
                p = pred_01[i, 0]     # (H, W)

                score, ssim_map = ssim(t, p, data_range=1.0, full=True)
                all_ssim.append(float(score))

                # Save SSIM map for the first num_maps_to_save images
                if epoch is not None and images_processed < num_maps_to_save:
                    _save_ssim_map(ssim_map, epoch, images_processed, run_name)

                images_processed += 1

    mean_fg_mse = float(np.mean(all_fg_mse))
    mean_bg_mse = float(np.mean(all_bg_mse))
    mean_ssim   = float(np.mean(all_ssim))
    std_ssim    = float(np.std(all_ssim))
    bg_fg_ratio = mean_bg_mse / (mean_fg_mse + 1e-8)

    return {
        'eval_mean_fg_mse': mean_fg_mse,
        'eval_mean_bg_mse': mean_bg_mse,
        'eval_bg_fg_ratio': bg_fg_ratio,
        'eval_mean_ssim':   mean_ssim,
        'eval_std_ssim':    std_ssim,
    }
