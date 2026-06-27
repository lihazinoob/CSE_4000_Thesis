from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils.preprocessing import preprocess_image
from ssl_pretraining.utils.patch_extraction import extract_patches

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_IMAGE = (
    _PROJECT_ROOT / 'data' / 'all' / 'BHSig260_Hindi' / '1' / 'H-S-1-F-01.tif'
)
_PLOTS_DIR = _PROJECT_ROOT / 'artifacts' / 'plots'

_TARGET_SIZE = (256, 256)
_PATCH_SIZE = 16
_NUM_COLS = _TARGET_SIZE[1] // _PATCH_SIZE   # 16
_NUM_ROWS = _TARGET_SIZE[0] // _PATCH_SIZE   # 16


def _load_original(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img


def plot_original_vs_preprocessed(original: np.ndarray, preprocessed: np.ndarray) -> None:
    """Side-by-side: raw image vs preprocessed image with sizes annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    orig_h, orig_w = original.shape
    pre_h, pre_w = preprocessed.shape

    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image', fontsize=13, fontweight='bold')
    axes[0].set_xlabel(f'{orig_w} × {orig_h} px', fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#cccccc')

    # Preprocessed image is a binary inverted image (ink=255, bg=0).
    # Display with white background for readability.
    axes[1].imshow(preprocessed, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Preprocessed Image\n(cropped · squared · resized · binarised)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(f'{pre_w} × {pre_h} px', fontsize=11)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#cccccc')

    fig.suptitle('BHSig Hindi — Preprocessing Pipeline', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _PLOTS_DIR / 'bh_sig_hindi_preprocessing_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {save_path}')


def plot_patch_grid(preprocessed: np.ndarray) -> None:
    """Preprocessed image with patch grid overlay + individual patch thumbnails."""
    patches = extract_patches(preprocessed, patch_size=_PATCH_SIZE)

    fig = plt.figure(figsize=(14, 7))

    # ------------------------------------------------------------------ #
    # Left: preprocessed image with grid overlay and patch index labels   #
    # ------------------------------------------------------------------ #
    ax_main = fig.add_axes([0.02, 0.05, 0.44, 0.88])
    ax_main.imshow(preprocessed, cmap='gray', vmin=0, vmax=255)

    for row in range(_NUM_ROWS):
        for col in range(_NUM_COLS):
            x = col * _PATCH_SIZE
            y = row * _PATCH_SIZE
            rect = mpatches.Rectangle(
                (x - 0.5, y - 0.5), _PATCH_SIZE, _PATCH_SIZE,
                linewidth=0.8, edgecolor='#e84848', facecolor='none',
            )
            ax_main.add_patch(rect)
            patch_idx = row * _NUM_COLS + col
            ax_main.text(
                x + _PATCH_SIZE / 2, y + _PATCH_SIZE / 2,
                str(patch_idx),
                color='#e84848', fontsize=3.5, ha='center', va='center',
                fontweight='bold',
            )

    ax_main.set_xlim(-0.5, _TARGET_SIZE[1] - 0.5)
    ax_main.set_ylim(_TARGET_SIZE[0] - 0.5, -0.5)
    ax_main.set_title(
        f'Patch Grid Overlay\n({_NUM_ROWS}×{_NUM_COLS} grid, {_PATCH_SIZE}×{_PATCH_SIZE} px each)',
        fontsize=12, fontweight='bold',
    )
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # ------------------------------------------------------------------ #
    # Right: individual patch thumbnails in the same grid layout          #
    # ------------------------------------------------------------------ #
    thumb_left = 0.52
    thumb_width = 0.46
    thumb_bottom = 0.05
    thumb_height = 0.88

    cell_w = thumb_width / _NUM_COLS
    cell_h = thumb_height / _NUM_ROWS

    for idx, patch in enumerate(patches):
        row = idx // _NUM_COLS
        col = idx % _NUM_COLS

        left = thumb_left + col * cell_w
        bottom = thumb_bottom + thumb_height - (row + 1) * cell_h

        ax = fig.add_axes([left, bottom, cell_w * 0.92, cell_h * 0.92])
        ax.imshow(patch, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#e84848')
            spine.set_linewidth(0.6)
        ax.text(
            0.5, 0.97, str(idx),
            transform=ax.transAxes,
            color='#e84848', fontsize=3.0, ha='center', va='top',
            fontweight='bold',
        )

    fig.text(
        0.52 + thumb_width / 2, 0.97,
        f'Individual Patches ({_PATCH_SIZE}×{_PATCH_SIZE} px)',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
    )
    fig.suptitle(
        'BHSig Hindi — Patch Extraction from Preprocessed Image',
        fontsize=14, fontweight='bold', y=1.01,
    )

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _PLOTS_DIR / 'bh_sig_hindi_patch_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved → {save_path}')


def main() -> None:
    original = _load_original(_SAMPLE_IMAGE)
    preprocessed = preprocess_image(original, target_size=_TARGET_SIZE)

    plot_original_vs_preprocessed(original, preprocessed)
    plot_patch_grid(preprocessed)


if __name__ == '__main__':
    main()
