from typing import List

import numpy as np

def extract_patches(
        image: np.ndarray,
        patch_size: int
) -> List[np.ndarray]:
    h, w = image.shape
    patches = []

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch)
    return patches