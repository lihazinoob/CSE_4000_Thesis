from typing import Dict, List, Tuple

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.preprocessing import preprocess_image
from utils.transformation import transform_preprocessed_image


class _SignaturePathDataset(Dataset):
    def __init__(self, image_paths: List[str], target_size: Tuple[int, int]) -> None:
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = transform_preprocessed_image()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Could not read image: {path}')
        processed = preprocess_image(image, target_size=self.target_size)
        return self.transform(processed), path


@torch.no_grad()
def encode_all_signatures(
    model: nn.Module,
    image_paths: List[str],
    target_size: Tuple[int, int],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Encode every image path into an L2-normalized embedding in one batched pass.

    Args:
        model:       Trained DownstreamSignatureEmbeddingModel in eval mode.
        image_paths: All unique image paths to encode.
        target_size: Must match the resolution used during training (256, 256).
        device:      CUDA or CPU device.
        batch_size:  Images processed per forward pass.

    Returns:
        Dict mapping image_path (str) → embedding tensor (CPU, shape [embedding_dim]).
    """
    model.eval()
    dataset = _SignaturePathDataset(image_paths, target_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    path_to_embedding: Dict[str, torch.Tensor] = {}

    for image_tensors, paths in loader:
        embeddings = model(image_tensors.to(device))
        for path, embedding in zip(paths, embeddings):
            path_to_embedding[path] = embedding.detach().cpu()

    return path_to_embedding
