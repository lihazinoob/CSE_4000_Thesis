import random
from typing import Dict, List, Tuple

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_image
from utils.transformation import transform_preprocessed_image


def _load_signature_tensor(
    image_path: str,
    target_size: Tuple[int, int],
    transform,
) -> torch.Tensor:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')
    processed = preprocess_image(image, target_size=target_size)
    return transform(processed)


def _build_writer_path_pools(
    inventory_df: pd.DataFrame,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    writer_to_genuine_paths: Dict[int, List[str]] = {}
    writer_to_forgery_paths: Dict[int, List[str]] = {}

    for writer_id, writer_df in inventory_df.groupby('writer_id'):
        genuine_paths = sorted(
            writer_df.loc[writer_df['signature_type'] == 'G', 'image_path'].astype(str).tolist()
        )
        forgery_paths = sorted(
            writer_df.loc[writer_df['signature_type'] == 'F', 'image_path'].astype(str).tolist()
        )

        if len(genuine_paths) < 2:
            raise ValueError(
                f'Writer {writer_id} has fewer than two genuine signatures '
                f'(found {len(genuine_paths)}). Cannot form anchor-positive pairs.'
            )
        if len(forgery_paths) < 1:
            raise ValueError(
                f'Writer {writer_id} has no forgery signatures. '
                f'Cannot form intra-negative samples.'
            )

        writer_to_genuine_paths[int(writer_id)] = genuine_paths
        writer_to_forgery_paths[int(writer_id)] = forgery_paths

    return writer_to_genuine_paths, writer_to_forgery_paths


class DynamicDualTripletDataset(Dataset):
    """
    Training dataset that emits 4-tuples:
        (anchor_genuine, positive_genuine, negative_intra_forgery, negative_inter_genuine)

    - anchor and positive are both genuine signatures from the same writer.
    - negative_intra is a skilled forgery of the same writer (intra-writer, wrong type).
    - negative_inter is a genuine signature from a *different* writer (inter-writer, right type).

    Triplets are re-sampled every epoch via set_epoch() so the model sees diverse
    combinations without storing all combinations up front.

    Args:
        inventory_df: DataFrame with columns [writer_id (int), signature_type ('G'|'F'), image_path (str)].
        target_size:  (H, W) to resize images to. Must match SSL pre-training resolution (256, 256).
        seed:         Base seed for reproducible but epoch-varying sampling.
    """

    def __init__(
        self,
        inventory_df: pd.DataFrame,
        target_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.target_size = target_size
        self.seed = int(seed)
        self.current_epoch = 0
        self.transform = transform_preprocessed_image()

        self.writer_to_genuine_paths, self.writer_to_forgery_paths = _build_writer_path_pools(
            inventory_df.reset_index(drop=True).copy()
        )
        self.writer_ids: List[int] = sorted(self.writer_to_genuine_paths.keys())

        if len(self.writer_ids) < 2:
            raise ValueError(
                f'Dataset needs at least two writers for inter-negative sampling '
                f'(found {len(self.writer_ids)}).'
            )

        # One anchor record per genuine signature — dataset length scales with genuine count.
        self.anchor_records: List[Dict[str, object]] = []
        for writer_id in self.writer_ids:
            for anchor_path in self.writer_to_genuine_paths[writer_id]:
                self.anchor_records.append({
                    'writer_id': writer_id,
                    'anchor_path': anchor_path,
                })

    def __len__(self) -> int:
        return len(self.anchor_records)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch's DataLoader iteration to refresh triplet sampling."""
        self.current_epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.anchor_records[index]
        anchor_writer_id: int = int(record['writer_id'])
        anchor_path: str = str(record['anchor_path'])

        # Per-sample RNG that varies by epoch so sampling changes each epoch.
        rng = random.Random(self.seed + (self.current_epoch * 100003) + index)

        # Positive: a different genuine from the same writer.
        positive_candidates = [
            p for p in self.writer_to_genuine_paths[anchor_writer_id] if p != anchor_path
        ]
        positive_path: str = rng.choice(positive_candidates)

        # Intra-negative: a skilled forgery from the same writer.
        negative_intra_path: str = rng.choice(self.writer_to_forgery_paths[anchor_writer_id])

        # Inter-negative: a genuine from a different writer.
        inter_writer_id: int = rng.choice(
            [wid for wid in self.writer_ids if wid != anchor_writer_id]
        )
        negative_inter_path: str = rng.choice(self.writer_to_genuine_paths[inter_writer_id])

        return {
            'anchor': _load_signature_tensor(anchor_path, self.target_size, self.transform),
            'positive': _load_signature_tensor(positive_path, self.target_size, self.transform),
            'negative_intra': _load_signature_tensor(negative_intra_path, self.target_size, self.transform),
            'negative_inter': _load_signature_tensor(negative_inter_path, self.target_size, self.transform),
            'anchor_writer_id': anchor_writer_id,
            'inter_writer_id': inter_writer_id,
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_intra_path': negative_intra_path,
            'negative_inter_path': negative_inter_path,
        }
