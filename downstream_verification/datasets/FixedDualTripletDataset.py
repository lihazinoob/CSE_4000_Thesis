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


def build_fixed_triplet_records(
    inventory_df: pd.DataFrame,
    tuples_per_anchor: int = 4,
    seed: int = 314,
) -> List[Dict[str, object]]:
    """
    Build a deterministic list of 4-tuple records from the given inventory.

    For each genuine anchor, `tuples_per_anchor` records are generated using a fixed
    seed derived from writer_id and anchor index. The resulting list is identical
    across runs, making validation and test loss directly comparable across epochs.

    Args:
        inventory_df:      DataFrame with columns [writer_id (int), signature_type ('G'|'F'), image_path (str)].
        tuples_per_anchor: Number of fixed 4-tuples generated per genuine anchor signature.
        seed:              Base seed for deterministic record construction.

    Returns:
        List of dicts, each with keys:
            anchor_writer_id, inter_writer_id,
            anchor_path, positive_path, negative_intra_path, negative_inter_path.
    """
    writer_to_genuine_paths, writer_to_forgery_paths = _build_writer_path_pools(inventory_df)
    writer_ids = sorted(writer_to_genuine_paths.keys())
    records: List[Dict[str, object]] = []

    for writer_id in writer_ids:
        inter_writer_candidates = [wid for wid in writer_ids if wid != writer_id]
        genuine_paths = writer_to_genuine_paths[writer_id]
        forgery_paths = writer_to_forgery_paths[writer_id]

        for anchor_index, anchor_path in enumerate(genuine_paths):
            rng = random.Random(seed + (writer_id * 1009) + anchor_index)
            positive_candidates = [p for p in genuine_paths if p != anchor_path]

            for tuple_index in range(tuples_per_anchor):
                positive_path = positive_candidates[tuple_index % len(positive_candidates)]
                negative_intra_path = forgery_paths[tuple_index % len(forgery_paths)]
                inter_writer_id = inter_writer_candidates[rng.randrange(len(inter_writer_candidates))]
                inter_genuine_paths = writer_to_genuine_paths[inter_writer_id]
                negative_inter_path = inter_genuine_paths[tuple_index % len(inter_genuine_paths)]

                records.append({
                    'anchor_writer_id': writer_id,
                    'inter_writer_id': inter_writer_id,
                    'anchor_path': anchor_path,
                    'positive_path': positive_path,
                    'negative_intra_path': negative_intra_path,
                    'negative_inter_path': negative_inter_path,
                })

    return records


class FixedDualTripletDataset(Dataset):
    """
    Validation / test dataset backed by a pre-built, deterministic list of 4-tuple records.

    Unlike DynamicDualTripletDataset, the triplet assignments never change between epochs,
    so validation and test loss values are directly comparable across training runs.

    Typical usage:
        records = build_fixed_triplet_records(val_inventory_df, tuples_per_anchor=4, seed=314)
        val_dataset = FixedDualTripletDataset(records, target_size=(256, 256))

    Args:
        triplet_records: Output of build_fixed_triplet_records().
        target_size:     (H, W) to resize images to. Must match SSL pre-training resolution (256, 256).
    """

    def __init__(
        self,
        triplet_records: List[Dict[str, object]],
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        if not triplet_records:
            raise ValueError('triplet_records is empty. Cannot build FixedDualTripletDataset.')
        self.triplet_records = list(triplet_records)
        self.target_size = target_size
        self.transform = transform_preprocessed_image()

    def __len__(self) -> int:
        return len(self.triplet_records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.triplet_records[index]
        anchor_path: str = str(record['anchor_path'])
        positive_path: str = str(record['positive_path'])
        negative_intra_path: str = str(record['negative_intra_path'])
        negative_inter_path: str = str(record['negative_inter_path'])

        return {
            'anchor': _load_signature_tensor(anchor_path, self.target_size, self.transform),
            'positive': _load_signature_tensor(positive_path, self.target_size, self.transform),
            'negative_intra': _load_signature_tensor(negative_intra_path, self.target_size, self.transform),
            'negative_inter': _load_signature_tensor(negative_inter_path, self.target_size, self.transform),
            'anchor_writer_id': int(record['anchor_writer_id']),
            'inter_writer_id': int(record['inter_writer_id']),
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_intra_path': negative_intra_path,
            'negative_inter_path': negative_inter_path,
        }
