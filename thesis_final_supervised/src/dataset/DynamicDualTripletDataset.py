import random
from typing import Tuple, Dict, List
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    if image is None:
        raise ValueError('Received a null image during preprocessing.')

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(thresh)
    if coordinates is None:
        raise ValueError('No foreground pixels were detected.')

    x, y, w, h = cv2.boundingRect(coordinates)
    crop = image[y:y + h, x:x + w]

    height_target, width_target = target_size
    height_crop, width_crop = crop.shape
    scale = min(width_target / max(1, width_crop), height_target / max(1, height_crop))
    new_width = max(1, int(round(width_crop * scale)))
    new_height = max(1, int(round(height_crop * scale)))
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.ones(target_size, dtype=np.uint8) * 255
    x_offset = (width_target - new_width) // 2
    y_offset = (height_target - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    _, final_img = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return final_img


SIGNATURE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def load_signature_tensor(image_path: str, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    processed = preprocess_image(image, target_size=target_size)
    return SIGNATURE_TRANSFORM(processed)


def build_writer_path_pools(
    inventory_df: pd.DataFrame
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    writer_to_genuine_paths = {}
    writer_to_forgery_paths = {}

    for writer_id, writer_df in inventory_df.groupby('writer_id'):
        genuine_paths = sorted(writer_df.loc[writer_df['signature_type'] == 'genuine', 'image_path'].astype(str).tolist())
        forgery_paths = sorted(writer_df.loc[writer_df['signature_type'] == 'forgery', 'image_path'].astype(str).tolist())

        if len(genuine_paths) < 2:
            raise ValueError(f'Writer {writer_id} has fewer than two genuine signatures.')
        if len(forgery_paths) < 1:
            raise ValueError(f'Writer {writer_id} has no forgery signatures.')

        writer_to_genuine_paths[int(writer_id)] = genuine_paths
        writer_to_forgery_paths[int(writer_id)] = forgery_paths

    return writer_to_genuine_paths, writer_to_forgery_paths

def load_triplet_sample(
    anchor_path: str,
    positive_path: str,
    negative_intra_path: str,
    negative_inter_path: str,
    anchor_writer_id: int,
    inter_writer_id: int,
    target_size: Tuple[int, int],
) -> Dict[str, object]:
    return {
        'anchor': load_signature_tensor(anchor_path, target_size),
        'positive': load_signature_tensor(positive_path, target_size),
        'negative_intra': load_signature_tensor(negative_intra_path, target_size),
        'negative_inter': load_signature_tensor(negative_inter_path, target_size),
        'anchor_writer_id': int(anchor_writer_id),
        'inter_writer_id': int(inter_writer_id),
        'anchor_path': str(anchor_path),
        'positive_path': str(positive_path),
        'negative_intra_path': str(negative_intra_path),
        'negative_inter_path': str(negative_inter_path),
    }

class DynamicDualTripletDataset(Dataset):
    def __init__(
            self,
            inventory_dataframe: pd.DataFrame,
            target_size: Tuple[int, int] = (512, 512),
            seed: int = 42
    ):
        self.inventory_dataframe = inventory_dataframe.reset_index(drop=True).copy()
        self.target_size = target_size
        self.seed = int(seed)
        self.current_epoch = 0
        self.writer_to_genuine_paths, self.writer_to_forgery_paths = build_writer_path_pools(
            self.inventory_dataframe,
        )
        self.writer_ids = sorted(self.writer_to_genuine_paths.keys())
        self.anchor_records = []

        for writer_id in self.writer_ids:
            for anchor_path in self.writer_to_genuine_paths[writer_id]:
                self.anchor_records.append({'writer_id': writer_id, 'anchor_path': anchor_path})

    def __len__(self) -> int:
        return len(self.anchor_records)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def __getitem__(self, index: int):
        record = self.anchor_records[index]
        anchor_writer_id = int(record['writer_id'])
        anchor_path = str(record['anchor_path'])
        rng = random.Random(self.seed + (self.current_epoch * 100003) + index)

        positive_candidates = [path for path in self.writer_to_genuine_paths[anchor_writer_id] if path != anchor_path]
        positive_path = rng.choice(positive_candidates)
        negative_intra_path = rng.choice(self.writer_to_forgery_paths[anchor_writer_id])

        inter_writer_candidates = [writer_id for writer_id in self.writer_ids if writer_id != anchor_writer_id]
        inter_writer_id = rng.choice(inter_writer_candidates)
        negative_inter_path = rng.choice(self.writer_to_genuine_paths[inter_writer_id])

        return load_triplet_sample(
            anchor_path=anchor_path,
            positive_path=positive_path,
            negative_intra_path=negative_intra_path,
            negative_inter_path=negative_inter_path,
            anchor_writer_id=anchor_writer_id,
            inter_writer_id=inter_writer_id,
            target_size=self.target_size,
        )

def build_fixed_triplet_records(
    inventory_df: pd.DataFrame,
    tuples_per_anchor: int = 4,
    seed: int = 314,
) -> List[Dict[str, object]]:
    writer_to_genuine_paths, writer_to_forgery_paths = build_writer_path_pools(inventory_df)
    writer_ids = sorted(writer_to_genuine_paths.keys())
    records = []

    for writer_id in writer_ids:
        inter_writer_candidates = [candidate for candidate in writer_ids if candidate != writer_id]
        for anchor_index, anchor_path in enumerate(writer_to_genuine_paths[writer_id]):
            rng = random.Random(seed + (writer_id * 1009) + anchor_index)
            positive_candidates = [path for path in writer_to_genuine_paths[writer_id] if path != anchor_path]

            for tuple_index in range(tuples_per_anchor):
                positive_path = positive_candidates[tuple_index % len(positive_candidates)]
                negative_intra_path = writer_to_forgery_paths[writer_id][tuple_index % len(writer_to_forgery_paths[writer_id])]
                inter_writer_id = inter_writer_candidates[rng.randrange(len(inter_writer_candidates))]
                negative_inter_candidates = writer_to_genuine_paths[inter_writer_id]
                negative_inter_path = negative_inter_candidates[tuple_index % len(negative_inter_candidates)]

                records.append({
                    'anchor_writer_id': int(writer_id),
                    'inter_writer_id': int(inter_writer_id),
                    'anchor_path': str(anchor_path),
                    'positive_path': str(positive_path),
                    'negative_intra_path': str(negative_intra_path),
                    'negative_inter_path': str(negative_inter_path),
                })

    return records

class FixedDualTripletDataset(Dataset):
    def __init__(self, triplet_records: List[Dict[str, object]], target_size: Tuple[int, int] = (512, 512)):
        self.triplet_records = list(triplet_records)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.triplet_records)

    def __getitem__(self, index: int):
        record = self.triplet_records[index]
        return load_triplet_sample(
            anchor_path=record['anchor_path'],
            positive_path=record['positive_path'],
            negative_intra_path=record['negative_intra_path'],
            negative_inter_path=record['negative_inter_path'],
            anchor_writer_id=int(record['anchor_writer_id']),
            inter_writer_id=int(record['inter_writer_id']),
            target_size=self.target_size,
        )
