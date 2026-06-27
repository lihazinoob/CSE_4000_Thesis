from typing import Tuple, Dict, Optional

import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

from ssl_pretraining.utils.patch_extraction import extract_patches
from utils.preprocessing import preprocess_image


class SignatureSSLDataset(Dataset):
    def  __init__(
            self,
            inventory_dataframe: pd.DataFrame,
            patch_size: int,
            target_size: Tuple[int, int],
            transform,
            num_patches: Optional[int] = None,
    ):
        super().__init__()
        self.inventory_dataframe = inventory_dataframe
        self.patch_size = patch_size
        self.target_size = target_size
        self.transform = transform
        self.num_patches = num_patches

        if self.inventory_dataframe.empty:
            raise ValueError('SignatureSSLDataset received an empty inventory.')

    def __len__(self) -> int:
        return len(self.inventory_dataframe)

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, object]:
        row = self.inventory_dataframe.iloc[index]
        image_path = str(row['image_path'])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')

        signature_image = preprocess_image(
            image = image,
            target_size = self.target_size,
        )

        signature_image_tensor = self.transform(signature_image)

        patches = extract_patches(
            image = signature_image,
            patch_size = self.patch_size,
        )

        expected_patch_count = (self.target_size[0] // self.patch_size) * (self.target_size[1] // self.patch_size)

        if len(patches) != expected_patch_count:
            raise AssertionError(
                f'Expected {expected_patch_count} patches, found {len(patches)} for {image_path}'
            )

        if self.num_patches is not None:
            if self.num_patches > len(patches):
                raise ValueError(
                    f'num_patches ({self.num_patches}) exceeds available patches ({len(patches)})'
                )
            patches = random.sample(patches, self.num_patches)

        patches_tensor = torch.stack(
            [
                self.transform(patch) for patch in patches
            ],
            dim = 0
        )

        return {
            'image': signature_image_tensor,
            'patches': patches_tensor,
            'signature_type': str(row['signature_type']),
            'writer_id': str(row['writer_id']),
            'image_path': image_path,
        }
