from pathlib import Path
from typing import List

import pandas as pd


def build_cedar_inventory(writer_ids: List[str], data_dir: Path) -> pd.DataFrame:
    """
    Build a signature inventory DataFrame for the given CEDAR writer IDs.

    CEDAR file naming convention (PNG):
      original_{writer}_{n}.png    → genuine   (signature_type = 'G')
      forgeries_{writer}_{n}.png   → forgery   (signature_type = 'F')

    Each writer folder is expected to contain exactly 24 genuine and 24 forgery images.
    """
    rows = []
    for writer_id in writer_ids:
        writer_dir = data_dir / writer_id
        if not writer_dir.exists():
            print(f'Warning: writer directory not found: {writer_dir}')
            continue
        for image_file in sorted(writer_dir.glob('*.png')):
            stem = image_file.stem
            if stem.startswith('original_'):
                signature_type = 'G'
            elif stem.startswith('forgeries_'):
                signature_type = 'F'
            else:
                print(f'Warning: unrecognised filename, skipping: {image_file.name}')
                continue
            rows.append({
                'writer_id': int(writer_id),
                'signature_type': signature_type,
                'image_path': str(image_file),
            })

    inventory_df = pd.DataFrame(rows)
    if inventory_df.empty:
        raise ValueError(
            f'No signature images found for CEDAR writers {writer_ids} in: {data_dir}'
        )
    return inventory_df
