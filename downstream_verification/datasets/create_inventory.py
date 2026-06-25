from pathlib import Path
from typing import List

import pandas as pd


def build_downstream_inventory(writer_ids: List[str], data_dir: Path) -> pd.DataFrame:
    rows = []
    for writer_id in writer_ids:
        writer_dir = data_dir / writer_id
        if not writer_dir.exists():
            print(f'Warning: writer directory not found: {writer_dir}')
            continue
        for image_file in sorted(writer_dir.glob('*.tif')):
            parts = image_file.stem.split('-')
            if len(parts) < 4:
                print(f'Warning: unexpected filename format, skipping: {image_file.name}')
                continue
            signature_type = parts[3]  # 'G' or 'F'
            rows.append({
                'writer_id': int(writer_id),
                'signature_type': signature_type,
                'image_path': str(image_file),
            })
    inventory_df = pd.DataFrame(rows)
    if inventory_df.empty:
        raise ValueError(f'No signature images found for downstream writers in: {data_dir}')
    return inventory_df