import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def parse_bhsig_filename(file_name: str) -> Dict[str, str]:
    parts = Path(file_name).stem.split('-')
    if len(parts) < 5:
        raise ValueError(f'Unexpected BHSig file name format: {file_name}')
    return {
        'script': parts[0],
        'subset': parts[1],
        'writer_token': parts[2],
        'label_token': parts[3],
        'sample_token': parts[4],
    }


def build_bhsig_inventory(dataset_root: str) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    rows = []
    for writer_dir in sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda path: int(path.name)):
        writer_id = int(writer_dir.name)
        for image_path in sorted(writer_dir.iterdir()):
            if image_path.suffix.lower() not in {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}:
                continue
            parsed = parse_bhsig_filename(image_path.name)
            signature_type = 'genuine' if parsed['label_token'].upper() == 'G' else 'forgery'
            rows.append({
                'writer_id': writer_id,
                'signature_type': signature_type,
                'label': 1 if signature_type == 'genuine' else 0,
                'sample_index': int(parsed['sample_token']),
                'image_path': str(image_path),
                'file_name': image_path.name,
            })

    inventory_df = pd.DataFrame(rows).sort_values(
        ['writer_id', 'signature_type', 'sample_index', 'file_name']
    ).reset_index(drop=True)

    if inventory_df.empty:
        raise ValueError('The dataset inventory is empty.')
    return inventory_df


def split_inventory_from_summary(inventory_df: pd.DataFrame, split_summary: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_writers = set(int(writer_id) for writer_id in split_summary['train_writers'])
    val_writers = set(int(writer_id) for writer_id in split_summary['val_writers'])
    test_writers = set(int(writer_id) for writer_id in split_summary['test_writers'])

    train_df = inventory_df[inventory_df['writer_id'].isin(train_writers)].reset_index(drop=True)
    val_df = inventory_df[inventory_df['writer_id'].isin(val_writers)].reset_index(drop=True)
    test_df = inventory_df[inventory_df['writer_id'].isin(test_writers)].reset_index(drop=True)
    return train_df, val_df, test_df


def verify_writer_disjoint_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, split_summary: Dict[str, object]) -> None:
    train_writers = set(train_df['writer_id'].unique().tolist())
    val_writers = set(val_df['writer_id'].unique().tolist())
    test_writers = set(test_df['writer_id'].unique().tolist())

    assert train_writers.isdisjoint(val_writers), 'Train and validation writers overlap.'
    assert train_writers.isdisjoint(test_writers), 'Train and test writers overlap.'
    assert val_writers.isdisjoint(test_writers), 'Validation and test writers overlap.'

    assert train_writers == set(int(writer_id) for writer_id in split_summary['train_writers'])
    assert val_writers == set(int(writer_id) for writer_id in split_summary['val_writers'])
    assert test_writers == set(int(writer_id) for writer_id in split_summary['test_writers'])

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        writer_counts = split_df.groupby(['writer_id', 'signature_type']).size().unstack(fill_value=0)
        assert writer_counts['genuine'].min() >= 2, f'{split_name} split has a writer with fewer than two genuine samples.'
        assert writer_counts['forgery'].min() >= 1, f'{split_name} split has a writer without forgery samples.'

    print('Writer-disjoint split verification passed.')


def summarize_split(name: str, split_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = split_df.groupby(['writer_id', 'signature_type']).size().unstack(fill_value=0)
    print(f'--- {name.upper()} ---')
    print(f"Writers: {split_df['writer_id'].nunique()}")
    print(f'Samples: {len(split_df)}')
    print(f"Genuine: {(split_df['signature_type'] == 'genuine').sum()}")
    print(f"Forgery: {(split_df['signature_type'] == 'forgery').sum()}")
    return summary_df
