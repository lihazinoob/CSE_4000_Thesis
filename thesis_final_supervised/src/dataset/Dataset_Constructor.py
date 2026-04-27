import json
from typing import Tuple, Dict
from torch.utils.data import DataLoader
from thesis_final_supervised.src.dataset.bhsig_inventory import (
    build_bhsig_inventory,
    split_inventory_from_summary,
    verify_writer_disjoint_splits,
    summarize_split
)
from thesis_final_supervised.src.dataset.DynamicDualTripletDataset import (
    DynamicDualTripletDataset,
    FixedDualTripletDataset,
    build_fixed_triplet_records
)


def create_dataset(
    split_summary_path: str,
    dataset_root: str,
    target_size: Tuple[int, int] = (512, 512),
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    
    with open(split_summary_path, 'r', encoding='utf-8') as fp:
        split_summary = json.load(fp)
    
    print("Building inventory...")
    inventory_df = build_bhsig_inventory(dataset_root)
    
    print("Splitting inventory...")
    train_df, val_df, test_df = split_inventory_from_summary(inventory_df, split_summary)
    
    verify_writer_disjoint_splits(train_df, val_df, test_df, split_summary)
    
    train_summary = summarize_split('train', train_df)
    val_summary = summarize_split('validation', val_df)
    test_summary = summarize_split('test', test_df)

    print("Building datasets...")
    train_dataset = DynamicDualTripletDataset(train_df, target_size=target_size)
    
    val_records = build_fixed_triplet_records(val_df)
    val_dataset = FixedDualTripletDataset(val_records, target_size=target_size)
    
    test_records = build_fixed_triplet_records(test_df)
    test_dataset = FixedDualTripletDataset(test_records, target_size=target_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }