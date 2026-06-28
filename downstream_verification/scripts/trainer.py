import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from downstream_verification.datasets.DynamicDualTripletDataset import DynamicDualTripletDataset
from downstream_verification.datasets.FixedDualTripletDataset import (
    FixedDualTripletDataset,
    build_fixed_triplet_records,
)
from downstream_verification.datasets.create_inventory import build_downstream_inventory
from downstream_verification.loss.dual_triplet_loss import DualTripletLoss
from downstream_verification.models.Embedding_Model import DownstreamSignatureEmbeddingModel
from downstream_verification.utils.plot_history import plot_training_history
from downstream_verification.utils.train_validation import train_and_validate_model
from utils.dataloader_settings import resolve_dataloader_settings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPLIT_JSON_PATH = _PROJECT_ROOT / 'artifacts' / 'json' / 'bh_sig_hindi_writer_split.json'
_DATA_DIR = _PROJECT_ROOT / 'data' / 'all' / 'BHSig260_Hindi'
_ENCODER_CHECKPOINT_PATH = (
    _PROJECT_ROOT / 'artifacts' / 'saved_models' / 'encoder_model' / 'encoder_epoch50.pt'
)

# ── Dataset ───────────────────────────────────────────────────────────────────
TARGET_IMAGE_SIZE = (256, 256)
TRAIN_DATASET_SEED = 42
VAL_TUPLES_PER_ANCHOR = 4
VAL_SEED = 314
TEST_TUPLES_PER_ANCHOR = 4
TEST_SEED = 628

# ── Model ─────────────────────────────────────────────────────────────────────
TRAINABLE_ENCODER_STAGES: Tuple[str, ...] = ()   # encoder fully frozen
PROJECTOR_HIDDEN_DIM = 256
EMBEDDING_DIM = 256
NORM_TYPE = 'batch'

# ── Loss ──────────────────────────────────────────────────────────────────────
INTRA_MARGIN = 0.2
INTER_MARGIN = 0.2
INTER_LOSS_WEIGHT = 1.0

# ── Optimiser ─────────────────────────────────────────────────────────────────
BASE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20


def _get_optimizer_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    decay_params = []
    no_decay_params = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or param_name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


def train():
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_of_gpus = torch.cuda.device_count()
    use_data_parallel = number_of_gpus > 1
    print(f'Device: {device} | GPUs: {number_of_gpus}')

    # ── Batch size and LR scale with GPU count ────────────────────────────────
    batch_size = 8 * max(number_of_gpus, 1)
    learning_rate = BASE_LEARNING_RATE * max(number_of_gpus, 1)

    number_of_workers, pin_memory = resolve_dataloader_settings(
        number_of_gpus=number_of_gpus
    )

    # ── Writer splits ─────────────────────────────────────────────────────────
    with open(_SPLIT_JSON_PATH, 'r') as f:
        splits = json.load(f)

    train_writer_ids: List[str] = splits['downstream_verification_training']
    val_writer_ids: List[str] = splits['downstream_verification_validation']
    test_writer_ids: List[str] = splits['downstream_verification_testing']
    print(
        f'Writers — train: {len(train_writer_ids)} | '
        f'val: {len(val_writer_ids)} | '
        f'test: {len(test_writer_ids)}'
    )

    # ── Inventories ───────────────────────────────────────────────────────────
    train_inventory_df = build_downstream_inventory(train_writer_ids, _DATA_DIR)
    val_inventory_df = build_downstream_inventory(val_writer_ids, _DATA_DIR)
    test_inventory_df = build_downstream_inventory(test_writer_ids, _DATA_DIR)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = DynamicDualTripletDataset(
        inventory_df=train_inventory_df,
        target_size=TARGET_IMAGE_SIZE,
        seed=TRAIN_DATASET_SEED,
    )

    val_dataset = FixedDualTripletDataset(
        triplet_records=build_fixed_triplet_records(
            inventory_df=val_inventory_df,
            tuples_per_anchor=VAL_TUPLES_PER_ANCHOR,
            seed=VAL_SEED,
        ),
        target_size=TARGET_IMAGE_SIZE,
    )

    test_dataset = FixedDualTripletDataset(
        triplet_records=build_fixed_triplet_records(
            inventory_df=test_inventory_df,
            tuples_per_anchor=TEST_TUPLES_PER_ANCHOR,
            seed=TEST_SEED,
        ),
        target_size=TARGET_IMAGE_SIZE,
    )

    print(
        f'Dataset sizes — train: {len(train_dataset)} | '
        f'val: {len(val_dataset)} | '
        f'test: {len(test_dataset)}'
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    print(
        f'Batches per epoch — train: {len(train_loader)} | '
        f'val: {len(val_loader)} | '
        f'test: {len(test_loader)}'
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DownstreamSignatureEmbeddingModel(
        pretrained_ssl_checkpoint_path=str(_ENCODER_CHECKPOINT_PATH),
        trainable_encoder_stages=TRAINABLE_ENCODER_STAGES,
        projector_hidden_dim=PROJECTOR_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        norm_type=NORM_TYPE,
    ).to(device)

    if use_data_parallel:
        model = nn.DataParallel(model)
        print(f'Wrapped model with DataParallel across {number_of_gpus} GPUs.')

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_function = DualTripletLoss(
        intra_margin=INTRA_MARGIN,
        inter_margin=INTER_MARGIN,
        inter_loss_weight=INTER_LOSS_WEIGHT,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        _get_optimizer_param_groups(model, weight_decay=WEIGHT_DECAY),
        lr=learning_rate,
    )

    print(f'Batch size: {batch_size} | Learning rate: {learning_rate} | Epochs: {NUM_EPOCHS}')

    # ── Training ──────────────────────────────────────────────────────────────
    train_history_df = train_and_validate_model(
        model=model,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        val_inventory_df=val_inventory_df,
        target_image_size=TARGET_IMAGE_SIZE,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
    )

    print('Training complete.')
    print(train_history_df)

    plot_training_history(train_history_df)


if __name__ == "__main__":
    train()
