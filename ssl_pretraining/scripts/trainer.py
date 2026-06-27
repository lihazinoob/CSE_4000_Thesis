import torch
from torch.utils.data import DataLoader

from ssl_pretraining.datasets.create_mendeley_inventory import create_mendeley_inventory
from ssl_pretraining.datasets.load_mendeley_inventory import MendeleyInventoryLoader
from ssl_pretraining.datasets.SignatureSSLDataset import SignatureSSLDataset
from ssl_pretraining.models.sslnetwork import SelfSupervisedNetwork
from ssl_pretraining.utils.training.train import train_self_supervised_model
from utils.transformation import transform_preprocessed_image
from utils.dataloader_settings import resolve_dataloader_settings

# Fixed seed — identical across all 6 weight runs so that initialization
# noise cannot confound the ablation comparison.
TRAINING_SEED = 42


def train():

    # ------------------------------------------------------------------ #
    # Loss weight under test — change this value for each ablation run.   #
    # Sweep: {1, 5, 10, 15, 20, 30}                                       #
    # ------------------------------------------------------------------ #
    foreground_weight = 1.0
    background_weight = 1.0

    # run_name is derived automatically — do not edit this line.
    # It namespaces every output file (CSVs, checkpoints, SSIM maps)
    # so that runs with different weights never overwrite each other.
    run_name = f'fg_weight_{foreground_weight:g}'

    # ------------------------------------------------------------------ #
    # Device setup                                                         #
    # ------------------------------------------------------------------ #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_of_gpus   = torch.cuda.device_count()
    use_data_parallel = number_of_gpus > 1

    base_batch_size = 4
    batch_size      = base_batch_size * max(number_of_gpus, 1)

    base_learning_rate = 0.01
    learning_rate      = base_learning_rate * max(1, number_of_gpus)

    # ------------------------------------------------------------------ #
    # Build Mendeley inventory JSON                                        #
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 50)
    print('Step 1 — Building Mendeley writer split inventory')
    create_mendeley_inventory()

    # ------------------------------------------------------------------ #
    # Load train / eval inventory DataFrames                              #
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 50)
    print('Step 2 — Loading Mendeley inventory')
    loader = MendeleyInventoryLoader()
    train_df, eval_df = loader.load_inventory()

    # ------------------------------------------------------------------ #
    # Datasets                                                            #
    # ------------------------------------------------------------------ #
    transform = transform_preprocessed_image()

    train_dataset = SignatureSSLDataset(
        inventory_dataframe=train_df,
        patch_size=16,
        target_size=(256, 256),
        transform=transform,
        num_patches=None,
    )

    eval_dataset = SignatureSSLDataset(
        inventory_dataframe=eval_df,
        patch_size=16,
        target_size=(256, 256),
        transform=transform,
        num_patches=None,
    )

    # ------------------------------------------------------------------ #
    # DataLoaders                                                         #
    # ------------------------------------------------------------------ #
    number_of_workers, pin_memory = resolve_dataloader_settings(
        number_of_gpus=number_of_gpus
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=number_of_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # ------------------------------------------------------------------ #
    # Reproducible model initialisation — must happen before any          #
    # torch.nn.Module is created so that all 6 weight runs start from     #
    # exactly the same parameter values.                                  #
    # ------------------------------------------------------------------ #
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)

    self_supervised_network = SelfSupervisedNetwork(norm_type='batch')
    self_supervised_network = self_supervised_network.to(device)

    if use_data_parallel:
        self_supervised_network = torch.nn.DataParallel(self_supervised_network)

    # ------------------------------------------------------------------ #
    # Train                                                               #
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 50)
    print(f'Step 3 — Starting SSL pretraining  |  run: {run_name}')
    train_self_supervised_model(
        ssl_model=self_supervised_network,
        train_loader=train_dataloader,
        eval_loader=eval_dataloader,
        device=device,
        number_of_epochs=50,
        warmup_epochs=5,
        learning_rate=learning_rate,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        save_frequency=5,
        run_name=run_name,
    )


if __name__ == '__main__':
    train()
