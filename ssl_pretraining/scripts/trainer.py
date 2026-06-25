from torch.utils.data import DataLoader

from ssl_pretraining.datasets.SignatureSSLDataset import SignatureSSLDataset
from ssl_pretraining.datasets.load_bhsig_hindi_inventory import InventoryLoader
from ssl_pretraining.models.sslnetwork import SelfSupervisedNetwork
from ssl_pretraining.utils.training.train import train_self_supervised_model
from utils.transformation import transform_preprocessed_image
from utils.dataloader_settings import resolve_dataloader_settings

import torch

def train():

    # get the device information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the number of gpus available in the device
    number_of_gpus = torch.cuda.device_count()
    use_data_parallel= number_of_gpus > 1

    # Get the batch size according to the number of gpus
    base_batch_size = 8
    batch_size = base_batch_size * max(number_of_gpus, 1)

    # Get the learning rate according to the number of gpus
    base_learning_rate = 0.01
    learning_rate = base_learning_rate * max(1, number_of_gpus)

    # Define the foreground_weight and background_weight for the loss function
    foreground_weight = 5.0
    background_weight = 1.0


    print("\n" + "="*50)
    inventory = InventoryLoader()
    inventory_dataframe = inventory.load_inventory()

    # Call the SignatureSSLDataset with the loaded inventory
    print("\n" + "="*50)

    self_supervised_train_dataset = SignatureSSLDataset(
        inventory_dataframe= inventory_dataframe,
        patch_size= 32,
        target_size= (256,256),
        transform=transform_preprocessed_image(),
        num_patches=16,
    )

    #     Resolve the dataloader settings
    number_of_workers,pin_memory = resolve_dataloader_settings(
        number_of_gpus= number_of_gpus
    )

    # Create the pytorch dataloader for batch processing
    self_supervised_train_dataloader = DataLoader(
        dataset= self_supervised_train_dataset,
        batch_size= batch_size,
        shuffle= True,
        num_workers= number_of_workers,
        pin_memory= pin_memory,
        drop_last= False
    )

    # Create an instance of the Self_Supervised_Network
    self_supervised_network = SelfSupervisedNetwork(
        norm_type='batch'
    )
    # pass this model to cuda
    self_supervised_network = self_supervised_network.to(device)

    # do a sanity check
    # sanity_check(
    #     self_supervised_train_loader=self_supervised_train_dataloader,
    #     use_data_parallel=use_data_parallel,
    #     self_supervised_model=self_supervised_network,
    #     device=device,
    # )

    # Call the train function
    train_self_supervised_model(
        ssl_model=self_supervised_network,
        train_loader=self_supervised_train_dataloader,
        device=device,
        number_of_epochs= 50,
        warmup_epochs=5,
        learning_rate=learning_rate,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        save_frequency=5
    )


if __name__ == "__main__":
    train()