import torch
from thesis_final_supervised.src.model.encoder.Encoder import LightWeightEncoder


def load_pretrained_model(
        device,
        model_path: str,
):
    # Instantiate the model
    encoder = LightWeightEncoder()

    # Load the entire saved dictionary
    full_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract the state dictionary which is nested under the 'encoder_state_dict' key
    encoder_state_dict = full_checkpoint['encoder_state_dict']

    # Load the extracted state dictionary into the model
    encoder.load_state_dict(encoder_state_dict)

    print("Pre-trained encoder model loaded successfully.")
    return encoder