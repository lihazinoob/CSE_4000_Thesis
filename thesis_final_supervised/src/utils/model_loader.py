import torch

from thesis_final_supervised.src.model.DownstreamVerificationNetwork import SignatureVerificationNetwork
from thesis_final_supervised.src.model.encoder.Encoder import LightWeightEncoder


def _remap_verification_state_dict_keys(state_dict):
    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            new_key = f"embedding_network.{key}"
        elif key.startswith('projector.'):
            new_key = key.replace('projector.', 'projector_head.', 1)
        else:
            new_key = key
        remapped_state_dict[new_key] = value
    return remapped_state_dict


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

def load_verification_model(
        device,
        model_path: str,
):
    # Instantiate the model
    verification_model = SignatureVerificationNetwork()

    # Load the entire saved dictionary
    full_checkpoint = torch.load(model_path, map_location=device)

    # Downstream verification checkpoints store the full model weights.
    if isinstance(full_checkpoint, dict) and 'model_state_dict' in full_checkpoint:
        verification_model_state_dict = full_checkpoint['model_state_dict']
    else:
        verification_model_state_dict = full_checkpoint

    verification_model_state_dict = _remap_verification_state_dict_keys(
        verification_model_state_dict
    )

    
    # Load the extracted state dictionary into the model
    verification_model.load_state_dict(verification_model_state_dict)

    print("Signature Verification model loaded successfully.")
    return verification_model
