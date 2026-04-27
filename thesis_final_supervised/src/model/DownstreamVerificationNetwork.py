import torch
from typing import Dict, Tuple

from thesis_final_supervised.src.model.encoder.Encoder import LightWeightEncoder


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key[7:] if key.startswith('module.') else key: value for key, value in state_dict.items()}


def extract_encoder_state_dict(ssl_checkpoint_or_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(ssl_checkpoint_or_state_dict, dict) and 'encoder_state_dict' in ssl_checkpoint_or_state_dict:
        state_dict = ssl_checkpoint_or_state_dict['encoder_state_dict']
    elif isinstance(ssl_checkpoint_or_state_dict, dict) and 'model_state_dict' in ssl_checkpoint_or_state_dict:
        state_dict = ssl_checkpoint_or_state_dict['model_state_dict']
    elif isinstance(ssl_checkpoint_or_state_dict, dict) and 'model' in ssl_checkpoint_or_state_dict:
        state_dict = ssl_checkpoint_or_state_dict['model']
    else:
        state_dict = ssl_checkpoint_or_state_dict

    state_dict = strip_module_prefix(state_dict)

    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            encoder_state_dict[key[len('encoder.'):]] = value
        elif key.startswith(('stem.', 'stage1.', 'stage2.', 'stage3.', 'stage4.', 'global_pool.')):
            encoder_state_dict[key] = value
    return encoder_state_dict


class SignatureEmbeddingNetwork(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int = 256,
            norm_type: str = 'batch'
    ):
        super().__init__()
        
        self.encoder = LightWeightEncoder(
            norm_type=norm_type,
        )

    def forward(self, image_tensor : torch.Tensor) -> torch.Tensor:
        signature_embedding_from_encoder_backbone = self.encoder(image_tensor)
        return signature_embedding_from_encoder_backbone


class SignatureVerificationNetwork(torch.nn.Module):
    def __init__(
            self,
            pretrained_ssl_checkpoint_path: str = None,
            trainable_encoder_stages: Tuple[str, ...] = (),
            projector_hidden_dim: int = 256,
            embedding_dim: int = 256,
            norm_type: str = 'batch',
    ):
        super().__init__()
        self.embedding_network = SignatureEmbeddingNetwork(
            norm_type=norm_type
        )

        # Now define the projector portion
        self.projector_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.embedding_network.encoder.out_channels,
                projector_hidden_dim
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            torch.nn.Linear(
                projector_hidden_dim,
                embedding_dim
            ),
        )
        
        self.trainable_encoder_stages = tuple(trainable_encoder_stages)

        if pretrained_ssl_checkpoint_path is not None:
            self.load_pretrained_encoder(pretrained_ssl_checkpoint_path)

        self.configure_partial_finetuning()

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        encoder_state_dict = extract_encoder_state_dict(checkpoint)
        
        missing_keys, unexpected_keys = self.embedding_network.encoder.load_state_dict(encoder_state_dict, strict=False)

        print(f'Loaded pretrained SSL encoder from: {checkpoint_path}')
        print(f'Encoder keys loaded: {len(encoder_state_dict)}')
        print(f'Missing keys after load: {len(missing_keys)}')
        print(f'Unexpected keys after load: {len(unexpected_keys)}')

    def configure_partial_finetuning(self) -> None:
        for parameter in self.embedding_network.encoder.parameters():
            parameter.requires_grad = False

        for stage_name in self.trainable_encoder_stages:
            if hasattr(self.embedding_network.encoder, stage_name):
                for parameter in getattr(self.embedding_network.encoder, stage_name).parameters():
                    parameter.requires_grad = True
            else:
                # Based on LightWeightEncoder, stages are stem, stage1, stage2, etc.
                raise ValueError(f'Unknown encoder stage: {stage_name}')

        for parameter in self.projector_head.parameters():
            parameter.requires_grad = True

        trainable_encoder_params = sum(p.numel() for p in self.embedding_network.encoder.parameters() if p.requires_grad)
        frozen_encoder_params = sum(p.numel() for p in self.embedding_network.encoder.parameters() if not p.requires_grad)
        trainable_projector_params = sum(p.numel() for p in self.projector_head.parameters() if p.requires_grad)

        print(f'Trainable encoder stages: {self.trainable_encoder_stages}')
        print(f'Trainable encoder params: {trainable_encoder_params:,}')
        print(f'Frozen encoder params: {frozen_encoder_params:,}')
        print(f'Trainable projector params: {trainable_projector_params:,}')

    def forward(
        self,
        image_tensor : torch.Tensor
    ) -> torch.Tensor:
        feature = self.embedding_network(image_tensor)
        projected_feature = self.projector_head(feature)
        return torch.nn.functional.normalize(projected_feature, p=2, dim=1)