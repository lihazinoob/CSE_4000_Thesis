from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_pretraining.models.Encoder import Encoder


class DownstreamSignatureEmbeddingModel(nn.Module):
    def __init__(
            self,
            pretrained_ssl_checkpoint_path: str,
            trainable_encoder_stages: Tuple[str, ...],
            projector_hidden_dim: int,
            embedding_dim: int,
            norm_type: str,

    ):
        super().__init__()
        self.encoder = Encoder(
            norm_type=norm_type,
        )

        self.projector = nn.Sequential(
            nn.Linear(
                in_features=self.encoder.out_channels,
                out_features=projector_hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                in_features=projector_hidden_dim,
                out_features=embedding_dim
            )
        )

        self._load_pretrained_encoder(pretrained_ssl_checkpoint_path)
        self._configure_partial_finetuning(trainable_encoder_stages)

    def _load_pretrained_encoder(self, checkpoint_path: str) -> None:
        encoder_state_dict = torch.load(checkpoint_path, map_location='cpu')
        missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=True)
        print(f'Loaded pretrained SSL encoder from: {checkpoint_path}')
        print(f'Missing keys:    {len(missing_keys)}')
        print(f'Unexpected keys: {len(unexpected_keys)}')

    def _configure_partial_finetuning(self, trainable_encoder_stages: Tuple[str, ...]) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for stage_name in trainable_encoder_stages:
            if not hasattr(self.encoder, stage_name):
                raise ValueError(f'Unknown encoder stage: {stage_name}')
            for parameter in getattr(self.encoder, stage_name).parameters():
                parameter.requires_grad = True

        frozen_encoder_params = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        trainable_encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        trainable_projector_params = sum(p.numel() for p in self.projector.parameters())

        print(f'Trainable encoder stages: {trainable_encoder_stages if trainable_encoder_stages else "none (fully frozen)"}')
        print(f'Frozen encoder params:    {frozen_encoder_params:,}')
        print(f'Trainable encoder params: {trainable_encoder_params:,}')
        print(f'Trainable projector params: {trainable_projector_params:,}')

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        features = self.encoder(
            image_tensor,
            pool=True
        )
        projected = self.projector(features)
        return F.normalize(projected, p=2, dim=1)