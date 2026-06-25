import torch.nn as nn

from ssl_pretraining.models.encoder.IdentityResidualBlock import IdentityResidualBlock
from ssl_pretraining.models.encoder.ProjectionResidualBlock import ProjectionResidualBlock


class TransitionResidualStage(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=2,
            num_identity_blocks=1,
            norm_type='batch',
            activation_layer=nn.ReLU):
        super().__init__()

        blocks = [
            ProjectionResidualBlock(
                in_channels = in_channels,
                out_channels = out_channels,
                stride=stride,
                norm_type=norm_type,
                activation_layer=activation_layer
            )
        ]

        for _ in range(num_identity_blocks):
            blocks.append(
                IdentityResidualBlock(
                in_channels=out_channels,
                norm_type=norm_type,
                activation_layer=activation_layer
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, x):
        return self.blocks(x)