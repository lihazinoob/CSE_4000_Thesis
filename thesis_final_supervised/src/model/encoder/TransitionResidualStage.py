import torch

from thesis_final_supervised.src.model.encoder.IdentityResidualBlock import IdentityResidualBlock
from thesis_final_supervised.src.model.encoder.ProjectionResidualBlock import ProjectionResidualBlock


class TransitionResidualStage(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=2,
            num_identity_blocks=1,
            norm_type='batch',
            activation_layer=torch.nn.ReLU
    ):
        super().__init__()
        blocks = [
            ProjectionResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                norm_type=norm_type,
                activation_layer=activation_layer
            )
        ]
        for _ in range(num_identity_blocks):
            blocks.append(
                IdentityResidualBlock(
                    out_channels,
                    norm_type=norm_type,
                    activation_layer=activation_layer
                )
            )
        self.blocks = torch.nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, x):
        return self.blocks(x)