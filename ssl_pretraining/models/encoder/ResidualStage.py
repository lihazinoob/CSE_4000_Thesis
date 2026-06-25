
import torch.nn as nn

from ssl_pretraining.models.encoder.IdentityResidualBlock import IdentityResidualBlock


# Creates two IdentityResidualBlock
class ResidualStage(nn.Module):
    def __init__(
            self,
            channels,
            num_blocks=2,
            norm_type='batch',
            activation_layer=nn.ReLU
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            IdentityResidualBlock(
                in_channels=channels,
                norm_type=norm_type,
                activation_layer=activation_layer
            )
            for _ in range(num_blocks)
        ])
        self.out_channels = channels

    def forward(self, x):
        return self.blocks(x)