import torch

from thesis_final_supervised.src.model.encoder.IdentityResidualBlock import IdentityResidualBlock


class ResidualStage(torch.nn.Module):
    def __init__(
            self,
            channels,
            num_blocks=2,
            norm_type='batch',
            activation_layer=torch.nn.ReLU
    ):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            *[
                IdentityResidualBlock(
                    channels=channels,
                    norm_type=norm_type,
                    activation_layer=activation_layer
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_channels = channels

    def forward(self, x):
        return self.blocks(x)