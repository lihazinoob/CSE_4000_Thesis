import torch

from thesis_final_supervised.src.utils.build_norm_layer import build_norm_layer


class IdentityResidualBlock(torch.nn.Module):
    def __init__(
            self,
            channels,
            norm_type='batch',
            activation_layer=torch.nn.ReLU
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels= channels,
            out_channels = channels,
            kernel_size= 3,
            stride= 1,
            padding= 1,
            bias=False
        )

        self.norm1 = build_norm_layer(
            num_features=channels,
            norm_type=norm_type
        )

        self.act1 = activation_layer(
            inplace=True
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.norm2 = build_norm_layer(
            num_features=channels,
            norm_type=norm_type
        )

        self.out_act = activation_layer(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act1(
            self.norm1(
                self.conv1(x)
            )
        )
        out = self.norm2(
            self.conv2(out)
        )
        out = self.out_act(out + residual)
        return out
