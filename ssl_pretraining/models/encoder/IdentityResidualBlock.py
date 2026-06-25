import torch.nn as nn

from ssl_pretraining.utils.normalization_layer import build_norm_layer


class IdentityResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            norm_type='batch',
            activation_layer=nn.ReLU
    ):
        super().__init__()

        # Apply first Convolutional layer that generates same number of output channels as input channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Apply Normalization layer
        self.norm1 = build_norm_layer(
            num_features=in_channels,
            norm_type=norm_type
        )

        # Apply ReLU activation function
        self.act1 = activation_layer(
            inplace=True
        )

        # Apply second Convolutional layer that generates same number of output channels as input channels
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Apply Normalization layer
        self.norm2 = build_norm_layer(
            num_features=in_channels,
            norm_type=norm_type
        )

        # Apply ReLU activation function
        self.out_act = activation_layer(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.out_act(out + residual)
        return out

