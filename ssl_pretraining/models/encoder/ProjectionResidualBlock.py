import torch.nn as nn

from ssl_pretraining.utils.normalization_layer import build_norm_layer


class ProjectionResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=2,
            norm_type='batch',
            activation_layer=nn.ReLU
    ):
        super().__init__()

        # Apply First Convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        # Apply Normalization layer
        self.norm1 = build_norm_layer(
            num_features=out_channels,
            norm_type=norm_type
        )

        # Apply ReLU activation
        self.act1 = activation_layer(inplace=True)

        # Apply Second Convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Apply Normalization again
        self.norm2 = build_norm_layer(
            num_features=out_channels,
            norm_type=norm_type
        )

        # Apply the shortcut from the input feature to the output through a 1 X 1 convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),

            build_norm_layer(
                num_features=out_channels,
                norm_type=norm_type
            ),
        )
        self.out_act = activation_layer(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.out_act(out + residual)
        return out