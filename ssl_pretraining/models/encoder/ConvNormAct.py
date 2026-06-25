from torch import nn

from ssl_pretraining.utils.normalization_layer import build_norm_layer


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            norm_type='batch',
            activation_layer=nn.ReLU
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(

            # Define a Convolutional block
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            # Normalization Layer
            build_norm_layer(
                out_channels,
                norm_type=norm_type
            ),

            # Apply ReLU activation
            activation_layer(inplace=True),
        )

    def forward(self, x):
        return self.block(x)