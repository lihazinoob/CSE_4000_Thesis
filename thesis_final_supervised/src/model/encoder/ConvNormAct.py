import torch

from thesis_final_supervised.src.utils.build_norm_layer import build_norm_layer


class ConvNormAct(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = None,
            norm_type = 'batch',
            activation_layer = torch.nn.ReLU
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            build_norm_layer(
                num_features = out_channels,
                norm_type = norm_type
            ),
            activation_layer(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)