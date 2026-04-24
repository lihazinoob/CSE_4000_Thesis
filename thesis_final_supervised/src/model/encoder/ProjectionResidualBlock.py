import torch

from thesis_final_supervised.src.utils.build_norm_layer import build_norm_layer


class ProjectionResidualBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=2,
            norm_type='batch',
            activation_layer=torch.nn.ReLU
    ):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels = out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.norm1 = build_norm_layer(
            num_features=out_channels,
            norm_type=norm_type
        )

        self.act1 = activation_layer(
            inplace=True
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels= out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.norm2 = build_norm_layer(
            num_features=out_channels,
            norm_type=norm_type
        )

        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels= in_channels,
                out_channels = out_channels,
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
