import torch.nn as nn

from ssl_pretraining.models.encoder.ConvNormAct import ConvNormAct


class SignatureStem(nn.Module):
    def __init__(
            self,
            in_channels=1,
            stem_channels=(32, 32),
            norm_type='batch',
            activation_layer=nn.ReLU
    ):
        super().__init__()
        blocks = []
        current_in_channels = in_channels

        for current_out_channels in stem_channels:

            blocks.append(
                ConvNormAct(
                    in_channels=current_in_channels,
                    out_channels=current_out_channels,
                    kernel_size=3,
                    stride=1,
                    norm_type=norm_type,
                    activation_layer=activation_layer,
                )
            )

            current_in_channels = current_out_channels

        self.layers = nn.Sequential(*blocks)
        self.out_channels = stem_channels[-1]

    def forward(self, x):
        return self.layers(x)
