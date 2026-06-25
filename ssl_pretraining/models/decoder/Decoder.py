import torch.nn as nn

from ssl_pretraining.utils.normalization_layer import build_norm_layer


class UnetUpBlock(nn.Module):
    def __init__(
            self,
            inner_nc,
            outer_nc,
            norm_type='batch',
            activation_layer=nn.ReLU
    ):
        super().__init__()
        self.model = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=inner_nc,
                out_channels=outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            build_norm_layer(
                outer_nc,
                norm_type=norm_type
            ),

            activation_layer(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class UNetDecoder(nn.Module):
    def __init__(
            self,
            latent_dim=256,
            out_channels=1,
            norm_type='batch'
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.deconv_1 = UnetUpBlock(latent_dim, 256, norm_type=norm_type)
        self.deconv_2 = UnetUpBlock(256, 256, norm_type=norm_type)
        self.deconv_3 = UnetUpBlock(256, 256, norm_type=norm_type)
        self.deconv_4 = UnetUpBlock(256, 128, norm_type=norm_type)
        self.deconv_5 = UnetUpBlock(128, 64, norm_type=norm_type)
        self.deconv_6 = UnetUpBlock(64, 32, norm_type=norm_type)
        self.deconv_7 = UnetUpBlock(32, 16, norm_type=norm_type)
        self.deconv_8 = UnetUpBlock(16, 8, norm_type=norm_type)
        self.final_image = nn.Sequential(
            nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)
        x = self.deconv_6(x)
        x = self.deconv_7(x)
        x = self.deconv_8(x)
        x = self.final_image(x)
        return x
