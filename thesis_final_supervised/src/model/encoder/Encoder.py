import torch

from thesis_final_supervised.src.model.encoder.ResidualStage import ResidualStage
from thesis_final_supervised.src.model.encoder.SignatureStem import SignatureStem
from thesis_final_supervised.src.model.encoder.TransitionResidualStage import TransitionResidualStage


class LightWeightEncoder(torch.nn.Module):
    def __init__(
            self,
            in_channels=1,
            stem_channels=(32, 32),
            stage1_blocks=2,
            stage2_out_channels=64,
            stage2_identity_blocks=1,
            stage3_out_channels=128,
            stage3_identity_blocks=1,
            stage4_out_channels=256,
            stage4_identity_blocks=1,
            norm_type='batch'
    ):
        super().__init__()
        self.stem = SignatureStem(
            in_channels = in_channels,
            stem_channels = stem_channels,
            norm_type=norm_type
        )

        self.stage1 = ResidualStage(
            channels = self.stem.channels,
            num_blocks = stage1_blocks,
            norm_type=norm_type
        )
        self.stage2 = TransitionResidualStage(
            self.stage1.out_channels,
            stage2_out_channels, stride=2,
            num_identity_blocks=stage2_identity_blocks,
            norm_type=norm_type
        ),
        self.stage3 = TransitionResidualStage(
            self.stage2.out_channels,
            stage3_out_channels,
            stride=2,
            num_identity_blocks=stage3_identity_blocks,
            norm_type=norm_type
        )
        self.stage4 = TransitionResidualStage(
            self.stage3.out_channels,
            stage4_out_channels,
            stride=2,
            num_identity_blocks=stage4_identity_blocks,
            norm_type=norm_type
        )
        self.out_channels = self.stage4.out_channels


    def forward_stem(self, x):
        return self.stem(x)

    def forward_stage1(self, x):
        return self.stage1(x)

    def forward_stage2(self, x):
        return self.stage2(x)

    def forward_stage3(self, x):
        return self.stage3(x)

    def forward_stage4(self, x):
        return self.stage4(x)

    def forward(self, x, pool=True):
        x = self.forward_stage4(self.forward_stage3(self.forward_stage2(self.forward_stage1(self.forward_stem(x)))))
        if pool:
            x = torch.mean(x, dim=(2, 3))
        return x
