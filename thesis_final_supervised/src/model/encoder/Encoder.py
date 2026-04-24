import torch

from thesis_final_supervised.src.model.encoder.ResidualStage import ResidualStage
from thesis_final_supervised.src.model.encoder.SignatureStem import SignatureStem


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

