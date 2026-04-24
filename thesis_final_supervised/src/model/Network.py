import torch

from thesis_final_supervised.src.model.encoder.Encoder import LightWeightEncoder


class SignatureEmbeddingNetwork(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int = 256,
            norm_type: str = 'batch'
    ):
        super().__init__()
        
        self.encoder = LightWeightEncoder(
            norm_type=norm_type,
        )

    def forward(self, image_tensor : torch.Tensor) -> torch.Tensor:
        signature_embedding_from_encoder_backbone = self.encoder(image_tensor)
        return signature_embedding_from_encoder_backbone