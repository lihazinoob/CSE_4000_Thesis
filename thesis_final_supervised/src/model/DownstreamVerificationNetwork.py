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


class SignatureVerificationNetwork(torch.nn.Module):
    def __init__(
            self,
            projector_hidden_dim: int = 256,
            embedding_dim: int = 256,
            norm_type: str = 'batch',
    ):
        super().__init__()
        self.embedding_network = SignatureEmbeddingNetwork(
            norm_type=norm_type
        )

        # Now define the projector portion
        self.projector_head = torch.nn.Sequential(
            torch.nn.Linear(
                self.embedding_network.encoder.out_channels,
                projector_hidden_dim
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            torch.nn.Linear(
                projector_hidden_dim,
                embedding_dim
            ),
        )

    def forward(
        self,
        image_tensor : torch.Tensor
    ) -> torch.Tensor:
        feature = self.embedding_network(image_tensor)
        projected_feature = self.projector_head(feature)
        return torch.nn.functional.normalize(projected_feature, p=2, dim=1)