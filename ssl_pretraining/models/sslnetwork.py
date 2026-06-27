import torch.nn as nn
import torch
from ssl_pretraining.models.Encoder import Encoder
from ssl_pretraining.models.attention.AttentionModule import Attention2DModule
from ssl_pretraining.models.decoder.Decoder import UNetDecoder


class SelfSupervisedNetwork(nn.Module):

    def __init__(self, norm_type='batch'):
        super().__init__()

        self.encoder = Encoder(norm_type=norm_type)

        self.attn_module = Attention2DModule(
            feature_dim=self.encoder.out_channels,
            attention_dim=self.encoder.out_channels // 2,
        )

        self.decoder = UNetDecoder(
            latent_dim=self.encoder.out_channels,
            out_channels=1,
            norm_type=norm_type,
        )

    def forward(self, image, patches):
        batch_size = image.shape[0]
        number_of_patches = patches.shape[1]

        # Encode full image with global average pooling → (N, feature_dim)
        image_feature_vector = self.encoder(image, pool=True)

        # Encode all patches without pooling
        patches_flat = patches.view(-1, patches.shape[2], patches.shape[3], patches.shape[4])
        patch_feature_maps = self.encoder(patches_flat, pool=False)

        feat_h, feat_w = patch_feature_maps.shape[2], patch_feature_maps.shape[3]
        patch_feature_maps = patch_feature_maps.view(
            batch_size, number_of_patches, self.encoder.out_channels, feat_h, feat_w
        )  # (N, P, C, H, W)

        # Compute attention-enriched context vector for each patch
        patch_attn_feats = []
        patch_attn_maps = []
        for patch_index in range(number_of_patches):
            patch_fmap = patch_feature_maps[:, patch_index, :, :, :]   # (N, C, H, W)
            attn_map, attn_feature = self.attn_module(image_feature_vector, patch_fmap)
            patch_attn_feats.append(attn_feature.unsqueeze(1))
            patch_attn_maps.append(attn_map)

        patch_attn_feats = torch.cat(patch_attn_feats, dim=1)  # (N, P, feature_dim)

        # Simple average pooling over all patches (following SURDS paper)
        patch_attn = patch_attn_feats.mean(dim=1)   # (N, feature_dim)

        reconstructed_image = self.decoder(patch_attn)
        return reconstructed_image, patch_attn_maps
