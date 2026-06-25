import torch.nn as nn
import torch
from ssl_pretraining.models.Encoder import Encoder
from ssl_pretraining.models.attention.AttentionModule import Attention2DModule
from ssl_pretraining.models.decoder.Decoder import UNetDecoder


class SelfSupervisedNetwork(nn.Module):

    def __init__(self, foreground_threshold=0.01, norm_type='batch'):
        super().__init__()

        # Define the Encoder
        self.encoder = Encoder(
            norm_type=norm_type
        )

        # Define the Attention2D Module
        self.attn_module = Attention2DModule(
            feature_dim=self.encoder.out_channels
        )

        # Define the UNet Decoder
        self.decoder = UNetDecoder(
            latent_dim=self.encoder.out_channels,
            out_channels=1,
            norm_type=norm_type
        )

        self.patch_score = nn.Linear(self.encoder.out_channels, 1)

        self.foreground_threshold = foreground_threshold

    def forward(self, image, patches):

        # get the batch size
        batch_size = image.shape[0]

        # get the patch shape
        number_of_patches = patches.shape[1]

        # Pass the image into the encoder to get the vectorized form of the image
        image_feature_vector = self.encoder(image, pool=True)

        patches_flat = patches.view(-1, patches.shape[2], patches.shape[3], patches.shape[4])

        # Pass the patches into the encoder to get the feature maps for each patch
        patch_feature_maps = self.encoder(patches_flat, pool=False)

        feat_h, feat_w = patch_feature_maps.shape[2], patch_feature_maps.shape[3]
        patch_feature_maps = patch_feature_maps.view(batch_size, number_of_patches, self.encoder.out_channels, feat_h, feat_w)

        patch_attn_feats = []
        patch_attn_maps = []
        for patch_index in range(number_of_patches):
            patch_fmap = patch_feature_maps[:, patch_index, :, :, :]
            attn_map, attn_feature = self.attn_module(image_feature_vector, patch_fmap)
            patch_attn_feats.append(attn_feature.unsqueeze(1))
            patch_attn_maps.append(attn_map)

        patch_attn_feats = torch.cat(patch_attn_feats, dim=1)
        foreground_ratio = (patches > 0).float().mean(dim=(2, 3, 4))
        patch_scores = (foreground_ratio > self.foreground_threshold).float()
        foreground_prior = patch_scores / (patch_scores.sum(dim=1, keepdim=True) + 1e-8)
        patch_weights = foreground_prior
        patch_attn = torch.sum(patch_weights.unsqueeze(-1) * patch_attn_feats, dim=1)
        reconstructed_image = self.decoder(patch_attn)
        return reconstructed_image, patch_attn_maps, foreground_ratio, foreground_prior, patch_scores, patch_weights
