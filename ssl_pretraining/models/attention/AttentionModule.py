import torch.nn as nn
import torch
import torch.nn.functional as F
class Attention2DModule(nn.Module):

    def __init__(
            self,
            feature_dim=256
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Declare a Attention Layer which is basically a Convolutional Layer
        self.attention_layer = nn.Linear(
            in_features=self.feature_dim,
            out_features=1
        )

    def forward(
            self,
            image_feature_vector,
            patch_feature_map
    ):
        image_feature_vector = image_feature_vector.unsqueeze(1)

        patch_fmap_ = patch_feature_map.view(
            patch_feature_map.shape[0],
            patch_feature_map.shape[1],
            patch_feature_map.shape[2] * patch_feature_map.shape[3],
        ).permute(0, 2, 1)

        active_sum_feat = torch.tanh(
            image_feature_vector +
            patch_fmap_
        )

        attn_wts = F.softmax(self.attention_layer(active_sum_feat), dim=1).permute(0, 2, 1)

        attn_output = torch.bmm(attn_wts, patch_fmap_).squeeze(1)

        return attn_wts, attn_output
