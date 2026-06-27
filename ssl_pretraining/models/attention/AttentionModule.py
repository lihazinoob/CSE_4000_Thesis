import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention2DModule(nn.Module):

    def __init__(
            self,
            feature_dim=256,
            attention_dim=128,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.attention_dim = attention_dim

        # W_f: 1x1 conv projects patch feature maps into the attention subspace
        self.W_f = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=attention_dim,
            kernel_size=1,
            bias=False,
        )

        # W_h: linear projects image feature vector into the attention subspace
        self.W_h = nn.Linear(feature_dim, attention_dim, bias=False)

        # W_att: computes a scalar attention score per spatial location from M
        self.W_att = nn.Linear(attention_dim, 1, bias=False)

    def forward(
            self,
            image_feature_vector,   # (N, feature_dim)
            patch_feature_map,      # (N, feature_dim, H, W)
    ):
        # Project patch feature maps via W_f (1x1 conv), then flatten spatial dims
        C_projected = self.W_f(patch_feature_map)                      # (N, attention_dim, H, W)
        C_projected_flat = C_projected.flatten(2).permute(0, 2, 1)    # (N, H*W, attention_dim)

        # Project image feature vector via W_h, expand for broadcasting over H*W
        Z_projected = self.W_h(image_feature_vector).unsqueeze(1)     # (N, 1, attention_dim)

        # M = tanh(W_f(C) + W_h(Z))  — broadcast Z over spatial positions
        M = torch.tanh(C_projected_flat + Z_projected)                # (N, H*W, attention_dim)

        # Attention weights: softmax over spatial positions
        attn_scores = self.W_att(M)                                    # (N, H*W, 1)
        attn_wts = F.softmax(attn_scores, dim=1).permute(0, 2, 1)    # (N, 1, H*W)

        # Context vector: attend over the ORIGINAL (unprojected) C features
        C_flat = patch_feature_map.flatten(2).permute(0, 2, 1)        # (N, H*W, feature_dim)
        attn_output = torch.bmm(attn_wts, C_flat).squeeze(1)          # (N, feature_dim)

        return attn_wts, attn_output
