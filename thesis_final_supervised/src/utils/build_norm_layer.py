import torch

def build_norm_layer(
        num_features,
        norm_type = 'batch'
):
    if norm_type == 'batch':
        norm_layer = torch.nn.BatchNorm2d(
            num_features=num_features,
        )
    if norm_type == 'instance':
        norm_layer = torch.nn.InstanceNorm2d(
            num_features=num_features,
            affine=True
        )
