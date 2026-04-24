import torch

def build_norm_layer(num_features, norm_type='batch'):
    if norm_type == 'batch':
        return torch.nn.BatchNorm2d(
            num_features
        )
    if norm_type == 'instance':
        return torch.nn.InstanceNorm2d(
            num_features,
            affine=True
        )
    raise ValueError(f'Unsupported norm_type: {norm_type}')