import torch.nn as nn

def build_norm_layer(
        num_features: int,
        norm_type: str = 'batch'
):
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_features)
    if norm_type == 'instance':
        return nn.InstanceNorm2d(num_features, affine=True)
    raise ValueError(f'Unsupported norm_type: {norm_type}')