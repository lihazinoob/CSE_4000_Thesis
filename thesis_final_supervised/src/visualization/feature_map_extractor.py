import torch


class FeatureMapExtractor:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.feature_map = None
        self.handle = self.target_layer.register_forward_hook(self._save_feature_map)

    def _save_feature_map(self, module, inputs, output):
        self.feature_map = output.detach()

    def extract(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.feature_map = None
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.feature_map is None:
            raise RuntimeError("Feature map was not captured. Check the target layer.")

        return self.feature_map

    def close(self):
        self.handle.remove()
