import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register PyTorch hooks to capture data during forward/backward passes
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Forward pass
        model_output = self.model(input_tensor)

        # If no specific class is requested, use the one with the highest score
        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1).item()

        # 2. Clear previous gradients
        self.model.zero_grad()

        # 3. Target the specific class and trigger backpropagation
        target = model_output[0][class_idx]
        target.backward()

        # 4. Global Average Pooling of gradients (Compute neuron importance weights)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # 5. Weight the feature map activations
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # 6. Generate final heatmap and apply ReLU
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)

        # 7. Normalize the heatmap between 0 and 1 for visualization
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()