import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    For EfficientNet-V2-S, pass target_layer = model.backbone.features[-1].
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global Average Pooling over spatial dimensions
        alpha = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Min-max normalization to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        _, _, h, w = input_tensor.shape
        return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def overlay_cam_on_image(original_rgb: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends a [0, 1] heatmap mask onto an RGB uint8 numpy image."""
    heatmap_uint8 = np.uint8(255 * np.clip(cam_mask, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)