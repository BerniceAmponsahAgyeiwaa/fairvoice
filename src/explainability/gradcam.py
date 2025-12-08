# gradcam_no_cv2.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer_name="bn3"):
        self.model = model
        self.model.eval()

        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.activations = None
        self.gradients = None

        # Hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: (1,1,n_mels,T)
        Returns: heatmap (numpy array 0–1)
        """

        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        target = logits[:, class_idx]
        self.model.zero_grad()
        target.backward()

        # GAP over gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()

    def overlay(self, spectrogram, heatmap, alpha=0.5):
        """
        spectrogram: (n_mels, T) numpy
        heatmap: (n_mels, T) numpy 0–1

        Returns: PIL.Image
        """

        # Convert grayscale spectrogram → RGB
        spec_img = Image.fromarray((spectrogram * 255).astype(np.uint8))
        spec_img = spec_img.convert("RGB")

        # Heatmap → RGB using a colormap (manual "jet" style)
        cmap = self._jet_colormap(heatmap)

        heatmap_img = Image.fromarray(cmap)

        # Blend
        blended = Image.blend(spec_img, heatmap_img, alpha)
        return blended

    def _jet_colormap(self, heatmap):
        """
        Create a Jet-like colormap manually using NumPy
        heatmap: 2D array 0-1
        Returns: (H,W,3) uint8 RGB
        """

        h = heatmap
        r = np.clip(1.5 - np.abs(4*h - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4*h - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4*h - 1), 0, 1)

        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        return rgb
