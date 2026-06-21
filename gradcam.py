
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MedCLIP-main', 'MedCLIP-main'))

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AgeGenderModel, MEDCLIP_TRANSFORM
from train import AgeDataset, get_labels, get_images


# ---------------------------------------------------------------------------
# Grad-CAM for Swin Transformer token outputs
# ---------------------------------------------------------------------------

class SwinGradCAM:

    def __init__(self, model: AgeGenderModel, target_layer: nn.Module,
                 task: str = 'age'):
        assert task in ('age', 'sex'), "task must be 'age' or 'sex'"
        self.model        = model
        self.target_layer = target_layer
        self.task         = task
        self._acts        = None
        self._grads       = None

        self._fwd_hook = target_layer.register_forward_hook(self._fwd)
        self._bwd_hook = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        # out: (B, num_tokens, C)  for SwinStage blocks
        self._acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        # grad_out[0]: (B, num_tokens, C)
        self._grads = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @torch.enable_grad()
    def generate(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Returns heatmap array of shape (B, H, W) with values in [0, 1].
        pixel_values: (B, 3, 224, 224), already on the correct device.
        """
        self.model.zero_grad()
        pixel_values = pixel_values.requires_grad_(False)

        age_pred, sex_pred = self.model(pixel_values)

        if self.task == 'age':
            # sum over batch so we get per-image gradients via .backward()
            score = age_pred.sum()
        else:
            # winning class logit per sample, then sum
            score = sex_pred[torch.arange(len(sex_pred)),
                             torch.argmax(sex_pred, dim=1)].sum()

        self.model.zero_grad()
        score.backward()

        acts  = self._acts   # (B, N, C)
        grads = self._grads  # (B, N, C)

        # channel weights: mean over tokens
        weights = grads.mean(dim=1, keepdim=True)  # (B, 1, C)

        # weighted combination over channels, then mean -> (B, N)
        cam = (weights * acts).sum(dim=-1)          # (B, N)
        cam = torch.clamp(cam, min=0)

        # fold token sequence back to 2-D grid
        # Swin-Tiny with 224 input: final stage has 7x7 = 49 tokens
        N = cam.shape[1]
        H = W = int(N ** 0.5)
        if H * W != N:
            # fallback: use 1-D arrangement
            H, W = 1, N
        cam = cam.reshape(-1, H, W).cpu().numpy()  # (B, H, W)

        # normalise per image
        out = np.zeros_like(cam)
        for i in range(cam.shape[0]):
            mn, mx = cam[i].min(), cam[i].max()
            out[i] = (cam[i] - mn) / (mx - mn + 1e-8)
        return out


def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    img_np  : (H, W, 3) float in [0, 1]
    heatmap : (h, w) float in [0, 1]
    Returns (H, W, 3) uint8 RGB overlay.
    """
    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8       = np.uint8(255 * img_np)
    overlay         = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation; return (H, W, 3) float [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)


def run_gradcam(checkpoint, medclip_ckpt, save_dir, task='age',
                n_images=8, batch_size=8, num_workers=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- build test set ----
    ages, genders = get_labels()
    images_ls     = get_images()

    bins = [
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
        (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 35),
    ]
    test_id = []
    for lo, hi in bins:
        idx = ages[(ages >= lo) & (ages < hi)].index.tolist()
        cut = int(len(idx) * 0.8)
        test_id.extend(idx[cut:])

    test_images  = [images_ls[i] for i in test_id]
    test_ages    = [ages[i]      for i in test_id]
    test_genders = [genders[i]   for i in test_id]

    ds = AgeDataset(test_images, test_ages, test_genders, transform=MEDCLIP_TRANSFORM)
    dl = DataLoader(ds, batch_size=min(batch_size, n_images),
                    shuffle=False, num_workers=num_workers, pin_memory=True)

    # ---- load model ----
    model = AgeGenderModel(medclip_checkpoint=medclip_ckpt).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {checkpoint}")

    # ---- target layer: last block of last Swin stage ----
    # encoder.model  →  SwinModel
    # .encoder.layers[-1].blocks[-1]  →  last SwinStage last block
    target_layer = model.encoder.model.encoder.layers[-1].blocks[-1]
    gradcam = SwinGradCAM(model, target_layer, task=task)

    os.makedirs(save_dir, exist_ok=True)
    saved = 0

    for imgs, age_labels, sex_labels in tqdm(dl, desc="Grad-CAM"):
        if saved >= n_images:
            break
        imgs = imgs.to(device)

        heatmaps = gradcam.generate(imgs)   # (B, H, W)

        for j in range(imgs.shape[0]):
            if saved >= n_images:
                break
            img_np  = denormalize(imgs[j])
            heatmap = heatmaps[j]
            overlay = overlay_heatmap(img_np, heatmap, alpha=0.4)

            true_age    = float(age_labels[j])
            true_gender = int(sex_labels[j])

            # --- figure ---
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np);       axes[0].set_title("Original");         axes[0].axis("off")
            axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title(f"Grad-CAM ({task})"); axes[1].axis("off")
            axes[2].imshow(overlay);      axes[2].set_title("Overlay");           axes[2].axis("off")

            gender_str = "M" if true_gender == 1 else "F"
            fig.suptitle(f"True age={true_age:.1f}  sex={gender_str}", fontsize=11)
            plt.tight_layout()

            out_path = os.path.join(save_dir, f"gradcam_{task}_{saved:04d}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved += 1

    gradcam.remove_hooks()
    print(f"Saved {saved} Grad-CAM images to: {save_dir}")
