import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from scipy.ndimage import binary_erosion
import random
import config

COLORS = np.array([
    [0, 0, 0],        # Background - black
    [255, 0, 0],      # Muscle - red
    [0, 255, 0],      # Endometrium - green
    [0, 0, 255],      # Cavity - blue
]) / 255.0


def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3))
    for c in range(config.NUM_CLASSES):
        color[mask == c] = COLORS[c]
    return color


def plot_training_curves(train_losses, val_losses, val_dices, save_path=None):
    if save_path is None:
        save_path = f"{config.PLOTS_DIR}/training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(val_losses, label="Val Loss", color="darkorange")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(val_dices, color="darkorange", marker="o", markersize=2)
    axes[1].set(xlabel="Epoch", ylabel="Mean Dice", title="Validation Dice")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")


def binary_pred_to_class_mask(pred_logits):
    """Convert (3,H,W) binary logits to a single (H,W) class label map.
    Applies erosion to Cavity to prevent over-segmentation."""
    # binary = (torch.sigmoid(pred_logits) > 0.5).cpu().numpy()  # (3,H,W) bool
    probs = torch.sigmoid(pred_logits).cpu().numpy()
    binary = np.stack([probs[i] > config.THRESHOLDS[i] for i in range(3)])
    # Erode cavity (index 2) to shrink over-predicted regions
    binary[2] = binary_erosion(binary[2], iterations=2)
    mask = np.zeros(binary.shape[1:], dtype=np.int64)
    for i in range(3):
        mask[binary[i]] = i + 1  # Cavity wins (written last)
    return mask


def plot_predictions(model, test_loader, num_samples=4, save_path=None):
    if save_path is None:
        save_path = f"{config.PLOTS_DIR}/segmentation_results.png"
    model.eval()
    is_binary = config.MODE == "multibinary"

    all_samples = []
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
            outputs = model(images)
            if is_binary:
                pred = binary_pred_to_class_mask(outputs[0])
            else:
                pred = outputs.argmax(dim=1)[0].cpu().numpy()
            all_samples.append((images[0, 0].cpu().numpy(), masks[0].cpu().numpy(), pred))

    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    fig, axes = plt.subplots(len(selected), 3, figsize=(14, 4 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    for i, (img, gt, pr) in enumerate(selected):
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        gt_overlay = np.stack([img, img, img], axis=-1)
        gt_color = mask_to_color(gt)
        fg_gt = gt > 0
        gt_overlay[fg_gt] = 0.4 * gt_overlay[fg_gt] + 0.6 * gt_color[fg_gt]
        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title("Ground Truth Overlay")
        axes[i, 1].axis("off")

        pred_overlay = np.stack([img, img, img], axis=-1)
        pred_color = mask_to_color(pr)
        fg_mask = pr > 0
        pred_overlay[fg_mask] = 0.4 * pred_overlay[fg_mask] + 0.6 * pred_color[fg_mask]
        axes[i, 2].imshow(pred_overlay)
        axes[i, 2].set_title("Prediction Overlay")
        axes[i, 2].axis("off")

    legend_elements = [Patch(facecolor=COLORS[i+1], label=name) for i, name in enumerate(config.BINARY_CLASS_NAMES if config.MODE == "multibinary" else config.CLASS_NAMES[1:])]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved: {save_path}")
