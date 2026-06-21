import numpy as np
import torch
from scipy.ndimage import binary_erosion

import config


def compute_metrics(pred, target):
    """Compute Dice and IoU per class (multiclass)."""
    pred_mask = pred.argmax(dim=1)
    metrics = {}
    for c in range(config.NUM_CLASSES):
        p = (pred_mask == c).float().view(-1)
        t = (target == c).float().view(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        dice = (2 * intersection / (p.sum() + t.sum() + 1e-5)).item()
        iou = (intersection / (union + 1e-5)).item()
        metrics[config.CLASS_NAMES[c]] = {"dice": dice, "iou": iou}
    return metrics


def compute_metrics_binary(pred, target):
    """Compute Dice and IoU per binary head. pred: (B,3,H,W) logits."""
    pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()  # (B,3,H,W)
    # Erode cavity channel to fix over-segmentation
    pred_np[:, 2] = np.stack([binary_erosion(pred_np[b, 2], iterations=2) for b in range(pred_np.shape[0])])
    pred_binary = torch.from_numpy(pred_np).float().to(pred.device)
    metrics = {}
    for i, name in enumerate(config.BINARY_CLASS_NAMES):
        p = pred_binary[:, i].reshape(-1)
        t = (target == (i + 1)).float().reshape(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        dice = (2 * intersection / (p.sum() + t.sum() + 1e-5)).item()
        iou = (intersection / (union + 1e-5)).item()
        metrics[name] = {"dice": dice, "iou": iou}
    return metrics


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    is_binary = config.MODE == "multibinary"
    class_names = config.BINARY_CLASS_NAMES if is_binary else config.CLASS_NAMES[1:]
    all_metrics = {name: {"dice": [], "iou": []} for name in class_names}

    for images, masks in test_loader:
        images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
        outputs = model(images)
        metrics = compute_metrics_binary(outputs, masks) if is_binary else compute_metrics(outputs, masks)
        names = config.BINARY_CLASS_NAMES if is_binary else config.CLASS_NAMES
        for name in names:
            if name in all_metrics:
                all_metrics[name]["dice"].append(metrics[name]["dice"])
                all_metrics[name]["iou"].append(metrics[name]["iou"])

    print(f"\n{'Class':<15} {'Dice':>8} {'IoU':>8}")
    print("-" * 33)
    for name in class_names:
        mean_dice = np.mean(all_metrics[name]["dice"])
        mean_iou = np.mean(all_metrics[name]["iou"])
        print(f"{name:<15} {mean_dice:>8.3f} {mean_iou:>8.3f}")

    mean_dice = np.mean([np.mean(all_metrics[n]["dice"]) for n in class_names])
    mean_iou = np.mean([np.mean(all_metrics[n]["iou"]) for n in class_names])
    print(f"\nMean Dice: {mean_dice:.3f} | Mean IoU: {mean_iou:.3f}")
    return all_metrics
