import numpy as np
import torch

import config


def compute_metrics(pred, target):
    """Compute Dice and IoU per class."""
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


@torch.no_grad()
def evaluate(model, test_loader):
    """Run full evaluation on test set."""
    model.eval()
    all_metrics = {name: {"dice": [], "iou": []} for name in config.CLASS_NAMES}

    for images, masks in test_loader:
        images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
        outputs = model(images)
        metrics = compute_metrics(outputs, masks)

        for name in config.CLASS_NAMES:
            all_metrics[name]["dice"].append(metrics[name]["dice"])
            all_metrics[name]["iou"].append(metrics[name]["iou"])

    # Print summary (excluding background)
    classes_of_interest = config.CLASS_NAMES[1:]
    print(f"\n{'Class':<15} {'Dice':>8} {'IoU':>8}")
    print("-" * 33)
    for name in classes_of_interest:
        mean_dice = np.mean(all_metrics[name]["dice"])
        mean_iou = np.mean(all_metrics[name]["iou"])
        print(f"{name:<15} {mean_dice:>8.3f} {mean_iou:>8.3f}")

    mean_dice = np.mean([np.mean(all_metrics[n]["dice"]) for n in classes_of_interest])
    mean_iou = np.mean([np.mean(all_metrics[n]["iou"]) for n in classes_of_interest])
    print(f"\nMean Dice: {mean_dice:.3f} | Mean IoU: {mean_iou:.3f}")

    return all_metrics
