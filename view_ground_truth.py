import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import os
import random
import sys

IMG_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/png_data/images"
MASK_DIR = "/Users/postgres/Desktop/phd/Project/adenomyosis/png_data/masks"

CLASS_NAMES = ["Muscle", "Endometrium", "Cavity"]
COLORS = np.array([
    [0, 0, 0],        # 0 = Background
    [255, 0, 0],      # 1 = Muscle
    [0, 255, 0],      # 2 = Endometrium
    [0, 0, 255],      # 3 = Cavity
]) / 255.0


def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3))
    for c in range(4):
        color[mask == c] = COLORS[c]
    return color


def visualize_ground_truth(n=4):
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    selected = random.sample(files, min(n, len(files)))

    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = np.array([axes])

    for i, fname in enumerate(selected):
        img = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert("L")) / 255.0
        mask = np.array(Image.open(os.path.join(MASK_DIR, fname)))

        # Raw ultrasound
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Ultrasound: {fname}", fontsize=8)
        axes[i, 0].axis("off")

        # Ground truth drawn on ultrasound
        overlay = np.stack([img, img, img], axis=-1)
        gt_color = mask_to_color(mask)
        fg = mask > 0
        overlay[fg] = 0.4 * overlay[fg] + 0.6 * gt_color[fg]
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Ground Truth Overlay")
        axes[i, 1].axis("off")

    legend_elements = [Patch(facecolor=COLORS[i+1], label=CLASS_NAMES[i]) for i in range(3)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/ground_truth_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to plots/ground_truth_overlay.png")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    visualize_ground_truth(n)
