import matplotlib.pyplot as plt
import torch

from pathlib import Path
from PIL import Image


def rank_images(
        class_idx: int,
        class_name: str,
        logits: torch.Tensor,
        paths: list,
        dataset_root: Path,
        k: int = 5
):
    """
    Creates Top-k and Bottom-k ranking for one class.
    """
    
    scores = logits[:, class_idx]
    sorted_idx = torch.argsort(scores)

    bottom_idx = sorted_idx[:k]
    top_idx = sorted_idx[-k:]

    def plot(indices, title):
        plt.figure(figsize=(12, 3))
        for i, idx in enumerate(indices):
            img_path = dataset_root / paths[idx]
            img = Image.open(img_path).convert("RGB")

            plt.subplot(1, k, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{scores[idx]:.2f}")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    plot(top_idx, f"Top-{k} images for class '{class_name}'")
    plot(bottom_idx, f"Bottom-{k} images for class '{class_name}'")