# =========================
# ranking_final.py
# =========================

import random
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from datasets.dataloader import CustomDataLoader
from datasets.transforms import get_eval_transform
from utils.config import load_config
from utils.set_seed import set_seed


def rank_top_bottom_classes(
    logits: torch.Tensor,
    paths: list,
    dataset_root: Path,
    class_indices: list,
    class_names: list,
    k: int = 5,
    save_dir: Path = None
):
    if save_dir is None:
        save_dir = Path("outputs/ranking")
    save_dir.mkdir(parents=True, exist_ok=True)

    for cls_idx in class_indices:
        cls_name = class_names[cls_idx]
        scores = logits[:, cls_idx]
        sorted_idx = torch.argsort(scores)

        bottom_idx = sorted_idx[:k]
        top_idx = sorted_idx[-k:]

        def save_plot(indices, title, suffix):
            plt.figure(figsize=(12, 3))
            for i, idx in enumerate(indices):
                img_path = dataset_root / paths[idx]
                img = Image.open(img_path).convert("RGB")
                plt.subplot(1, k, i + 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"{scores[idx]:.2f}", fontsize=10)
            plt.suptitle(title, fontsize=12)
            plt.tight_layout()
            file_path = save_dir / f"{cls_name}_{suffix}.png"
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()
            print(f"Saved: {file_path}")

        save_plot(top_idx, f"Top-{k} images for class '{cls_name}'", "top")
        save_plot(bottom_idx, f"Bottom-{k} images for class '{cls_name}'", "bottom")


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    
    PROJECT_ROOT = Path(cfg["paths"]["project_root"])
    DATASET_ROOT = Path("coding_task_data")
    OUTPUT_DIR = PROJECT_ROOT / cfg["outputs"]["output_dir"]

    logits = torch.load(OUTPUT_DIR / "test_logits.pt")
    paths = torch.load(OUTPUT_DIR / "test_paths.pt")

    test_loader = CustomDataLoader(
        dataset_root=DATASET_ROOT,
        split_file=PROJECT_ROOT / cfg["splits"]["split_dir"] / cfg["splits"]["test"],
        transform=get_eval_transform(),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    ).get_data_loader()

    class_names = test_loader.dataset.classes
    n_classes = len(class_names)

    class_indices = random.sample(range(n_classes), 3)

    rank_top_bottom_classes(
        logits=logits,
        paths=paths,
        dataset_root=DATASET_ROOT,
        class_indices=class_indices,
        class_names=class_names,
        k=5,
        save_dir=OUTPUT_DIR / "ranking"
    )


if __name__ == "__main__":
    main()