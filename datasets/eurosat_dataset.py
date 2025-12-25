import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, List, Tuple


class EuroSATDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split_file: str,
        transform: Callable = None
    ) -> None:
        """
        dataset_root: path to directory containing EuroSAT_RGB

        split_file: path to train.txt / val.txt / test.txt
        
        transform: torchvision transforms
        """
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        with open(split_file, "r") as f:
            self.samples: List[str] = [line.strip() for line in f]

        self.classes = sorted(
            {Path(p).parts[-2] for p in self.samples}
        )
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rel_path = self.samples[idx]
        img_path = self.dataset_root.parent / rel_path

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        class_name = Path(rel_path).parts[-2]
        label = self.class_to_idx[class_name]

        return image, label, rel_path


# === Example ===

ds = EuroSATDataset(
    dataset_root="coding_task_data/EuroSAT_RGB",
    split_file="./splits/train.txt"
)

img, label, path = ds[0]
print(img.size, label, path)
