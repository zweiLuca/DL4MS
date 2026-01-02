# =======================
# mandatory coding task 3
# =======================

import torch

from pathlib import Path
from skimage.io import imread
from torch.utils.data import Dataset
from typing import List, Tuple


class EuroSATMSDataset(Dataset):

    # Channels B04,B03,B02 + B08,B05,B06
    CHANNEL_IDXS = [3, 2, 1, 7, 4, 5]  

    def __init__(self, dataset_root: str, split_file: str) -> None:
        self.dataset_root = Path(dataset_root)
        
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

        class_name = Path(rel_path).parts[-2]
        label = self.class_to_idx[class_name]

        image = imread(img_path)

        image = image.astype("float32") / 65535.0
        image = image[:, :, self.CHANNEL_IDXS]
        image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, label, rel_path
    

# === Example ===

if __name__ == "__main__":
    ds = EuroSATMSDataset(
        dataset_root="./coding_task_data/EuroSAT_MS",
        split_file="./splits/train_ms.txt"
    )

    img, label, path = ds[0]
    print(img.shape, label, path)
    print(img.dtype)
    print(img.min().item(), img.max().item())