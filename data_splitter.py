import os
import random

from pathlib import Path
from typing import List


SEED = 1234567
random.seed(SEED)


class DataSplitter():
    """Class for creating train, validation and test splits of a dataset."""
    
    def __init__(self, project_dir: str, data_dir: str, train_size: int = 2500, val_size: int = 1000, test_size: int = 2000) -> None:
        self.project_dir: Path = Path(project_dir)
        self.data_dir: Path = Path(data_dir)

        self.train_size: int = train_size
        self.val_size: int = val_size
        self.test_size: int = test_size

        self.train_split: List[str] = []
        self.val_split: List[str] = []
        self.test_split: List[str] = []

    def split_data(self) -> None:
        classes: List[str] = sorted(next(os.walk(self.data_dir))[1])
        n_classes = len(classes)

        n_train = self.train_size // n_classes
        n_val = self.val_size // n_classes
        n_test = self.test_size // n_classes

        root = os.path.basename(self.data_dir)

        for cls in classes:
            imgs: List[str] = sorted(next(os.walk(self.data_dir / cls))[2])
            random.shuffle(imgs)

            assert len(imgs) >= n_train + n_val + n_test

            train = imgs[:n_train]
            val = imgs[n_train:n_train + n_val]
            test = imgs[n_train + n_val:n_train + n_val + n_test]
            
            self.train_split.extend([f"{root}/{cls}/{i}" for i in train])
            self.val_split.extend([f"{root}/{cls}/{i}" for i in val])
            self.test_split.extend([f"{root}/{cls}/{i}" for i in test])

        if not self._validate_splits():
            raise AssertionError("Error: Splits are not disjoint!")

        self._save_splits()
    
    def _validate_splits(self) -> bool:
        train_set = set(self.train_split)
        val_set = set(self.val_split)
        test_set = set(self.test_split)
        
        return train_set.isdisjoint(val_set) and train_set.isdisjoint(test_set) and val_set.isdisjoint(test_set)
    
    def _save_splits(self):
        split_dir: str = self.project_dir / "splits"
        os.makedirs(split_dir, exist_ok=True)

        for name, split in zip(
            ["train.txt", "val.txt", "test.txt"],
            [self.train_split, self.val_split, self.test_split]
        ):
            with open(split_dir / name, "w") as f:
                for item in split:
                    f.write(item + "\n")


# === Example ===

ds = DataSplitter(".", "coding_task_data/EuroSAT_RGB") # works with absolute and relative paths
ds.split_data()
