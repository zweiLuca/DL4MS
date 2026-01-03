# =======================
# mandatory coding task 2
# =======================

import torch
from torch.utils.data import DataLoader

from datasets.eurosat_dataset import EuroSATDataset
from datasets.eurosat_ms_dataset import EuroSATMSDataset
from utils.seed_worker import seed_worker


class CustomDataLoader():
    def __init__(
            self,
            dataset_root: str,
            split_file: str,
            transform,
            batch_size: int,
            shuffle: bool,
            num_workers: int = 4,
            ms: bool = False,
            seed: int = 0
    ):
        self.dataset = EuroSATDataset(
            dataset_root=dataset_root,
            split_file=split_file,
            transform=transform
        )
        
        # --- For task 3 when using ms-images ---
        if ms:
            self.dataset = EuroSATMSDataset(
                dataset_root=dataset_root,
                split_file=split_file
            )

        g = torch.Generator()
        g.manual_seed(seed)

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    
    def get_data_loader(self):
        return self.loader