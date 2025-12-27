# =======================
# mandatory coding task 2
# =======================

from torch.utils.data import DataLoader

from datasets.eurosat_dataset import EuroSATDataset


class CustomDataLoader():
    def __init__(
            self,
            dataset_root: str,
            split_file: str,
            transform,
            batch_size: int,
            shuffle: bool,
            num_workers: int = 4
    ):
        self.dataset = EuroSATDataset(
            dataset_root=dataset_root,
            split_file=split_file,
            transform=transform
        )

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_data_loader(self):
        return self.loader