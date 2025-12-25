# =======================
# mandatory coding task 2
# =======================

from torch.utils.data import DataLoader

from eurosat_dataset import EuroSATDataset


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


# === Example ===

if __name__ == "__main__":
    from multiprocessing import freeze_support
    from transforms import get_train_transform_mild, get_train_transform_strong

    freeze_support()

    loader = CustomDataLoader(
        dataset_root="coding_task_data/EuroSAT_RGB",
        split_file="./splits/train.txt",
        transform=get_train_transform_strong(),
        batch_size=16,
        shuffle=True
        ).get_data_loader()

    images, labels, paths = next(iter(loader))
    print(images.shape, labels.shape)