import torch

from datasets.dataloader import CustomDataLoader
from datasets.transforms import get_eval_transform, get_train_transform_mild, get_train_transform_strong
from model.model import Model
from set_seed import set_seed
from training_loop import train_model
from utils.config import load_config

# ===============
# TRAINING SCRIPT
# ===============

def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    PROJECT_ROOT = cfg["paths"]["project_root"]
    DATASET_ROOT = cfg["paths"]["dataset_root"]

    TRAIN_SPLIT = cfg["paths"]["project_root"] / cfg["splits"]["train"]
    VAL_SPLIT = cfg["paths"]["project_root"] / cfg["splits"]["val"]
    MODEL_PATH = PROJECT_ROOT / cfg["outputs"]["model_dir"]

    BATCH_SIZE = cfg["training"]["batch_size"]
    NUM_EPOCHS = cfg["training"]["num_epochs"]
    LR = cfg["training"]["learning_rate"]

    if cfg["hardware"]["device"] == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = cfg["hardware"]["device"]

    if cfg["training"]["augmentation"] == "strong":
        train_transform = get_train_transform_strong()
        experiment_name = "strong_aug"
    else:
        train_transform = get_train_transform_mild()
        experiment_name = "mild_aug"

    val_transform = get_eval_transform()


    train_loader = CustomDataLoader(
        dataset_root=DATASET_ROOT,
        split_file=TRAIN_SPLIT,
        transform=train_transform,
        batch_size=BATCH_SIZE,
        shuffle=True
    ).get_data_loader()

    val_loader = CustomDataLoader(
        dataset_root=DATASET_ROOT,
        split_file=VAL_SPLIT,
        transform=val_transform,
        batch_size=BATCH_SIZE,
        shuffle=False
    ).get_data_loader()

    num_classes = len(train_loader.dataset.classes)

    model = Model(num_classes=num_classes).get_model()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    print(
        "========================\n"
        f"Training the model on {DEVICE} with:\n"
        f"- {cfg["training"]["augmentation"]} data augmentation"
        f"- {len(train_loader.dataset)} images\n"
        f"- {round(len(train_loader.dataset) / BATCH_SIZE)} batches\n"
        f"- batchsize {BATCH_SIZE}\n"
        f"- {NUM_EPOCHS} epochs\n"
        "========================\n"
    )
    history, best_state = train_model(
        model=model,
        trian_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=NUM_EPOCHS
    )

    MODEL_PATH.mkdir(exist_ok=True)

    torch.save(
        best_state,
        MODEL_PATH / f"best_model_{experiment_name}.pt"
    )

    print(f"Training finished for {experiment_name}.")

if __name__ == "__main__":
    main()