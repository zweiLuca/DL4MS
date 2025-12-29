import math
import torch

from datasets.dataloader import CustomDataLoader
from datasets.transforms import get_eval_transform, get_train_transform_mild, get_train_transform_strong
from evaluation.plots import plot_validation_accuracy, plot_validation_tpr
from models.model import Model
from training.training_loop import train_model
from utils.set_seed import set_seed
from utils.config import load_config

# ===============
# TRAINING SCRIPT
# ===============

def run_training(
        experiment_name: str,
        train_transform,
        cfg,
        device
    ):
    PROJECT_ROOT = cfg["paths"]["project_root"]
    DATASET_ROOT = cfg["paths"]["dataset_root"]

    TRAIN_SPLIT = PROJECT_ROOT / cfg["splits"]["split_dir"] / cfg["splits"]["train"]
    VAL_SPLIT = PROJECT_ROOT / cfg["splits"]["split_dir"] / cfg["splits"]["val"]
    
    MODEL_PATH = PROJECT_ROOT / cfg["outputs"]["model_dir"]
    OUTPUT_PATH = PROJECT_ROOT / cfg["outputs"]["output_dir"]

    BATCH_SIZE = cfg["training"]["batch_size"]
    NUM_EPOCHS = cfg["training"]["num_epochs"]
    LR = cfg["training"]["learning_rate"]

    DEVICE = device

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
        f"Training {experiment_name} on {DEVICE} with:\n"
        f"- {len(train_loader.dataset)} images\n"
        f"- {math.ceil(len(train_loader.dataset) / BATCH_SIZE)} batches\n"
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

    MODEL_PATH.mkdir(exist_ok=True, parents=True)
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

    model_file = MODEL_PATH / f"best_model_{experiment_name}.pt"
    torch.save(best_state, model_file)

    plots_dir = OUTPUT_PATH / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    plot_validation_accuracy(
        history=history,
        save_path=plots_dir / f"val_acc_{experiment_name}.png"
    )

    plot_validation_tpr(
        history=history,
        class_names=train_loader.dataset.classes,
        save_path=plots_dir / f"val_tpr_{experiment_name}.png",
    )

    best_val_acc = max(history["val_acc"])

    print(
        f"Training finished for {experiment_name}: "
        f"best val acc = {best_val_acc:.4f}\n"
    )

    return {
        "experiment": experiment_name,
        "best_val_acc": best_val_acc,
        "best_state": best_state,
        "model_path": model_file
    }

def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    if cfg["hardware"]["device"] == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = cfg["hardware"]["device"]

    results = []

    # mild augmentation
    results.append(
        run_training(
            experiment_name="mild_aug",
            train_transform=get_train_transform_mild(),
            cfg=cfg,
            device=DEVICE
        )
    )

    # strong augmentation
    results.append(
        run_training(
            experiment_name="strong_aug",
            train_transform=get_train_transform_strong(),
            cfg=cfg,
            device=DEVICE
        )
    )

    best = max(results, key=lambda x: x["best_val_acc"])

    PROJECT_ROOT = cfg["paths"]["project_root"]
    MODEL_PATH = PROJECT_ROOT / cfg["outputs"]["model_dir"]

    final_model_path = MODEL_PATH / "final_model.pt"
    torch.save(best["best_state"], final_model_path)

    print(
        "========================\n"
        "FINAL MODEL SELECTION\n"
        f"Selected experiment: {best['experiment']}\n"
        f"Validation accuracy: {best['best_val_acc']:.4f}\n"
        f"Saved as: {final_model_path}\n"
        "========================"
    )

if __name__ == "__main__":
    main()