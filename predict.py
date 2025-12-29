# =======================
# mandatory coding task 2
# =======================

import torch

from datasets.dataloader import CustomDataLoader
from datasets.transforms import get_eval_transform
from evaluation.eval import evaluate
from models.model import Model
from utils.set_seed import set_seed
from utils.config import load_config


# =================
# PREDICTION SCRIPT
# =================

@torch.no_grad()
def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    PROJECT_ROOT = cfg["paths"]["project_root"]
    DATASET_ROOT = cfg["paths"]["dataset_root"]

    TEST_SPLIT = PROJECT_ROOT / cfg["splits"]["split_dir"] / cfg["splits"]["test"]
    MODEL_PATH = PROJECT_ROOT / cfg["outputs"]["model_dir"]
    OUTPUT_PATH = PROJECT_ROOT / cfg["outputs"]["output_dir"]

    BATCH_SIZE = cfg["training"]["batch_size"]

    if cfg["hardware"]["device"] == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = cfg["hardware"]["device"]

    aug = cfg["training"]["augmentation"]
    if aug == "strong":
        experiment_name = "strong_aug"
    else:
        experiment_name = "mild_aug"

    test_transform = get_eval_transform()

    test_loader = CustomDataLoader(
        dataset_root=DATASET_ROOT,
        split_file=TEST_SPLIT,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        shuffle=False
    ).get_data_loader()

    num_classes = len(test_loader.dataset.classes)

    model = Model(num_classes=num_classes).get_model()
    model.load_state_dict(torch.load(MODEL_PATH / f"best_model_{experiment_name}.pt", map_location=DEVICE))
    model.to(DEVICE)

    test_metrics = evaluate(
        model,
        test_loader,
        DEVICE,
        return_logits=True,
        return_paths=True
    )

    logits = test_metrics["logits"]
    labels = test_metrics["labels"]
    paths = test_metrics["paths"]

    acc = test_metrics["accuracy"]
    tpr = test_metrics["tpr_per_class"]

    OUTPUT_PATH.mkdir(exist_ok=True)

    torch.save(logits, OUTPUT_PATH / "test_logits.pt")
    torch.save(paths, OUTPUT_PATH / "test_paths.pt")

    print(
        f"Accuracy: {acc}\n"
        f"TPR: {tpr}\n"
    )
    print("Test logits saved.")


if __name__ == "__main__":
    main()