# =======================
# mandatory coding task 2
# =======================

import torch

from datasets.dataloader import CustomDataLoader
from datasets.transforms import get_eval_transform
from eval import evaluate
from model.model import Model
from set_seed import set_seed
from utils.config import load_config


@torch.no_grad()
def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    PROJECT_ROOT = cfg["paths"]["project_root"]
    DATASET_ROOT = cfg["paths"]["dataset_root"]

    TEST_SPLIT = PROJECT_ROOT / cfg["splits"]["test"]
    MODEL_PATH = PROJECT_ROOT / cfg["outputs"]["model_dir"]
    OUTPUT_PATH = PROJECT_ROOT / cfg["outputs"]["output_dir"]

    BATCH_SIZE = cfg["training"]["batch_size"]

    if cfg["hardware"]["device"] == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = cfg["hardware"]["device"]

    val_transform = get_eval_transform()


    test_loader = CustomDataLoader(
        dataset_root=DATASET_ROOT,
        split_file=TEST_SPLIT,
        transform=val_transform,
        batch_size=BATCH_SIZE,
        shuffle=False
    ).get_data_loader()

    num_classes = len(test_loader.dataset.classes)

    model = Model(num_classes=num_classes).get_model()
    model.load_state_dict(torch.load(MODEL_PATH / "best_model_mild_aug.pt", map_location=DEVICE))
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