# =======================
# mandatory coding task 2
# =======================

import torch

from datasets.dataloader import CustomDataLoader
from datasets.transforms import get_eval_transform, get_ms_eval_transform
from evaluation.eval import evaluate
from models.custom_classifier import CustomClassifier
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

    # --- For task 3 when using ms-images ---
    MS = False
    if cfg["paths"]["dataset_root"].match("*_MS"):
        MS = True

    PROJECT_ROOT = cfg["paths"]["project_root"]
    DATASET_ROOT = cfg["paths"]["dataset_root"]

    # --- For task 3 when using ms-images ---
    if MS:
        TEST_SPLIT = PROJECT_ROOT / cfg["splits"]["split_dir"] / cfg["splits"]["test_ms"]
    else:
        TEST_SPLIT = PROJECT_ROOT / cfg["splits"]["split_dir"] / cfg["splits"]["test"]

    MODEL_PATH = PROJECT_ROOT / cfg["outputs"]["model_dir"]

    BATCH_SIZE = cfg["training"]["batch_size"]

    if cfg["hardware"]["device"] == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = cfg["hardware"]["device"]
    
    # --- For task 3 when using ms-images ---
    if MS:
        test_transform = get_ms_eval_transform()
    else:
        test_transform = get_eval_transform()

    test_loader = CustomDataLoader(
        dataset_root=DATASET_ROOT,
        split_file=TEST_SPLIT,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
        ms=MS
    ).get_data_loader()

    num_classes = len(test_loader.dataset.classes)

    print(
        "========================\n"
        f"Running the prediction on {DEVICE} with:\n"
        f"- dataset {DATASET_ROOT}\n"
        f"- {len(test_loader.dataset)} test images\n"
        "========================\n"
    )

    # --- For task 3 when using ms-images ---
    if MS:
        model = CustomClassifier(num_classes=num_classes)
        model_path = MODEL_PATH / f"final_model_ms.pt"
    else:
        model = Model(num_classes=num_classes).get_model()
        model_path = MODEL_PATH / f"final_model.pt"

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    test_metrics = evaluate(
        model,
        test_loader,
        DEVICE,
        return_logits=True,
        return_paths=True
    )

    acc = test_metrics["accuracy"]
    tpr = test_metrics["tpr_per_class"]

    print(
        f"Prediction finished: \n"
        f"- Accuracy: {acc}\n"
        f"- TPR: {tpr}\n"
    )

if __name__ == "__main__":
    main()