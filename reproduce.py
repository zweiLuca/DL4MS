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

# ====================
# REPRODUCTION ROUTINE
# ====================

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
    model.load_state_dict(torch.load(MODEL_PATH / f"final_model.pt", map_location=DEVICE))
    model.to(DEVICE)

    metrics = evaluate(
        model,
        test_loader,
        DEVICE,
        return_logits=True,
        return_paths=True
    )

    new_logits = metrics["logits"]

    saved_logits_path = OUTPUT_PATH / "test_logits.pt"

    if cfg["reproduce"]["save_logits"]:
        OUTPUT_PATH.mkdir(exist_ok=True)
        torch.save(new_logits, saved_logits_path)
        print("[INFO] Test logits saved.")
        return

    if not saved_logits_path.exists():
        raise FileNotFoundError(
            "Saved logits not found. Run with --save_logits once."
        )
    
    old_logits = torch.load(saved_logits_path)

    if old_logits.shape != new_logits.shape:
        raise RuntimeError(
            f"Shape mismatch: old {old_logits.shape}, new {new_logits.shape}"
        )

    if torch.allclose(old_logits, new_logits, atol=1e-6):
        print("[SUCCESS] Reproduction successful: logits match.")
    else:
        max_diff = (old_logits - new_logits).abs().max().item()
        raise RuntimeError(
            f"[FAILURE] Logits differ! Max abs diff: {max_diff}"
        )
    
if __name__ == "__main__":
    main()