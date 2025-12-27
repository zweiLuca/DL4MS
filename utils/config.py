import yaml
from pathlib import Path


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["project_root"] = Path(cfg["paths"]["project_root"])
    cfg["paths"]["dataset_root"] = Path(cfg["paths"]["dataset_root"])

    return cfg