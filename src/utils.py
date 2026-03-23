import os
import json
import random
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_SPLITS = os.path.join(PROJECT_ROOT, "data", "splits")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "experiments", "configs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "checkpoints")
BLOG_ASSETS = os.path.join(PROJECT_ROOT, "blog", "assets")


def load_config(name: str) -> dict:
    path = os.path.join(CONFIGS_DIR, name)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_metrics(run_name: str, metrics: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{run_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(run_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{run_name}_metrics.json")
    with open(path, "r") as f:
        return json.load(f)


def save_checkpoint(model, run_name: str):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINTS_DIR, f"{run_name}.pt")
    torch.save(model.state_dict(), path)


def load_checkpoint(model, run_name: str, device: torch.device = None):
    path = os.path.join(CHECKPOINTS_DIR, f"{run_name}.pt")
    state = torch.load(path, map_location=device or get_device(), weights_only=True)
    model.load_state_dict(state)
    return model


def append_to_summary(run_name: str, metrics: dict):
    """Append a row to experiments/results/summary.csv."""
    import csv

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "summary.csv")
    file_exists = os.path.exists(path)
    row = {"run_name": run_name, **metrics}
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
