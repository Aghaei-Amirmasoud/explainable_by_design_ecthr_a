import json
import os
import random
import time
from functools import wraps
from pathlib import Path

import numpy as np
import config


def seed_everything(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0     = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"{fn.__qualname__}  {time.perf_counter() - t0:.2f}s")
        return result
    return wrapper


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(array, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_npy(path):
    return np.load(path)


def format_metrics_table(metrics_by_name):
    all_keys  = sorted({k for m in metrics_by_name.values() for k in m})
    col_w     = max(len(n) for n in metrics_by_name) + 2
    metric_w  = max(len(k) for k in all_keys) + 2
    header    = f"{'Metric':<{metric_w}}" + "".join(f"{n:>{col_w}}" for n in metrics_by_name)
    rows      = [header, "─" * len(header)]
    for key in all_keys:
        row = f"{key:<{metric_w}}"
        for m in metrics_by_name.values():
            row += f"{m.get(key, float('nan')):>{col_w}.4f}"
        rows.append(row)
    return "\n".join(rows)
