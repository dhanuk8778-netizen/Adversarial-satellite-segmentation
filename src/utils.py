from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_cuda and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: torch.nn.Module, path: str, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location=None) -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@contextmanager
def timer(name: str = "block"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] {elapsed:.2f}s")
