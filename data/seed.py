import numpy as np
import torch
from torch.utils.data import get_worker_info

def seed_worker(worker_id: int):
    info = get_worker_info()
    ds = info.dataset  # this is the worker's dataset copy

    # torch.initial_seed() is different per worker; good entropy source
    seed = (torch.initial_seed() + worker_id) % (2**32)

    # Reseed dataset RNG
    ds.rng = np.random.default_rng(seed)

    # Reseed augmentation RNG (critical)
    if getattr(ds, "augment", None) is not None and hasattr(ds.augment, "rng"):
        ds.augment.rng = np.random.default_rng(seed + 12345)