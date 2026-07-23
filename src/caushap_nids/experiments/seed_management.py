"""Reproducible seed management for ablation experiments."""
from __future__ import annotations

import random

import numpy as np


def set_all_seeds(seed: int) -> np.random.Generator:
    """Set Python, NumPy, and (if available) PyTorch seeds. Returns a numpy RNG."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *tags: str | int) -> int:
    """Deterministically derive a child seed from a base seed and string tags.

    Useful for giving each (config, dataset, run_index) combination a unique seed
    without requiring an explicit seed list.
    """
    h = hash((base_seed, *tags)) & 0xFFFF_FFFF
    return int(h)
