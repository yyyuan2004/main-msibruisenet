"""
seed.py — Global random seed fixing for reproducibility
=========================================================

Fixes seeds for: Python random, NumPy, PyTorch (CPU & CUDA).
Also sets deterministic cuDNN behaviour.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, set cuDNN to deterministic mode
                       (may reduce performance slightly).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
