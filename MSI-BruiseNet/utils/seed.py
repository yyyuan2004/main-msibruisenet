"""
seed.py - Global random seed fixing for reproducibility.

Responsibility:
    - Fix seeds for Python random, NumPy, PyTorch (CPU + CUDA).
    - Enable deterministic cuDNN behaviour.

I/O:
    Input  : integer seed
    Output : None (side effect: global RNG states set)
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
