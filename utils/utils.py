import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


def fix_seed(seed: int, deterministic: bool = False) -> None:
    """
    ランダムシードの固定

    Args:
        seed(int)          : 固定するシード値
        deterministic(bool): GPUに決定的動作させるか
                             Falseだと速いが学習結果が異なる
    """
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
