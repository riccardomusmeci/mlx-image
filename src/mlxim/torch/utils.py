import os
from typing import Dict

import mlx.core as mx
import torch
from tqdm import tqdm


def pth_to_mlx_weights(ckpt_path: str) -> Dict[str, mx.array]:
    """Convert a checkpoint of PyTorch to a MLX weights dict (key, mx.array).

    Args:
        ckpt_path (str): path to PyTorch checkpoint

    Returns:
        Dict[str, mx.array]: MLX weights dict (key, mx.array)
    """
    mlx_weights = {}
    if not ckpt_path.endswith(".pth") and not ckpt_path.endswith(".pt"):
        raise ValueError(f"Invalid file format: {ckpt_path}")
    pth_state_dict = torch.load(ckpt_path, map_location="cpu")
    for k, w in tqdm(pth_state_dict.items(), total=len(pth_state_dict.keys()), desc="Converting.."):
        w = mx.array(w.detach().cpu().numpy())
        mlx_weights[k] = w

    return mlx_weights
