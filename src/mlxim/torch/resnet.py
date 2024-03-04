from typing import Dict

import mlx.core as mx
import torch

from ..model import create_model, get_weights
from .utils import pth_to_mlx_weights


def resnet_to_mlx(model_name: str, pth_ckpt_path: str, verbose: bool = False) -> Dict[str, mx.array]:
    """Convert a PyTorch ResNet18 model to a MLX weights dict (key, mx.array).

    Args:
        pth_state_dict (str): path to ResNet18 PyTorch state dict
        verbose (bool, optional): verbose mode. Defaults to False.

    Returns:
        Dict[str, mx.array]: MLX weights dict (key, mx.array)
    """

    # it still has the keys of the original model
    pth_resnet_weights = pth_to_mlx_weights(pth_ckpt_path)
    model = create_model(model_name, weights=False)
    # getting mlx model keys
    mlx_resnet_weights = get_weights(model)
    # new dict to store the converted weights
    mlx_weights = {}
    for k, v in mlx_resnet_weights.items():
        pth_k = k
        if k not in pth_resnet_weights:
            if k.startswith("layer"):
                pth_k = k.replace(".layers.", ".")
        if "conv" in k or "downsample" in k and len(v.shape) == 4:
            if verbose:
                print(f"Transposing {k} from {pth_resnet_weights[pth_k].shape} to {v.shape}")
            pth_v = pth_resnet_weights[pth_k]
            if pth_v.transpose(0, 2, 3, 1).shape == v.shape:
                v = pth_v.transpose(0, 2, 3, 1)
            else:
                print(f"[ERROR] {k} - mlx={v.shape} - torch={pth_v.shape}")
        else:
            v = pth_resnet_weights[pth_k]
        mlx_weights[k] = v
    return mlx_weights
