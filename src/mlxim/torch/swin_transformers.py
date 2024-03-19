from typing import Dict, Optional

import mlx.core as mx
import torch
from tqdm import tqdm

from .utils import pth_to_mlx_weights
from ..model import create_model, get_weights


def torchvision_swin_to_mlx(model_name: str, pth_ckpt_path: str, verbose: bool = False) -> Optional[Dict[str, mx.array]]:
    """Convert a torchvision Vision Transformer model to a MLX weights dict (key, mx.array).

    Args:
        model_name (str): model name
        pth_ckpt_path (str): path to Vision Transformer PyTorch state dict
    """

    model = create_model(model_name, weights=False)
    torch_state_dict = pth_to_mlx_weights(pth_ckpt_path)
    mlx_weights = get_weights(model)
    pretrained_mlx_weights = {}
    for k, v in torch_state_dict.items():
        if k.startswith("features.0"):
            mlx_k = k.replace("features.0.", "patch_embed.layers.")
        elif k.startswith("features."):
            k_split = k.split(".")
            if "cpb_mlp" in k: # for Swin Transformer v2
                mlx_k = f"features.layers.{int(k_split[1])-1}.layers.{k_split[2]}.attn.cpb_mlp.layers." + ".".join(k_split[5:])
            elif "mlp" in k:
                mlx_k = f"features.layers.{int(k_split[1])-1}.layers.{k_split[2]}.mlp.layers." + ".".join(k_split[4:])
            elif "reduction" in k or len(k_split) == 4: 
                mlx_k = f"features.layers.{int(k_split[1])-1}." + ".".join(k_split[2:])
            else: 
                mlx_k = f"features.layers.{int(k_split[1])-1}.layers." + ".".join(k_split[2:])
        else: #head
            mlx_k = k
        
        if mlx_k not in mlx_weights:
            if verbose:
                print(f"[ERROR] Not found {mlx_k} in mlx_weights")
        
        if len(v.shape) == 4:
            if v.shape == mlx_weights[mlx_k].shape:
                pretrained_mlx_weights[mlx_k] = v
            elif v.transpose(0, 2, 3, 1).shape == mlx_weights[mlx_k].shape:
                pretrained_mlx_weights[mlx_k] = v.transpose(0, 2, 3, 1)
            else:
                if verbose:
                    print(f"Not able to convert for k {k} with shape {v.shape} to {mlx_weights[mlx_k].shape}")
        else:
            pretrained_mlx_weights[mlx_k] = v
    
    is_ok = True
    for k in mlx_weights:
        if k not in pretrained_mlx_weights:
            print(f"[ERROR] Key {k} in mlx ViT implementation ({model_name}) not found")
            is_ok = False

    if is_ok is False:
        return None
    else:
        return pretrained_mlx_weights