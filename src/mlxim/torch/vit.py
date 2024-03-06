from typing import Dict, Optional

import mlx.core as mx
import torch
from tqdm import tqdm

from ..model import create_model, get_weights


def torchvision_vit_to_mlx(model_name: str, pth_ckpt_path: str, verbose: bool = False) -> Optional[Dict[str, mx.array]]:
    """Convert a torchvision Vision Transformer model to a MLX weights dict (key, mx.array).

    Args:
        model_name (str): model name
        pth_ckpt_path (str): path to Vision Transformer PyTorch state dict
    """

    model = create_model(model_name, weights=False)
    torch_state_dict = torch.load(pth_ckpt_path, map_location="cpu")
    weights = get_weights(model)
    mlx_weights = {}
    for k, v in torch_state_dict.items():
        if "encoder_layer" in k:
            if "in_proj" in k:
                """
                encoder.layers.layers.0.self_attention.query_proj.weight (768, 768)
                encoder.layers.layers.0.self_attention.key_proj.weight (768, 768)
                encoder.layers.layers.0.self_attention.value_proj.weight (768, 768)
                encoder.layers.layers.0.self_attention.out_proj.weight (768, 768)

                encoder.layers.encoder_layer_0.self_attention.out_proj.weight
                """
                splits = k.split(".")
                idx = splits[2].split("_")[-1]
                data = splits[-1].split("_")[-1]
                q_weight, k_weight, v_weight = torch.chunk(v, chunks=3, dim=0)
                query = mx.array(q_weight.detach().cpu().numpy())
                key = mx.array(k_weight.detach().cpu().numpy())
                value = mx.array(v_weight.detach().cpu().numpy())

                mlx_weights[f"encoder.layers.layers.{idx}.self_attention.query_proj.{data}"] = query
                mlx_weights[f"encoder.layers.layers.{idx}.self_attention.key_proj.{data}"] = key
                mlx_weights[f"encoder.layers.layers.{idx}.self_attention.value_proj.{data}"] = value

            elif "out_proj" in k:
                """
                encoder.layers.layers.0.self_attention.out_proj.weight (768, 768)
                encoder.layers.layers.0.self_attention.out_proj.bias (768,)
                """
                splits = k.split(".")
                data = splits[-1]
                idx = splits[2].split("_")[-1]
                v = mx.array(v.detach().cpu().numpy())
                mlx_weights[f"encoder.layers.layers.{idx}.self_attention.out_proj.{data}"] = v
            else:
                if "mlp" in k:
                    splits = k.split(".")
                    idx = splits[2].split("_")[-1]
                    linear = splits[4]
                    data = splits[5]
                    mlx_k = f"encoder.layers.layers.{idx}.mlp.{linear}.{data}"
                else:
                    splits = k.split(".")
                    idx = splits[2].split("_")[-1]
                    ln = splits[3]
                    data = splits[4]
                    mlx_k = f"encoder.layers.layers.{idx}.{ln}.{data}"
                mlx_weights[mlx_k] = mx.array(v.detach().cpu().numpy())
        elif "heads" in k:
            k_split = k.split(".")
            data = k_split[-1]
            mlx_k = f"heads.layers.0.{data}"
            mlx_weights[mlx_k] = mx.array(v.detach().cpu().numpy())
        else:
            v = mx.array(v.detach().cpu().numpy())
            if len(v.shape) == 4:
                v = v.transpose(0, 2, 3, 1)
            mlx_weights[k] = v
        # print(f"[ERROR] Key {k} in mlx ViT implementation ({model_name}) not found")
    # check if all keys are present
    is_ok = True
    for k in weights:
        if k not in mlx_weights:
            print(f"[ERROR] Key {k} in mlx ViT implementation ({model_name}) not found")
            is_ok = False

    if is_ok is False:
        return None
    else:
        return mlx_weights
