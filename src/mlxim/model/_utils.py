import os
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

from ._registry import MODEL_CONFIG


def num_params(model: nn.Module) -> int:
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    return nparams


def get_weights(model: nn.Module) -> dict:
    """Return the model weights dict.

    Args:
        model (nn.Module): a model

    Returns:
        dict: model weights dict
    """
    state_dict = dict(tree_flatten(model.parameters()))
    return state_dict


def save_weights(weights: Dict[str, mx.array], output_path: str) -> None:
    """Save MLX weights to a given path.

    Args:
        weights (Dict[str, mx.array]): MLX weights dict (key, mx.array)
        output_path (str): path to save weights
    """
    output_dir = os.path.dirname(output_path)
    if len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)
    mx.savez(output_path, **weights)


def load_weights(model: nn.Module, weights: str, strict: bool = True, verbose: bool = False) -> nn.Module:
    """Load weights from a given path.

    Args:
        model (nn.Module): a LLM model
        weights (str): path to weights
        strict (bool, optional): whether to strictly enforce that the keys in weights match the keys of the model. Defaults to True.
        verbose (bool, optional): whether to print information during loading. Defaults to False.

    Returns:
        nn.Module: an nn.Module with loaded weights
    """

    assert os.path.exists(weights), f"Weights path {weights} does not exist."

    if verbose:
        print(f"\n> Loading weights from {weights}")

    pretrained_weights = dict(list(mx.load(weights).items()))
    # create a torch-like state dict { layer_name: weights }
    model_weights = dict(tree_flatten(model.parameters()))
    # check if pretrained_weights does not have more keys
    extras = set(pretrained_weights.keys()) - set(model_weights.keys())
    if extras:
        extras = " ".join(list(extras))  # type: ignore
        if strict:
            raise ValueError(f"Found extra keys in weights file: {extras}")
        else:
            if verbose:
                print(f"\t- [WARNING] Found extra keys in weights file: {extras}")

    # check if pretrained_weights does not have less keys
    missing = set(model_weights.keys()) - set(pretrained_weights.keys())
    if missing:
        missing = " ".join(list(missing))  # type: ignore
        if strict:
            raise ValueError(f"Missing keys in weights file: {missing}")
        else:
            if verbose:
                print(f"\t- [WARNING] Missing keys in weights file: {missing}")

    for k, w in model_weights.items():
        if k not in pretrained_weights:
            if strict:
                raise KeyError(f"Missing key {k} in weights file")
            else:
                if verbose:
                    print(f"> [WARNING] Missing key {k} in weights file")
            continue
        else:
            pretrained_w = pretrained_weights[k]
            # checking if pretrained_w has the same shape as w
            if pretrained_w.shape != w.shape:
                if strict:
                    raise ValueError(f"Expected shape {w.shape} for key {k}, got {pretrained_w.shape}")
                else:
                    if verbose:
                        print(f"> [WARNING] Expected shape {w.shape} for key {k}, got {pretrained_w.shape}")
                    pretrained_w = w
            model_weights[k] = pretrained_w

    model.update(tree_unflatten(list(model_weights.items())))
    return model


def download_from_hf(model_name: str, repo_id: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Download weights from HuggingFace Hub.

    Args:
        model_name (str): model name

    Returns:
        str: path to downloaded weights
    """

    if repo_id is None and filename is None:
        repo_id = MODEL_CONFIG[model_name].weights.repo_id
        filename = MODEL_CONFIG[model_name].weights.filename
    try:
        weights_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename)
    except Exception as e:
        print(f"[ERROR] Downloading weights from HuggingFace Hub failed for {model_name}: {e}.")
        quit()

    return weights_path
