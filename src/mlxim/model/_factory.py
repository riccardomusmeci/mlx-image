from typing import Any, Dict

import mlx.nn as nn

from ._registry import MODEL_ENTRYPOINT
from ._utils import download_from_hf, load_weights


def list_models() -> None:
    """List all available image models."""
    print("Available models:")
    for model_name in list(MODEL_ENTRYPOINT.keys()):
        print(f"\t- {model_name}")


def create_model(
    model_name: str,
    weights: bool = True,
    num_classes: int = 1000,
    strict: bool = False,
    verbose: bool = False,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    """Create an image model.

    Example:

    ```
    >>> from mlxim.model import create_model

    >>> # Create a Phi2 model with no pretrained weights.
    >>> model = create_model('resnet18')

    >>> # Create a resnet18 model with pretrained weights from HF.
    >>> model = create_model('resnet18', weights=True)

    >>> # Create a resnet18 model with custom weights.
    >>> model = create_model('resnet18', weights="path/to/weights.npz")
    ```

    Args:
        model_name (str): model name
        weights (bool, optional): if True, downloads weights from HF. If starts with "hf://", downloads weights from HF. If str, loads weights from the given path. Defaults to True.
        num_classes (int, optional): number of classes. Defaults to 1000.
        strict (bool, optional): if True, raises an error if some weights are not loaded. Defaults to False.
        verbose (bool, optional): if True, prints information during loading. Defaults to False.

    Raises:
        ValueError: if model_name is not available

    Returns:
        nn.Module: model
    """
    if model_name not in MODEL_ENTRYPOINT:
        raise ValueError(f"Model {model_name} not available")

    model = MODEL_ENTRYPOINT[model_name](num_classes=num_classes, **kwargs)

    if isinstance(weights, bool) and weights is True:
        weights_path = download_from_hf(model_name)
        model = load_weights(model, weights_path, strict=strict, verbose=verbose)
    elif isinstance(weights, str) and weights.startswith("hf://"):
        hf_weights_split = weights.replace("hf://", "").split("/")
        repo_id = "/".join(hf_weights_split[:-1])
        filename = hf_weights_split[-1]
        weights_path = download_from_hf(
            model_name=model_name,
            repo_id=repo_id,
            filename=filename,
        )
        model = load_weights(model, weights_path, strict=strict, verbose=verbose)
    elif isinstance(weights, str):
        model = load_weights(model, weights, strict=strict, verbose=verbose)  # type: ignore
    else:
        raise ValueError(f"Invalid weights type: {type(weights)}")

    return model
