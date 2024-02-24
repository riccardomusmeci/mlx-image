from typing import Any, Dict

import mlx.nn as nn

from ._registry import MODEL_ENTRYPOINT
from ._utils import download_from_hf, load_weights


def create_model(
    model_name: str,
    weights: bool = True,
    num_classes: int = 1000,
    strict: bool = False,
    verbose: bool = True,
    **kwargs: Dict[str, Any],
) -> nn.Module:
    """Create model.

    Args:
        model_name (str): model name
        weights (bool, optional): if True, downloads weights from HF. If a path loads weights from the given path. Defaults to True.
        num_classes (int, optional): number of classes. Defaults to 1000.
        strict (bool, optional): if True, raises an error if some weights are not loaded. Defaults to False.
        verbose (bool, optional): if True, prints information during loading. Defaults to True.

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
        model = load_weights(model, weights_path, strict=strict, verbose=verbose)  # type: ignore
    elif isinstance(weights, str):
        model = load_weights(model, weights, strict=strict, verbose=verbose)  # type: ignore

    return model
