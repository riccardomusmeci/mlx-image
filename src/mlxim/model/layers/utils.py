import collections.abc
from itertools import repeat
from math import prod
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _ntuple(n: int) -> Callable:
    """_ntuple.

    Args:
        n (int): tuple dim

    Returns:
        Callable: callable
    """

    def parse(x):  # type: ignore
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _randperm(n: int) -> mx.array:
    """
    MLX does not provide randperm, so emulate it by sorting random keys.
    Returns a random permutation of [0, n).
    """
    keys = mx.random.uniform(shape=(n,))
    return mx.argsort(keys)


def cat_keep_shapes(x_list: List[mx.array]) -> Tuple[mx.array, List[Tuple[int, ...]], List[int]]:
    """
    Flatten all leading dimensions of each tensor into a single token dimension,
    keeping the last (feature) dimension intact, and concatenate along tokens.

    Args:
        x_list: List of tensors with arbitrary leading dims and a last feature dim.

    Returns:
        flattened: Concatenated tensor of shape (sum_i tokens_i, C).
        shapes: Original shapes of each tensor.
        num_tokens: Number of tokens (product of all dims except last) per tensor.
    """
    shapes: List[Tuple[int, ...]] = [x.shape for x in x_list]
    num_tokens: List[int] = [int(prod(shape[:-1])) for shape in shapes]

    flattened_pieces = [
        x.reshape(-1, x.shape[-1]) for x in x_list
    ]
    flattened = mx.concat(flattened_pieces, axis=0)
    return flattened, shapes, num_tokens


def uncat_with_shapes(
    flattened: mx.array,
    shapes: List[Tuple[int, ...]],
    num_tokens: List[int],
) -> List[mx.array]:
    """
    Inverse of `cat_keep_shapes`.

    Args:
        flattened: Tensor of shape (sum_i tokens_i, C_flat).
        shapes: Original shapes returned by `cat_keep_shapes`.
        num_tokens: Token counts per original tensor.

    Returns:
        List of tensors, each reshaped back to its original leading
        dimensions with the (possibly updated) feature dimension.
    """
    outputs: List[mx.array] = []
    start = 0
    feature_dim = flattened.shape[-1]

    for shape, n_tokens in zip(shapes, num_tokens):
        end = start + n_tokens
        slice_i = flattened[start:end]

        target_shape = shape[:-1] + (feature_dim,)
        outputs.append(slice_i.reshape(target_shape))

        start = end

    return outputs

def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """
    MLX-friendly version of timm.named_apply.

    Iterates over all submodules (and optionally the root) using
    :meth:`mlx.nn.Module.named_modules` and applies ``fn(module=..., name=...)``.
    """
    # Collect all modules with their hierarchical names.
    for mod_name, mod in module.named_modules():
        if mod_name == "" and not include_root:
            continue
        fn(module=mod, name=mod_name)
    return module
