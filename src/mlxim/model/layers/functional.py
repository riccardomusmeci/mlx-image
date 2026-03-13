import mlx.core as mx
import mlx.nn as nn


def roll(x: mx.array, shifts: list[int], axes: list[int]) -> mx.array:
    """Roll the mx.array along the given dimensions.

    Args:
        x (mx.array): input mx.array
        shifts (List[int]): the number of places by which elements are shifted.
        axes (List[int]): the axis along which the elements are shifted.

    Returns:
        mx.array: mx.array with the elements shifted.
    """
    output = x
    for shift, axis in zip(shifts, axes):
        if shift == 0:
            continue
        if shift < 0:
            shift = x.shape[axis] + shift
        shift = shift % x.shape[axis]
        # split and concatenate
        splits = mx.split(output, [shift], axis=axis)
        output = mx.concatenate(list(splits[::-1]), axis=axis)
    return output


def normalize(x: mx.array, axis: int, eps: float = 1e-5) -> mx.array:
    """Normalize the input mx.array along the given axis.

    Args:
        x (mx.array): input array.
        axis (int): the axis along which the input mx.array is normalized.
        eps (float): A small value to avoid division by zero. Default: 1e-5.

    Returns:
        mx.array: The normalized mx.array.
    """
    return x / (mx.linalg.norm(x, 2, axis, keepdims=True) + eps)


def dropout(x: mx.array, p: float, training: bool) -> mx.array:
    """Dropout the input mx.array.

    Args:
        x (mx.array): input mx.array.
        p (float): dropout probability.
        training (bool): training flag.

    Returns:
        mx.array: The dropout mx.array.
    """
    if p > 0 and training:
        mask = mx.random.uniform(0, 1, x.shape, dtype=x.dtype) > p
        x = x * mask / (1 - p)
    return x


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    attn_mask: mx.array | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> mx.array:
    """Scaled dot-product attention."""
    attn_weights = mx.matmul(q, mx.swapaxes(k, -2, -1)) * scale
    if attn_mask is not None:
        attn_weights = mx.where(attn_mask != 0, attn_weights, float("-inf"))
    if is_causal:
        attn_weights = mx.tril(attn_weights, k=0)
    attn_weights = mx.softmax(attn_weights, axis=-1)
    if dropout_p > 0.0:
        attn_weights = nn.Dropout(dropout_p)(attn_weights)
    return mx.matmul(attn_weights, v)
