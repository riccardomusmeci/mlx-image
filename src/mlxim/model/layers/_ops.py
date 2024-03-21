from typing import Optional
import mlx.core as mx


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


def stochastic_depth(x: mx.array, p: float, mode: str, training: bool = True) -> mx.array:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        x (mx.array): input array
        p (float): probability of the x to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire x, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        mx.array: the randomly  0-tensor
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return x

    survival_rate = 1.0 - p
    if mode == "row":
        size = [x.shape[0]] + [1] * (x.ndim - 1)
    else:
        size = [1] * x.ndim
    noise = mx.random.bernoulli(survival_rate, size)
    if survival_rate > 0.0:
        noise = noise / survival_rate
    return x * noise
