import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm layer."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones(shape=(dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.square(x).mean(axis=-1, keepdims=True) + self.eps) * self.weight

    def reset_parameters(self):
        self.weight = mx.ones_like(self.weight)
