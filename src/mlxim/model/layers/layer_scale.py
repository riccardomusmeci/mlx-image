import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    """Layer Scale layer."""

    def __init__(
        self,
        dim: int,
        init_values: float | mx.array = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = mx.full((dim,), init_values, dtype=mx.float32)
        self.init_values = init_values

    def reset_parameters(self):
        self.gamma = mx.full(self.gamma.shape, self.init_values, dtype=self.gamma.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma
