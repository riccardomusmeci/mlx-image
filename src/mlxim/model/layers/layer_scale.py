

from typing import Union

import mlx.core as mx
import mlx.nn as nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, mx.array] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = mx.zeros(shape=(dim,), dtype=mx.float32)
        self.init_values = init_values

    def reset_parameters(self):
        mx.fill_(self.gamma, self.init_values)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma
