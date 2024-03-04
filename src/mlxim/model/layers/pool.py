from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn


class AdaptiveAvgPool2d(nn.Module):
    """AdaptiveAvgPool2d module.

    Args:
        output_size (Union[int, Tuple[int, int]], optional): output size. Defaults to 1.
    """

    def __init__(self, output_size: Union[int, Tuple[int, int]] = 1) -> None:
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def __call__(self, x: mx.array) -> mx.array:
        """AdaptiveAvgPool2d call.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output array
        """
        B, H, W, C = x.shape
        x = x.reshape(
            B, H // self.output_size[0], self.output_size[0], W // self.output_size[1], self.output_size[1], C
        )
        x = mx.mean(x, axis=(1, 3))
        return x
