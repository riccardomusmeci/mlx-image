import math
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def rope_rotate_half(x: mx.array) -> mx.array:
    x1, x2 = x.split(2, axis=-1)
    return mx.concat([-x2, x1], axis=-1)


def rope_apply(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    return (x * cos) + (rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    """RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: mx.Dtype | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype or mx.float32
        self.periods = mx.zeros(shape=(D_head // 4,), dtype=self.dtype)
        self._init_weights()
        self.freeze(keys="periods", strict=False)

    def __call__(self, *, H: int, W: int) -> tuple[mx.array, mx.array]:
        dtype = self.dtype

        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = mx.arange(0.5, H, dtype=dtype) / max_HW
            coords_w = mx.arange(0.5, W, dtype=dtype) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = mx.arange(0.5, H, dtype=dtype) / min_HW
            coords_w = mx.arange(0.5, W, dtype=dtype) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = mx.arange(0.5, H, dtype=dtype) / H
            coords_w = mx.arange(0.5, W, dtype=dtype) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        mesh_h, mesh_w = mx.meshgrid(coords_h, coords_w, indexing="ij")
        coords = mx.stack([mesh_h, mesh_w], axis=-1)
        coords = coords.reshape(-1, coords.shape[-1])
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift_hw = mx.random.uniform(low=-self.shift_coords, high=self.shift_coords, shape=(2,), dtype=dtype)
            coords = coords + shift_hw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = float(np.log(self.jitter_coords))
            jitter_min = -jitter_max
            jitter_hw = mx.random.uniform(low=jitter_min, high=jitter_max, shape=(2,), dtype=dtype)
            jitter_hw = mx.exp(jitter_hw)
            coords = coords * jitter_hw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = float(np.log(self.rescale_coords))
            rescale_min = -rescale_max
            rescale_hw = mx.random.uniform(low=rescale_min, high=rescale_max, shape=(1,), dtype=dtype)
            rescale_hw = mx.exp(rescale_hw)
            coords = coords * rescale_hw

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.reshape(angles.shape[0], -1)
        angles = mx.tile(angles, (1, 2))
        cos = mx.cos(angles)
        sin = mx.sin(angles)

        return (sin, cos)

    def _init_weights(self):
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (2 * mx.arange(self.D_head // 4, dtype=dtype) / (self.D_head // 2))
        else:
            assert self.max_period is not None and self.min_period is not None
            base = self.max_period / self.min_period
            exponents = mx.linspace(0, 1, self.D_head // 4, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period

        self.periods = periods
