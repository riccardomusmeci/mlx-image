import math
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# RoPE-related functions:
def rope_rotate_half(x: mx.array) -> mx.array:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.split(2, axis=-1)
    return mx.concat([-x2, x1], axis=-1)

def rope_apply(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)

class RopePositionEmbedding(nn.Module):
    """
    RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
    Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
    """
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

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        # In MLX we simply store `periods` as a regular parameter array and avoid gradients.
        self.dtype = dtype or mx.float32
        self.periods = mx.zeros(shape=(D_head // 4,), dtype=self.dtype)
        self._init_weights()
        # Freeze periods so they are treated as non-trainable
        self.freeze(keys="periods", strict=False)

    def __call__(self, *, H: int, W: int) -> tuple[mx.array, mx.array]:
        dtype = self.dtype

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = mx.arange(0.5, H, dtype=dtype) / max_HW  # [H]
            coords_w = mx.arange(0.5, W, dtype=dtype) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = mx.arange(0.5, H, dtype=dtype) / min_HW  # [H]
            coords_w = mx.arange(0.5, W, dtype=dtype) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = mx.arange(0.5, H, dtype=dtype) / H  # [H]
            coords_w = mx.arange(0.5, W, dtype=dtype) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        mesh_h, mesh_w = mx.meshgrid(coords_h, coords_w, indexing="ij")
        coords = mx.stack([mesh_h, mesh_w], axis=-1)  # [H, W, 2]
        coords = coords.reshape(-1, coords.shape[-1])  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = mx.random.uniform(
                low=-self.shift_coords,
                high=self.shift_coords,
                shape=(2,),
                dtype=dtype,
            )
            coords = coords + shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = float(np.log(self.jitter_coords))
            jitter_min = -jitter_max
            jitter_hw = mx.random.uniform(
                low=jitter_min, high=jitter_max, shape=(2,), dtype=dtype
            )
            jitter_hw = mx.exp(jitter_hw)
            coords = coords * jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = float(np.log(self.rescale_coords))
            rescale_min = -rescale_max
            rescale_hw = mx.random.uniform(
                low=rescale_min, high=rescale_max, shape=(1,), dtype=dtype
            )
            rescale_hw = mx.exp(rescale_hw)
            coords = coords * rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.reshape(angles.shape[0], -1)  # [HW, D//2]
        angles = mx.tile(angles, (1, 2))  # [HW, D]
        cos = mx.cos(angles)  # [HW, D]
        sin = mx.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * mx.arange(self.D_head // 4, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = mx.linspace(0, 1, self.D_head // 4, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]

        self.periods = periods

if __name__ == "__main__":
    rope = RopePositionEmbedding(embed_dim=1024, num_heads=16)
    print(rope(H=224, W=224))
