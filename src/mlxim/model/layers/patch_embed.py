import math

import mlx.core as mx
import mlx.nn as nn

from .utils import to_2tuple


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,H,W,C) -> (B,N,D)"""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = to_2tuple(img_size)
        patch_HW = to_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_HW,
            stride=patch_HW,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        # Handle NCHW input by transposing to NHWC
        if x.ndim == 4 and x.shape[1] == self.in_chans and x.shape[-1] != self.in_chans:
            x = x.transpose(0, 2, 3, 1)

        B, H, W, C = x.shape
        x = self.proj(x)
        B, H_out, W_out, C_out = x.shape
        x = x.reshape(B, H_out * W_out, C_out)
        x = self.norm(x)

        if not self.flatten_embedding:
            x = x.reshape(B, H_out, W_out, self.embed_dim)

        return x

    def reset_parameters(self):
        fan_in = self.in_chans * self.patch_size[0] * self.patch_size[1]
        std = 1.0 / math.sqrt(fan_in)
        self.proj.weight = mx.random.uniform(
            low=-std,
            high=std,
            shape=self.proj.weight.shape,
            dtype=self.proj.weight.dtype,
        )
        if "bias" in self.proj:
            self.proj.bias = mx.random.uniform(
                low=-std,
                high=std,
                shape=self.proj.bias.shape,
                dtype=self.proj.bias.dtype,
            )
