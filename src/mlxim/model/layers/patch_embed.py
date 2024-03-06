# from typing import Optional, Callable

# import mlx.nn as nn
# import mlx.core as mx

# from ..layers.utils import to_2tuple


# class PatchEmbed(nn.Module):
#     """2D Image to Patch Embedding

#     Args:
#         img_size (int, tuple): input image size. Default: 224
#         patch_size (int, tuple): patch size. Default: 16
#         in_chans (int): number of input channels. Default: 3
#         embed_dim (int): number of linear projection output channels. Default: 768
#         norm_layer (callable): normalization layer. Default: None
#         flatten (bool): whether to flatten the output tensor. Default: True
#         output_fmt (str): output format. Default: None
#         bias (bool): whether to use bias in the projection layer. Default: True
#         strict_img_size (bool): whether to strictly enforce the input image size. Default: True
#         dynamic_img_pad (bool): whether to dynamically pad the input image. Default: False
#     """

#     def __init__(
#         self,
#         img_size: Optional[int] = 224,
#         patch_size: int = 16,
#         in_chans: int = 3,
#         embed_dim: int = 768,
#         norm_layer: Optional[Callable] = None,
#         flatten: bool = True,
#         # output_fmt: Optional[str] = None,
#         bias: bool = True,
#         strict_img_size: bool = True,
#         dynamic_img_pad: bool = False,
#     ):
#         super().__init__()
#         self.patch_size = to_2tuple(patch_size)
#         if img_size is not None:
#             self.img_size = to_2tuple(img_size)
#             self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
#             self.num_patches = self.grid_size[0] * self.grid_size[1]
#         else:
#             self.img_size = None  # type: ignore
#             self.grid_size = None  # type: ignore
#             self.num_patches = None  # type: ignore

#         # if output_fmt is not None:
#         #     self.flatten = False
#         #     self.output_fmt = Format(output_fmt)
#         # else:
#         #     # flatten spatial dim and transpose to channels last, kept for bwd compat
#         #     self.flatten = flatten
#         #     self.output_fmt = Format.NCHW
#         self.strict_img_size = strict_img_size
#         self.dynamic_img_pad = dynamic_img_pad
#         self.embed_dim = embed_dim
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def __call__(self, x: mx.array) -> mx.array:
#         B, H, W, C = x.shape
#         if self.img_size is not None:
#             if self.strict_img_size:
#                 assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
#                 assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
#             elif not self.dynamic_img_pad:
#                 assert (
#                     H % self.patch_size[0] == 0
#                 ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
#                 assert (
#                     W % self.patch_size[1] == 0
#                 ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
#         if self.dynamic_img_pad:
#             pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
#             pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
#             x = F.pad(x, (0, pad_w, 0, pad_h))
#         x = self.proj(x)
#         x = x.reshape(N, H * W, self.embed_dim)
#         # if self.flatten:
#         #     x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
#         # elif self.output_fmt != Format.NCHW:
#         #     x = nchw_to(x, self.output_fmt)
#         x = self.norm(x)
#         return x
