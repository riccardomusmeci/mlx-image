import math
from functools import partial
from typing import Any, Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlxim.model.layers.misc import StochasticDepth
from mlxim.model.layers.mlp import MLP
from mlxim.model.layers.pool import AdaptiveAvgPool2d

from ..layers import functional as F


def _patch_merging_pad(x: mx.array) -> mx.array:
    H, W, _ = x.shape[-3:]
    x = mx.pad(x, [(0, 0), (0, W % 2), (0, H % 2), (0, 0)])
    x0 = x[..., 0::2, 0::2, :]  # C, H/2, W/2
    x1 = x[..., 1::2, 0::2, :]  # C, H/2, W/2
    x2 = x[..., 0::2, 1::2, :]  # C, H/2, W/2
    x3 = x[..., 1::2, 1::2, :]  # C, H/2, W/2
    x = mx.concatenate([x0, x1, x2, x3], -1)  # H/2, W/2, 4*C
    return x


def _get_relative_position_bias(
    relative_position_bias_table: mx.array,
    relative_position_index: mx.array,
    window_size: List[int],
) -> mx.array:
    """Get relative position bias.

    Args:
        relative_position_bias_table (mx.array): relative position bias table.
        relative_position_index (mx.array): relative position index.
        window_size (List[int]): window size.

    Returns:
        mx.array: relative position bias.
    """
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.reshape(N, N, -1)
    relative_position_bias = relative_position_bias.transpose(2, 0, 1)[None]
    return relative_position_bias


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x (mx.array): input mx.array with expected layout of [..., H, W, C]

        Returns:
            mx.array: mx.array with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2, W/2, 2*C
        return x


class PatchMergingV2(nn.Module):
    """Patch Merging Layer for Swin Transformer V2.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # difference

    def __call__(self, x: mx.array):
        """
        Args:
            x (mx.array): input mx.array with expected layout of [..., H, W, C]
        Returns:
            mx.array: mx.array with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x


def shifted_window_attention(
    input: mx.array,
    qkv_weight: mx.array,
    proj_weight: mx.array,
    relative_position_bias: mx.array,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[mx.array] = None,
    proj_bias: Optional[mx.array] = None,
    logit_scale: Optional[mx.array] = None,
    training: bool = True,
) -> mx.array:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        input (mx.array[N, C, H, W]): The input mx.array or 4-dimensions.
        qkv_weight (mx.array[in_dim, out_dim]): The weight mx.array of query, key, value.
        proj_weight (mx.array[out_dim, out_dim]): The weight mx.array of projection.
        relative_position_bias (mx.array): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (mx.array[out_dim], optional): The bias mx.array of query, key, value. Default: None.
        proj_bias (mx.array[out_dim], optional): The bias mx.array of projection. Default: None.
        logit_scale (mx.array[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        mx.array[N, C, H, W]: The output mx.array after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = mx.pad(input, pad_width=[(0, 0), (0, pad_r), (0, pad_b), (0, 0)])
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = F.roll(x, shifts=(-shift_size[0], -shift_size[1]), axes=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.reshape(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        C,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias
        length = qkv_bias.size // 3
        qkv_bias[length : 2 * length] = 0
    qkv = mx.matmul(x, qkv_weight.T) + qkv_bias
    qkv = qkv.reshape(x.shape[0], x.shape[1], 3, num_heads, C // num_heads).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, axis=-1) @ F.normalize(k, axis=-1).transpose(0, 1, 3, 2)
        logit_scale = mx.clip(logit_scale, a_min=None, a_max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = mx.matmul(q, k.transpose(0, 1, 3, 2))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = mx.zeros((pad_H, pad_W), dtype=x.dtype)  # x.new_zeros((pad_H, pad_W))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.reshape(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        attn_mask = attn_mask.transpose(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask[:, None] - attn_mask[:, :, None]
        # attn_mask[attn_mask != 0] = -100.0
        attn_mask = mx.where(attn_mask != 0, -100.0, 0.0)
        attn = attn.reshape(x.shape[0] // num_windows, num_windows, num_heads, x.shape[1], x.shape[1])
        attn = attn + attn_mask[None, :, None, ...]
        attn = attn.reshape(-1, num_heads, x.shape[1], x.shape[1])

    attn = nn.softmax(attn, axis=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], C)
    x = mx.matmul(x, proj_weight.T) + proj_bias
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.reshape(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = F.roll(x, shifts=(shift_size[0], shift_size[1]), axes=(1, 2))  # type: ignore

    # unpad features
    x = x[:, :H, :W, :]
    return x


class ShiftedWindowAttention(nn.Module):
    """Shifted Window Attention for Swin Transformer.

    Args:
        dim (int): number of input channels
        window_size (List[int]): window size
        shift_size (List[int]): shift size for shifted window attention
        num_heads (int): number of attention heads
        qkv_bias (bool): if True, add bias to the query, key, value projection. Default: True
        proj_bias (bool): if True, add bias to the output projection. Default: True
        attention_dropout (float): attention dropout rate. Default: 0.0
        dropout (float): dropout rate. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self) -> None:
        """Define relative position bias table."""
        # define a parameter table of relative position bias
        self.relative_position_bias_table = mx.zeros(
            (
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.normal(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self) -> None:
        """Define relative position index."""
        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww

        # coords_flatten = np.flatten(coords, 1)  # 2, Wh*Ww
        coords_flatten = coords.reshape(coords.shape[0], -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.relative_position_index = mx.array(relative_position_index)

    def get_relative_position_bias(self) -> mx.array:
        """Get relative position bias.

        Returns:
            mx.array: relative position bias
        """
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size,  # type: ignore
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x (mx.array): mx.array with layout of [B, C, H, W]
        Returns:
            mx.array with same layout as input, i.e. [B, C, H, W]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """Shifted Window Attention for Swin Transformer V2.

    Args:
        dim (int): number of input channels
        window_size (List[int]): window size
        shift_size (List[int]): shift size for shifted window attention
        num_heads (int): number of attention heads
        qkv_bias (bool): if True, add bias to the query, key, value projection. Default: True
        proj_bias (bool): if True, add bias to the output projection. Default: True
        attention_dropout (float): attention dropout rate. Default: 0.0
        dropout (float): dropout rate. Default: 0.0
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = mx.log(10 * mx.ones((num_heads, 1, 1)))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, num_heads, bias=False),
        )
        if qkv_bias:
            length = self.qkv.bias.size // 3
            self.qkv.bias[length : 2 * length] = 0

    def define_relative_position_bias_table(self) -> None:
        """Define relative position bias table."""
        # get relative_coords_table
        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=np.float32)
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=np.float32)
        relative_coords_table = np.stack(np.meshgrid(relative_coords_h, relative_coords_w, indexing="ij"))
        relative_coords_table = relative_coords_table.transpose(1, 2, 0)[None]  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = np.sign(relative_coords_table) * np.log2(np.abs(relative_coords_table) + 1.0) / 3.0
        self.relative_coords_table = mx.array(relative_coords_table)

    def get_relative_position_bias(self) -> mx.array:
        """Get relative position bias.

        Returns:
            mx.array: relative position bias
        """
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads),
            self.relative_position_index,  # type: ignore
            self.window_size,
        )
        relative_position_bias = 16 * mx.sigmoid(relative_position_bias)
        return relative_position_bias

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x (mx.array): mx.array with layout of [B, C, H, W]
        Returns:
            mx.array with same layout as input, i.e. [B, C, H, W]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight)  # type: ignore
                if m.bias is not None:
                    nn.init.normal(m.bias, std=1e-6)  # type: ignore

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array with expected layout of [..., H, W, C]

        Returns:
            mx.array: output mx.array
        """
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockV2(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array with expected layout of [..., H, W, C]

        Returns:
            mx.array: output mx.array
        """
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                3,
                embed_dim,
                kernel_size=(patch_size[0], patch_size[1]),
                stride=(patch_size[0], patch_size[1]),
            ),
            nn.Identity(),
            norm_layer(embed_dim),
        )

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        if self.num_classes > 0:
            self.head = nn.Linear(num_features, num_classes)
        else:
            self.head = nn.Identity()  # type: ignore

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, std=0.02)  # type: ignore
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant(0)(m.bias)

    def get_features(self, x: mx.array) -> mx.array:
        """Get model features

        Args:
            x (mx.array): mx.array

        Returns:
            mx.array: features
        """
        x = self.patch_embed(x)
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = mx.flatten(x, start_axis=1)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): mx.array

        Returns:
            mx.array: mx.array
        """
        x = self.get_features(x)
        x = self.head(x)
        return x
