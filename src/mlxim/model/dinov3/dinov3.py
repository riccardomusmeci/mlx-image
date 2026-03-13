import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn

from ..layers import (
    CausalSelfAttention,
    LayerScale,
    PatchEmbed,
    RopePositionEmbedding,
    RoPESelfAttention,
    _randperm,
    cat_keep_shapes,
    named_apply,
    uncat_with_shapes,
)
from .ffn_layers import Mlp, SwiGLUFFN

logger = logging.getLogger("dinov3")

ffn_layer_dict: dict[str, type | partial] = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict: dict[str, type | partial] = {
    "layernorm": partial(nn.LayerNorm, eps=1e-5),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": nn.RMSNorm,
}

dtype_dict: dict[str, mx.Dtype] = {
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        std = 0.02
        module.weight = mx.random.normal(
            shape=module.weight.shape,
            dtype=module.weight.dtype,
            loc=0.0,
            scale=std,
        )
        if "bias" in module:
            module.bias = mx.zeros_like(module.bias)

    if isinstance(module, nn.LayerNorm):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if isinstance(module, LayerScale):
        module.reset_parameters()

    if isinstance(module, PatchEmbed):
        module.reset_parameters()


class SelfAttentionBlock(nn.Module):
    """Self-Attention block for DINOv3."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = RoPESelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(
        rope: tuple[mx.array, mx.array] | None, indices: mx.array
    ) -> tuple[mx.array, mx.array] | None:
        if rope is None:
            return None
        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        else:
            return sin, cos

    def _forward(self, x: mx.array, rope=None) -> mx.array:
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = _randperm(b)[:sample_subset_size]
            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)
            x_attn = x.at[indices_1].add(self.ls1(residual_1) * residual_scale_factor)

            indices_2 = _randperm(b)[:sample_subset_size]
            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))
            x_ffn = x_attn.at[indices_2].add(self.ls2(residual_2) * residual_scale_factor)
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: list[mx.array], rope_list: list | None = None) -> list[mx.array]:
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / s for b, s in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [_randperm(b)[:s] for _x, b, s in zip(x_list, b_list, sample_subset_sizes)]
            x_subset_1_list = [x[idx] for x, idx in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [self._maybe_index_rope(r, idx) for r, idx in zip(rope_list, indices_1_list)]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                x.at[idx].add(self.ls1(res) * scale)
                for x, res, idx, scale in zip(x_list, residual_1_list, indices_1_list, residual_scale_factors)
            ]

            indices_2_list = [_randperm(b)[:s] for _x, b, s in zip(x_list, b_list, sample_subset_sizes)]
            x_subset_2_list = [x[idx] for x, idx in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)
            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                xa.at[idx].add(self.ls2(res) * scale)
                for xa, res, idx, scale in zip(x_attn_list, residual_2_list, indices_2_list, residual_scale_factors)
            ]
        else:
            x_out = []
            rl = rope_list if rope_list is not None else [None] * len(x_list)
            for x, rope in zip(x_list, rl):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn_val = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn_val)
            x_ffn = x_out

        return x_ffn

    def __call__(
        self,
        x_or_x_list: mx.array | list[mx.array],
        rope_or_rope_list: list | None = None,
    ) -> mx.array | list[mx.array]:
        if isinstance(x_or_x_list, mx.array):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    """Causal Self-Attention block for DINOv3."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: float | None = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)
        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(in_features=dim, hidden_features=ffn_hidden_dim, drop=dropout_prob, act_layer=act_layer)
        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


class DinoVisionTransformer(nn.Module):
    """DINOv3 Vision Transformer with RoPE attention."""

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = mx.zeros((1, 1, embed_dim), dtype=mx.float32)
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = mx.zeros((1, n_storage_tokens, embed_dim), dtype=mx.float32)

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
        )

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = blocks_list
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None

        self.head = nn.Identity()
        self.mask_token = mx.zeros((1, embed_dim), dtype=mx.float32)

    def prepare_tokens_with_masks(self, x: mx.array, masks: mx.array | None = None) -> tuple[mx.array, tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)

        if masks is not None:
            cond = mx.expand_dims(masks.astype(mx.bool_), axis=-1)
            mask_token = self.mask_token.astype(x.dtype)
            mask_token = mx.expand_dims(mask_token, axis=0)
            x = mx.where(cond, mask_token, x)

        cls_token = self.cls_token

        if self.n_storage_tokens > 0 and self.storage_tokens is not None:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = mx.zeros((1, 0, cls_token.shape[-1]), dtype=cls_token.dtype)

        cls_tokens = mx.repeat(cls_token, repeats=B, axis=0)
        storage_tokens_exp = mx.repeat(storage_tokens, repeats=B, axis=0)

        x = mx.concat([cls_tokens, storage_tokens_exp, x], axis=1)
        return x, (H, W)

    def forward_features_list(
        self, x_list: list[mx.array], masks_list: list[mx.array | None]
    ) -> list[dict[str, mx.array]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for _r in rope]
            x = blk(x, rope_sincos)

        output: list[dict[str, mx.array]] = []
        for idx, (x_val, masks) in enumerate(zip(x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if (
                    self.untie_global_and_local_cls_norm
                    and self.training
                    and idx == 1
                    and self.local_cls_norm is not None
                ):
                    x_norm_cls_reg = self.local_cls_norm(x_val[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms and self.cls_norm is not None:
                    x_norm_cls_reg = self.cls_norm(x_val[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x_val[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x_val[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x_val)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x_val,
                    "masks": masks,
                }
            )
        return output

    def forward_features(
        self,
        x: mx.array | list[mx.array],
        masks: mx.array | list[mx.array | None] | None = None,
    ) -> dict[str, mx.array] | list[dict[str, mx.array]]:
        if isinstance(x, mx.array):
            masks_list: list[mx.array | None] = [masks if isinstance(masks, mx.array) else None]
            return self.forward_features_list([x], masks_list)[0]
        else:
            masks_resolved: list[mx.array | None] = list(masks) if isinstance(masks, list) else [None] * len(x)
            return self.forward_features_list(x, masks_resolved)

    def _get_intermediate_layers_not_chunked(self, x: mx.array, n: int | Sequence = 1) -> list[mx.array]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: mx.array,
        *,
        n: int | Sequence = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms and self.cls_norm is not None:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(mx.concat([x_norm_cls_reg, x_norm_patch], axis=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                mx.transpose(out.reshape(B, h // self.patch_size, w // self.patch_size, -1), (0, 3, 1, 2))
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        else:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def get_features(self, x: mx.array) -> mx.array:
        """Get CLS token features."""
        ret = self.forward_features(x)
        assert isinstance(ret, dict)
        return ret["x_norm_clstoken"]

    def __call__(
        self, x: mx.array, is_training: bool = False, **kwargs
    ) -> dict[str, mx.array] | list[dict[str, mx.array]] | mx.array:
        ret = self.forward_features(x, **kwargs)
        if is_training:
            return ret
        assert isinstance(ret, dict)
        return self.head(ret["x_norm_clstoken"])
