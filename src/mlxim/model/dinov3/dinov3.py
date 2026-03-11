import logging
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

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

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-5),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": nn.RMSNorm,
}

# Map string names to MLX dtypes.
dtype_dict: Dict[str, mx.Dtype] = {
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    """
    Initialization helper adapted for MLX modules.
    """
    if isinstance(module, nn.Linear):
        # Truncated normal is approximated with a plain normal here.
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
        # MLX LayerNorm exposes reset_parameters
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if isinstance(module, LayerScale):
        module.reset_parameters()

    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    # TODO: incase the the nn.RMSNorm works remove
    # if isinstance(module, RMSNorm):
    #     # For RMSNorm we simply reset the scale to ones.
    #     if hasattr(module, "reset_parameters"):
    #         module.reset_parameters()


class SelfAttentionBlock(nn.Module):
    """Self-Attention block.
    Args:
        dim: Dimension of the input features.
        num_heads: Number of attention heads.
        ffn_ratio: Ratio of FFN hidden dimension to embedding dimension.
        qkv_bias: Whether to use bias for QKV projection.
        proj_bias: Whether to use bias for output projection.
        ffn_bias: Whether to use bias for FFN.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        init_values: Initial values for LayerScale.
        drop_path: Drop path rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        attn_class: Attention class.
        ffn_layer: FFN class.
        mask_k_bias: Whether to use bias for mask K.
    """

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
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            return sin, cos

    def _forward(self, x: mx.array, rope=None) -> mx.array:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """

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

    def _forward_list(self, x_list: List[mx.array], rope_list=None) -> List[mx.array]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                _randperm(b)[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                x.at[indices_1].add(self.ls1(residual_1) * residual_scale_factor)
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                _randperm(b)[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                x_attn.at[indices_2].add(self.ls2(residual_2) * residual_scale_factor)
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def __call__(
        self,
        x_or_x_list: mx.array | List[mx.array],
        rope_or_rope_list: mx.array | List[mx.array] | None = None,
    ) -> mx.array | List[mx.array]:
        """
        Args:
            x_or_x_list: Input array or list of input arrays.
            rope_or_rope_list: RoPE embeddings or list of RoPE embeddings.
        Returns:
            Output array or list of output arrays.
        """
        if isinstance(x_or_x_list, mx.array):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    """Causal Self-Attention block.
    Args:
        dim: Dimension of the input features.
        num_heads: Number of attention heads.
        ffn_ratio: Ratio of FFN hidden dimension to embedding dimension.
        ls_init_value: Initial values for LayerScale.
        is_causal: Whether to use causal attention.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        dropout_prob: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
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
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        """
        Keep MLX's default parameter initialization; this method is
        retained only for API compatibility with the original code.
        """
        _ = init_attn_std
        _ = init_proj_std
        _ = init_fc_std
        _ = factor

    def __call__(
        self,
        x: mx.array,
    ):
        """
        Args:
            x: Input array.
        Returns:
            Output array.
        """
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


class DinoVisionTransformer(nn.Module):
    """Dino Vision Transformer.
    Args:
        img_size: Size of the input image.
        patch_size: Size of the patches.
        in_chans: Number of input channels.
        pos_embed_rope_base: Base for RoPE embeddings.
        pos_embed_rope_min_period: Minimum period for RoPE embeddings.
        pos_embed_rope_max_period: Maximum period for RoPE embeddings.
        pos_embed_rope_normalize_coords: Normalization method for RoPE embeddings.
        pos_embed_rope_shift_coords: Shift for RoPE embeddings.
        pos_embed_rope_jitter_coords: Jitter for RoPE embeddings.
        pos_embed_rope_rescale_coords: Rescale for RoPE embeddings.
        pos_embed_rope_dtype: Data type for RoPE embeddings.
        embed_dim: Dimension of the embedding.
        depth: Depth of the transformer.
        num_heads: Number of attention heads.
        ffn_ratio: Ratio of FFN hidden dimension to embedding dimension.
        qkv_bias: Whether to use bias for QKV projection.
        proj_bias: Whether to use bias for output projection.
        ffn_bias: Whether to use bias for FFN.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        init_values: Initial values for LayerScale.
        drop_path: Drop path rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        attn_class: Attention class.
        ffn_layer: FFN class.
        mask_k_bias: Whether to use bias for mask K.
    """

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

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
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

        # CLS, storage, and mask tokens are plain MLX arrays treated as parameters.
        self.cls_token = mx.zeros((1, 1, embed_dim), dtype=mx.float32)
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = mx.zeros((1, n_storage_tokens, embed_dim), dtype=mx.float32)
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
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
        logger.info(f"using {ffn_layer} layer as FFN")
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

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = mx.zeros((1, embed_dim), dtype=mx.float32)

    def init_weights(self):
        """Initialize the weights."""
        # Reinitialize RoPE periods and token parameters.
        self.rope_embed._init_weights()

        std = 0.02
        self.cls_token = mx.random.normal(
            shape=self.cls_token.shape,
            dtype=self.cls_token.dtype,
            loc=0.0,
            scale=std,
        )
        if self.n_storage_tokens > 0 and self.storage_tokens is not None:
            self.storage_tokens = mx.random.normal(
                shape=self.storage_tokens.shape,
                dtype=self.storage_tokens.dtype,
                loc=0.0,
                scale=std,
            )
        self.mask_token = mx.zeros_like(self.mask_token)

        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: mx.array, masks: Optional[mx.array] = None
    ) -> Tuple[mx.array, Tuple[int, int]]:
        """
        Prepares tokens with masks.
        Args:
            x: Input array.
            masks: Mask array.
        Returns:
            Tuple of output array and shape.
        """
        x = self.patch_embed(x)  # [B, H, W, D]
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)  # [B, HW, D]

        if masks is not None:
            # masks: [B, HW] -> [B, HW, 1]
            cond = mx.expand_dims(masks.astype(bool), axis=-1)
            mask_token = self.mask_token.astype(x.dtype)  # [1, D]
            mask_token = mx.expand_dims(mask_token, axis=0)  # [1, 1, D]
            x = mx.where(cond, mask_token, x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token

        if self.n_storage_tokens > 0 and self.storage_tokens is not None:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = mx.zeros((1, 0, cls_token.shape[-1]), dtype=cls_token.dtype)

        # Expand CLS and storage tokens across the batch and concatenate.
        cls_tokens = mx.repeat(cls_token, repeats=B, axis=0)
        storage_tokens_exp = mx.repeat(storage_tokens, repeats=B, axis=0)

        x = mx.concat(
            [
                cls_tokens,
                storage_tokens_exp,
                x,
            ],
            axis=1,
        )

        return x, (H, W)

    def forward_features_list(
        self, x_list: List[mx.array], masks_list: List[Optional[mx.array]]
    ) -> List[Dict[str, mx.array]]:
        """
        Forward pass for a list of inputs.
        Args:
            x_list: List of input arrays.
            masks_list: List of mask arrays.
        Returns:
            List of output dictionaries.
        """
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output: List[Dict[str, mx.array]] = []
        for idx, (x_val, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x_val[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
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
        x: mx.array | List[mx.array],
        masks: Optional[mx.array | List[Optional[mx.array]]] = None,
    ) -> Union[Dict[str, mx.array], List[Dict[str, mx.array]]]:
        """
        Forward pass for a list of inputs.
        Args:
            x: Input array or list of input arrays.
            masks: Mask array or list of mask arrays.
        Returns:
            Dictionary of output arrays or list of dictionaries.
        """
        if isinstance(x, mx.array):
            return self.forward_features_list([x], [masks])[0]
        else:
            if masks is None:
                masks = [None] * len(x)
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: mx.array, n: int = 1) -> List[mx.array]:
        """
        Get intermediate layers.
        Args:
            x: Input array.
            n: Number of layers to take.
        Returns:
            List of output arrays.
        """
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
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
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[mx.array, Tuple[mx.array, ...]]]:
        """
        Get intermediate layers.
        Args:
            x: Input array.
            n: Number of layers to take.
            reshape: Whether to reshape the output.
            return_class_token: Whether to return the class token.
            return_extra_tokens: Whether to return extra tokens.
            norm: Whether to normalize the output.
        Returns:
            Tuple of output arrays.
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(mx.concat((x_norm_cls_reg, x_norm_patch), axis=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                mx.transpose(
                    out.reshape(B, h // self.patch_size, w // self.patch_size, -1),
                    (0, 3, 1, 2),
                )
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def __call__(self, *args, is_training: bool = False, **kwargs) -> Union[List[Dict[str, mx.array]], mx.array]:
        """
        Forward pass for a list of inputs.
        Args:
            x: Input array or list of input arrays.
            masks: Mask array or list of mask arrays.
            is_training: Whether the model is in training mode.
            **kwargs: Additional keyword arguments.
        Returns:
            Dictionary of output arrays or list of dictionaries.
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(**kwargs) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    model = vit_small(patch_size=16)
    model.init_weights()
    x = mx.random.uniform(shape=(1, 3, 224, 224))
    print(model(x, is_training=True))
    print(model)
