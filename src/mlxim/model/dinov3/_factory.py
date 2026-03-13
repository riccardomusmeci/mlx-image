import mlx.nn as nn

from .._config import HFWeights, Metrics, ModelConfig, Transform
from .dinov3 import DinoVisionTransformer

dinov3_configs = {
    "vit_small_patch16_224.dinov3": ModelConfig(
        metrics=Metrics(dataset="LVD-1689M"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_small_patch16_224.dinov3-mlxim", filename="model.safetensors"),
    ),
    "vit_small_plus_patch16_224.dinov3": ModelConfig(
        metrics=Metrics(dataset="LVD-1689M"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_small_plus_patch16_224.dinov3-mlxim", filename="model.safetensors"),
    ),
    "vit_base_patch16_224.dinov3": ModelConfig(
        metrics=Metrics(dataset="LVD-1689M"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_base_patch16_224.dinov3-mlxim", filename="model.safetensors"),
    ),
}


def vit_small_patch16_224_dinov3(num_classes: int = 0, **kwargs) -> DinoVisionTransformer:
    return DinoVisionTransformer(
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


def vit_small_plus_patch16_224_dinov3(num_classes: int = 0, **kwargs) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        ffn_layer="swiglu",
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
        **kwargs,
    )


def vit_base_patch16_224_dinov3(num_classes: int = 0, **kwargs) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
        **kwargs,
    )
