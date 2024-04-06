from functools import partial

import mlx.nn as nn

from .._config import HFWeights, Metrics, ModelConfig, Transform
from .vit import VisionTransformer

vit_configs = {
    "vit_base_patch16_224": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80634, accuracy_at_5=0.95012),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_base_patch16_224-mlxim", filename="model.safetensors"),
    ),
    "vit_base_patch16_224.swag_lin": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81904, accuracy_at_5=0.96169),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_base_patch16_224.swag_lin-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_base_patch32_224": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.7423, accuracy_at_5=0.91401),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_base_patch32_224-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_base_patch16_384.swag_e2e": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.85667, accuracy_at_5=0.97657),
        transform=Transform(img_size=384, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_base_patch16_384.swag_e2e-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_large_patch16_224": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.7934, accuracy_at_5=0.9427),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_large_patch16_224-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_large_patch16_224.swag_lin": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.8532, accuracy_at_5=0.9745),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_large_patch16_224.swag_lin-mlxim", filename="model.safetensors"),
    ),
    "vit_large_patch16_512.swag_e2e": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.88476, accuracy_at_5=0.98612),
        transform=Transform(img_size=512, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_large_patch16_512.swag_e2e-mlxim", filename="model.safetensors"),
    ),
    "vit_huge_patch14_224.swag_lin": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.86052, accuracy_at_5=0.97715),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_huge_patch14_224.swag_lin-mlxim", filename="model.safetensors"),
    ),
    "vit_huge_patch14_518.swag_e2e": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.89119, accuracy_at_5=0.98854),
        transform=Transform(img_size=518, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_huge_patch14_518.swag_e2e-mlxim", filename="model.safetensors"),
    ),
    "vit_small_patch14_518.dinov2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1k"),
        transform=Transform(img_size=518, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_small_patch14_518.dinov2-mlxim", filename="model.safetensors"),
    ),
    "vit_base_patch14_518.dinov2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1k"),
        transform=Transform(img_size=518, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_base_patch14_518.dinov2-mlxim", filename="model.safetensors"),
    ),
    "vit_large_patch14_518.dinov2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1k"),
        transform=Transform(img_size=518, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/vit_large_patch14_518.dinov2-mlxim", filename="model.safetensors"),
    ),
    "vit_small_patch16_224.dino": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_small_patch16_224.dino-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_small_patch8_224.dino": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_small_patch8_224.dino-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_base_patch16_224.dino": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_base_patch16_224.dino-mlxim",
            filename="model.safetensors",
        ),
    ),
    "vit_base_patch8_224.dino": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K"),
        transform=Transform(img_size=224, interpolation="bicubic"),
        weights=HFWeights(
            repo_id="mlx-vision/vit_base_patch8_224.dino-mlxim",
            filename="model.safetensors",
        ),
    ),
}


def vit_tiny_patch16_224(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_small_patch16_224(num_classes: int = 100) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_small_patch8_224(num_classes: int = 100) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=8,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_base_patch16_224(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_base_patch8_224(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=8,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_base_patch16_384(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=384,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_base_patch32_224(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_large_patch16_224(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_large_patch16_512(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=512,
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_huge_patch14_224(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_huge_patch14_518(num_classes: int = 1000) -> VisionTransformer:
    return VisionTransformer(
        image_size=518,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        num_classes=num_classes,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attn_bias=True,
    )


def vit_small_patch14_518_dinov2(num_classes: int = 1000) -> VisionTransformer:
    """ViT-S/14 for DINOv2"""
    return VisionTransformer(
        image_size=518,
        patch_size=14,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        num_classes=num_classes,
    )


def vit_base_patch14_518_dinov2(num_classes: int = 1000) -> VisionTransformer:
    """ViT-B/14 for DINOv2"""
    return VisionTransformer(
        image_size=518,
        patch_size=14,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=0,
    )


def vit_large_patch14_518_dinov2(num_classes: int = 1000) -> VisionTransformer:
    """ViT-L/14 for DINOv2"""
    return VisionTransformer(
        image_size=518,
        patch_size=14,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=num_classes,
    )
