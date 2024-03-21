from functools import partial

import mlx.nn as nn
from torchvision.models import swin_transformer

from .._config import HFWeights, Metrics, ModelConfig, Transform
from .swin_transformer import PatchMergingV2, SwinTransformer, SwinTransformerBlockV2

swin_configs = {
    "swin_tiny_patch4_window7_224": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80266, accuracy_at_5=0.94962),
        transform=Transform(img_size=224, crop_pct=224 / 232, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/swin_tiny_patch4_window7_224-mlxim", filename="model.safetensors"),
    ),
    "swin_small_patch4_window7_224": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81752, accuracy_at_5=0.95569),
        transform=Transform(img_size=224, crop_pct=224 / 246, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/swin_small_patch4_window7_224-mlxim", filename="model.safetensors"),
    ),
    "swin_base_patch4_window7_224": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.82172, accuracy_at_5=0.95717),
        transform=Transform(img_size=224, crop_pct=224 / 238, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/swin_base_patch4_window7_224-mlxim", filename="model.safetensors"),
    ),
    "swin_v2_tiny_patch4_window8_256": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80806, accuracy_at_5=0.95178),
        transform=Transform(img_size=256, crop_pct=256 / 260, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/swin_v2_tiny_patch4_window8_256-mlxim", filename="model.safetensors"),
    ),
    "swin_v2_small_patch4_window8_256": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81614, accuracy_at_5=0.95515),
        transform=Transform(img_size=256, crop_pct=256 / 260, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/swin_s2_small_patch4_window8_256-mlxim", filename="model.safetensors"),
    ),
    "swin_v2_base_patch4_window8_256": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81752, accuracy_at_5=0.95655),
        transform=Transform(img_size=256, crop_pct=256 / 272, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/swin_v2_base_patch4_window8_256-mlxim", filename="model.safetensors"),
    ),
}


def swin_tiny_patch4_window7_224(num_classes: int = 1000) -> SwinTransformer:
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
    )


def swin_small_patch4_window7_224(num_classes: int = 1000) -> SwinTransformer:
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        num_classes=num_classes,
    )


def swin_base_patch4_window7_224(num_classes: int = 1000) -> SwinTransformer:
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        num_classes=num_classes,
    )


def swin_v2_tiny_patch4_window8_256(num_classes: int = 1000) -> SwinTransformer:
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )


def swin_v2_base_patch4_window8_256(num_classes: int = 1000) -> SwinTransformer:
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )


def swin_v2_small_patch4_window8_256(num_classes: int = 1000) -> SwinTransformer:
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )
