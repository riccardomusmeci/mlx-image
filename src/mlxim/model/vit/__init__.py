from functools import partial

from ._factory import *

__all__ = [
    "VIT_ENTRYPOINT",
    "VIT_CONFIG",
]

VIT_ENTRYPOINT = {
    "vit_base_patch16_224": vit_base_patch16_224,
    "vit_base_patch16_224.swag_lin": vit_base_patch16_224,
    "vit_base_patch16_224.dino": vit_base_patch16_224,
    "vit_base_patch32_224": vit_base_patch32_224,
    "vit_base_patch16_384.swag_e2e": vit_base_patch16_384,
    "vit_large_patch16_224": vit_large_patch16_224,
    "vit_large_patch16_224.swag_lin": vit_large_patch16_224,
    "vit_large_patch16_512.swag_e2e": vit_large_patch16_512,
    "vit_huge_patch14_224.swag_lin": vit_huge_patch14_224,
    "vit_huge_patch14_518.swag_e2e": vit_huge_patch14_518,
    "vit_small_patch14_518.dinov2": vit_small_patch14_518_dinov2,
    "vit_base_patch14_518.dinov2": vit_base_patch14_518_dinov2,
    "vit_large_patch14_518.dinov2": vit_large_patch14_518_dinov2,
    "vit_small_patch16_224.dino": vit_small_patch16_224,
    "vit_small_patch8_224.dino": vit_small_patch8_224,
    "vit_base_patch8_224.dino": vit_base_patch8_224,
}

VIT_CONFIG = {
    "vit_base_patch16_224": vit_configs["vit_base_patch16_224"],
    "vit_base_patch16_224.swag_lin": vit_configs["vit_base_patch16_224.swag_lin"],
    "vit_base_patch32_224": vit_configs["vit_base_patch32_224"],
    "vit_base_patch16_384.swag_e2e": vit_configs["vit_base_patch16_384.swag_e2e"],
    "vit_large_patch16_224": vit_configs["vit_large_patch16_224"],
    "vit_large_patch16_224.swag_lin": vit_configs["vit_large_patch16_224.swag_lin"],
    "vit_large_patch16_512.swag_e2e": vit_configs["vit_large_patch16_512.swag_e2e"],
    "vit_huge_patch14_224.swag_lin": vit_configs["vit_huge_patch14_224.swag_lin"],
    "vit_huge_patch14_518.swag_e2e": vit_configs["vit_huge_patch14_518.swag_e2e"],
    "vit_small_patch14_518.dinov2": vit_configs["vit_small_patch14_518.dinov2"],
    "vit_base_patch14_518.dinov2": vit_configs["vit_base_patch14_518.dinov2"],
    "vit_large_patch14_518.dinov2": vit_configs["vit_large_patch14_518.dinov2"],
    "vit_small_patch16_224.dino": vit_configs["vit_small_patch16_224.dino"],
    "vit_small_patch8_224.dino": vit_configs["vit_small_patch8_224.dino"],
    "vit_base_patch16_224.dino": vit_configs["vit_base_patch16_224.dino"],
    "vit_base_patch8_224.dino": vit_configs["vit_base_patch8_224.dino"],
}
