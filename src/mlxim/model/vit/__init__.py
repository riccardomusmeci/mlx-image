from ._factory import *

__all__ = [
    "VIT_ENTRYPOINT",
    "VIT_CONFIG",
]

VIT_ENTRYPOINT = {
    "vit_base_patch16_224": vit_base_patch16_224,
    "vit_base_patch16_224.swag_lin": vit_base_patch16_224,
    "vit_base_patch32_224": vit_base_patch32_224,
    "vit_base_patch16_384.swag_e2e": vit_base_patch16_384,
    "vit_large_patch16_224": vit_large_patch16_224,
    "vit_large_patch16_224.swag_lin": vit_large_patch16_224,
    "vit_large_patch16_512.swag_e2e": vit_large_patch16_512,
    "vit_huge_patch14_224.swag_lin": vit_huge_patch14_224,
    "vit_huge_patch14_518.swag_e2e": vit_huge_patch14_518,
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
}
