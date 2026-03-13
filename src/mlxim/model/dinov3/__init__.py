from ._factory import *

__all__ = ["DINOV3_ENTRYPOINT", "DINOV3_CONFIG"]

DINOV3_ENTRYPOINT = {
    "vit_small_patch16_224.dinov3": vit_small_patch16_224_dinov3,
    "vit_small_plus_patch16_224.dinov3": vit_small_plus_patch16_224_dinov3,
    "vit_base_patch16_224.dinov3": vit_base_patch16_224_dinov3
}

DINOV3_CONFIG = {
    "vit_small_patch16_224.dinov3": dinov3_configs["vit_small_patch16_224.dinov3"],
    "vit_small_plus_patch16_224.dinov3": dinov3_configs["vit_small_plus_patch16_224.dinov3"],
    "vit_base_patch16_224.dinov3": dinov3_configs["vit_base_patch16_224.dinov3"]
}
