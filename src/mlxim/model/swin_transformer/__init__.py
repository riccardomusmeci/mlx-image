from ._factory import (
    swin_base_patch4_window7_224,
    swin_configs,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
    swin_v2_base_patch4_window8_256,
    swin_v2_small_patch4_window8_256,
    swin_v2_tiny_patch4_window8_256,
)

__all__ = [
    "SWIN_ENTRYPOINT",
    "SWIN_CONFIG",
]

SWIN_ENTRYPOINT = {
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_v2_tiny_patch4_window8_256": swin_v2_tiny_patch4_window8_256,
    "swin_v2_small_patch4_window8_256": swin_v2_small_patch4_window8_256,
    "swin_v2_base_patch4_window8_256": swin_v2_base_patch4_window8_256,
}

SWIN_CONFIG = {
    "swin_tiny_patch4_window7_224": swin_configs["swin_tiny_patch4_window7_224"],
    "swin_small_patch4_window7_224": swin_configs["swin_small_patch4_window7_224"],
    "swin_base_patch4_window7_224": swin_configs["swin_base_patch4_window7_224"],
    "swin_v2_tiny_patch4_window8_256": swin_configs["swin_v2_tiny_patch4_window8_256"],
    "swin_v2_small_patch4_window8_256": swin_configs["swin_v2_small_patch4_window8_256"],
    "swin_v2_base_patch4_window8_256": swin_configs["swin_v2_base_patch4_window8_256"],
}
