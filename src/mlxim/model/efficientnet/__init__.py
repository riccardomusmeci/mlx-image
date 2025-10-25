from ._blocks import MBConv, MBConvConfig
from ._factory import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_configs,
)
from .efficientnet import EfficientNet

__all__ = [
    "EfficientNet",
    "MBConv",
    "MBConvConfig",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "EFFICIENTNET_ENTRYPOINT",
    "EFFICIENTNET_CONFIG",
]

EFFICIENTNET_ENTRYPOINT = {
    "efficientnet_b0": efficientnet_b0,
    "efficientnet_b1": efficientnet_b1,
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b3": efficientnet_b3,
    "efficientnet_b4": efficientnet_b4,
    "efficientnet_b5": efficientnet_b5,
    "efficientnet_b6": efficientnet_b6,
    "efficientnet_b7": efficientnet_b7,
}

EFFICIENTNET_CONFIG = {
    "efficientnet_b0": efficientnet_configs["efficientnet_b0"],
    "efficientnet_b1": efficientnet_configs["efficientnet_b1"],
    "efficientnet_b2": efficientnet_configs["efficientnet_b2"],
    "efficientnet_b3": efficientnet_configs["efficientnet_b3"],
    "efficientnet_b4": efficientnet_configs["efficientnet_b4"],
    "efficientnet_b5": efficientnet_configs["efficientnet_b5"],
    "efficientnet_b6": efficientnet_configs["efficientnet_b6"],
    "efficientnet_b7": efficientnet_configs["efficientnet_b7"],
}
