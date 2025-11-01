from .mobilenetv2 import InvertedResidual, MobileNetV2
from .mobilenetv3 import InvertedResidualConfig, MobileNetV3
from ._factory import mobilenet_configs, mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

__all__ = [
    "MobileNetV2",
    "MobileNetV3",
    "InvertedResidual",
    "InvertedResidualConfig",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "MOBILENET_ENTRYPOINT",
    "MOBILENET_CONFIG",
]

MOBILENET_ENTRYPOINT = {
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_large": mobilenet_v3_large,
    "mobilenet_v3_small": mobilenet_v3_small,
}

MOBILENET_CONFIG = {
    "mobilenet_v2": mobilenet_configs["mobilenet_v2"],
    "mobilenet_v3_large": mobilenet_configs["mobilenet_v3_large"],
    "mobilenet_v3_small": mobilenet_configs["mobilenet_v3_small"],
}
