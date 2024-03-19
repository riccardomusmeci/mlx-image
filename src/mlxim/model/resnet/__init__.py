from ._factory import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet_configs,
    wide_resnet50_2,
    wide_resnet101_2,
)

__all__ = [
    "RESNET_ENTRYPOINT",
    "RESNET_CONFIG",
]

RESNET_ENTRYPOINT = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}

RESNET_CONFIG = {
    "resnet18": resnet_configs["resnet18"],
    "resnet34": resnet_configs["resnet34"],
    "resnet50": resnet_configs["resnet50"],
    "resnet101": resnet_configs["resnet101"],
    "resnet152": resnet_configs["resnet152"],
    "wide_resnet50_2": resnet_configs["wide_resnet50_2"],
    "wide_resnet101_2": resnet_configs["wide_resnet101_2"],
}
