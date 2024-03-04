from .resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet_configs,
    wide_resnet50_2,
    wide_resnet101_2,
)

# TODO: waiting for groups and dilation support
# from .regnet import (
#     regnet_x_400mf,
#     regnet_configs
# )

MODEL_ENTRYPOINT = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
    # TODO: waiting for groups and dilation support
    # "resnext50_32x4d": resnext50_32x4d,
    # "resnext101_32x8d": resnext101_32x8d,
    # "resnext101_64x4d": resnext101_64x4d,
    # "regnet_x_400mf": regnet_x_400mf
}

MODEL_CONFIG = {
    "resnet18": resnet_configs["resnet18"],
    "resnet34": resnet_configs["resnet34"],
    "resnet50": resnet_configs["resnet50"],
    "resnet101": resnet_configs["resnet101"],
    "resnet152": resnet_configs["resnet152"],
    "wide_resnet50_2": resnet_configs["wide_resnet50_2"],
    "wide_resnet101_2": resnet_configs["wide_resnet101_2"],
    # TODO: waiting for groups and dilation support
    # "resnext50_32x4d": resnet_configs["resnext50_32x4d"],
    # "resnext101_32x8d": resnet_configs["resnext101_32x8d"],
    # "resnext101_64x4d": resnet_configs["resnext101_64x4d"],
    # "regnet_x_400mf": regnet_configs["regnet_x_400mf"]
}
