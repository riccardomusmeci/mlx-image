from functools import partial

import mlx.nn as nn

from .._config import HFWeights, Metrics, ModelConfig, Transform
from .regnet import BlockParams, RegNet

# TODO: waiting for groups and dilation support
# regnet_configs = {
#     "regnet_x_400mf": ModelConfig(
#         metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.74864, accuracy_at_5=0.92322),
#         transform=Transform(crop=224, resize=232),
#         weights=HFWeights(repo_id="mlx-vision/regnet_x_400mf-mlxim", filename=None),
#     )
# }


def regnet_y_400mf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )


def regnet_y_800mf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )


def regnet_x_400mf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs)
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )


def regnet_x_800mf(num_classes=100, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs)
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )
