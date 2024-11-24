from functools import partial

import mlx.nn as nn

from .._config import HFWeights, Metrics, ModelConfig, Transform
from .regnet import BlockParams, RegNet

regnet_configs = {
    "regnet_x_400mf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.71674, accuracy_at_5=0.90284),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_400mf-mlxim", filename="model.safetensors"),
    ),
    "regnet_x_800mf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.70968, accuracy_at_5=0.90024),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_800mf-mlxim", filename="model.safetensors"),
    ),
    "regnet_x_1_6gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.77248, accuracy_at_5=0.93672),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_1_6gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_x_3_2gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.79072, accuracy_at_5=0.94644),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_3_2gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_x_8gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.79712, accuracy_at_5=0.94582),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_8gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_x_16gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80682, accuracy_at_5=0.95276),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_16gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_x_32gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=None, accuracy_at_5=None),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_x_32gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_400mf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.73756, accuracy_at_5=0.91756),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_400mf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_800mf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.76808, accuracy_at_5=0.93476),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_800mf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_1_6gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.79128, accuracy_at_5=0.94762),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_1_6gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_3_2gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80348, accuracy_at_5=0.95276),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_3_2gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_8gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81056, accuracy_at_5=0.9562),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_8gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_16gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81134, accuracy_at_5=0.95524),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_16gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_32gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.78342, accuracy_at_5=0.93986),
        transform=Transform(img_size=232, crop_pct=224/232),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_32gf-mlxim", filename="model.safetensors"),
    ),
    "regnet_y_128gf": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=None, accuracy_at_5=None),
        transform=Transform(img_size=384),
        weights=HFWeights(repo_id="mlx-vision/regnet_y_128gf-mlxim", filename="model.safetensors"),
    )
}

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
    
def regnet_y_1_6gf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )

def regnet_y_3_2gf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )
    
def regnet_y_8gf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )

def regnet_y_16gf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )

def regnet_y_32gf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )

def regnet_y_128gf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )

def regnet_x_400mf(num_classes: int = 1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )


def regnet_x_800mf(num_classes=1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )
    
def regnet_x_1_6gf(num_classes=1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )
    
def regnet_x_3_2gf(num_classes=1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )

def regnet_x_8gf(num_classes=1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )
    
def regnet_x_16gf(num_classes=1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )
    
def regnet_x_32gf(num_classes=1000, **kwargs) -> RegNet:  # type: ignore
    block_params = BlockParams.from_init_params(
        depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs
    )
    return RegNet(
        block_params=block_params, norm_layer=partial(nn.BatchNorm, eps=1e-05, momentum=0.1), num_classes=num_classes
    )