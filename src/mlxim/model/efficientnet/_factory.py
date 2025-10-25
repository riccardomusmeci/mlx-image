"""Factory functions for EfficientNet models."""
from typing import List, Optional

from .._config import HFWeights, Metrics, ModelConfig, Transform
from ._blocks import MBConvConfig
from .efficientnet import EfficientNet


def _efficientnet_conf(
    arch: str,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
) -> tuple[List[MBConvConfig], Optional[int]]:
    """Get configuration for EfficientNet variants.

    Args:
        arch (str): Architecture name (e.g., 'efficientnet_b0')
        width_mult (float): Width multiplier for channels
        depth_mult (float): Depth multiplier for number of layers

    Returns:
        Tuple of (inverted_residual_setting, last_channel)
    """
    if arch.startswith("efficientnet_b"):
        # EfficientNet-B0 through B7 configuration
        # Format: (expand_ratio, kernel, stride, input_channels, out_channels, num_layers)
        inverted_residual_setting = [
            MBConvConfig(1, 3, 1, 32, 16, 1, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 16, 24, 2, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 24, 40, 2, width_mult, depth_mult),
            MBConvConfig(6, 3, 2, 40, 80, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 1, 80, 112, 3, width_mult, depth_mult),
            MBConvConfig(6, 5, 2, 112, 192, 4, width_mult, depth_mult),
            MBConvConfig(6, 3, 1, 192, 320, 1, width_mult, depth_mult),
        ]
        last_channel = None
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def efficientnet_b0(num_classes: int = 1000, dropout: float = 0.2) -> EfficientNet:
    """EfficientNet-B0 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B0 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b1(num_classes: int = 1000, dropout: float = 0.2) -> EfficientNet:
    """EfficientNet-B1 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B1 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b1", width_mult=1.0, depth_mult=1.1
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b2(num_classes: int = 1000, dropout: float = 0.3) -> EfficientNet:
    """EfficientNet-B2 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B2 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b2", width_mult=1.1, depth_mult=1.2
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b3(num_classes: int = 1000, dropout: float = 0.3) -> EfficientNet:
    """EfficientNet-B3 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B3 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b4(num_classes: int = 1000, dropout: float = 0.4) -> EfficientNet:
    """EfficientNet-B4 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B4 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b4", width_mult=1.4, depth_mult=1.8
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b5(num_classes: int = 1000, dropout: float = 0.4) -> EfficientNet:
    """EfficientNet-B5 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B5 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b5", width_mult=1.6, depth_mult=2.2
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b6(num_classes: int = 1000, dropout: float = 0.5) -> EfficientNet:
    """EfficientNet-B6 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B6 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b6", width_mult=1.8, depth_mult=2.6
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


def efficientnet_b7(num_classes: int = 1000, dropout: float = 0.5) -> EfficientNet:
    """EfficientNet-B7 model.

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate

    Returns:
        EfficientNet: EfficientNet-B7 model
    """
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b7", width_mult=2.0, depth_mult=3.1
    )

    return EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        num_classes=num_classes,
        last_channel=last_channel,
    )


# Model configuration registry
efficientnet_configs = {
    "efficientnet_b0": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.77692, accuracy_at_5=0.93532),
        transform=Transform(img_size=224, crop_pct=224/256, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b0-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b1": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.7840, accuracy_at_5=0.9423),
        transform=Transform(img_size=240, crop_pct=240/255, interpolation="bilinear"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b1-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80762, accuracy_at_5=0.9525),
        transform=Transform(img_size=288, crop_pct=288/288, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b2-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b3": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.82368, accuracy_at_5=0.96157),
        transform=Transform(img_size=300, crop_pct=300/320, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b3-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b4": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.83749, accuracy_at_5=0.96613),
        transform=Transform(img_size=380, crop_pct=380/384, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b4-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b5": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.8386, accuracy_at_5=0.9678),
        transform=Transform(img_size=456, crop_pct=456/456, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b5-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b6": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.8458, accuracy_at_5=0.9710),
        transform=Transform(img_size=528, crop_pct=528/528, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b6-mlxim", filename="model.safetensors"),
    ),
    "efficientnet_b7": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.84887, accuracy_at_5=0.97247),
        transform=Transform(img_size=600, crop_pct=600/600, interpolation="bicubic"),
        weights=HFWeights(repo_id="mlx-vision/efficientnet_b7-mlxim", filename="model.safetensors"),
    ),
}
