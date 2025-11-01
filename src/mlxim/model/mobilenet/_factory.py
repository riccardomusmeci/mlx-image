from .._config import HFWeights, Metrics, ModelConfig, Transform
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3, _mobilenet_v3_conf

mobilenet_configs = {
    "mobilenet_v2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.69846, accuracy_at_5=0.89653),
        transform=Transform(img_size=224, crop_pct=224/232, interpolation="bilinear"),
        weights=HFWeights(repo_id="mlx-vision/mobilenet_v2-mlxim", filename="model.safetensors"),
    ),
    "mobilenet_v3_large": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.69888, accuracy_at_5=0.89071),
        transform=Transform(img_size=224, crop_pct=224/232, interpolation="bilinear"),
        weights=HFWeights(repo_id="mlx-vision/mobilenet_v3_large-mlxim", filename="model.safetensors"),
    ),
    "mobilenet_v3_small": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.63176, accuracy_at_5=0.84807),
        transform=Transform(img_size=224, crop_pct=224/232, interpolation="bilinear"),
        weights=HFWeights(repo_id="mlx-vision/mobilenet_v3_small-mlxim", filename="model.safetensors"),
    ),
}


def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNetV2:
    """MobileNetV2 model.

    Args:
        num_classes (int): Number of output classes

    Returns:
        MobileNetV2: MobileNetV2 model
    """
    return MobileNetV2(num_classes=num_classes, **kwargs)


def mobilenet_v3_large(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs,
) -> MobileNetV3:
    """Constructs a large MobileNetV3 architecture.

    Args:
        num_classes: Number of classes
        width_mult: Width multiplier
        reduced_tail: Whether to reduce the tail
        dilated: Whether to use dilated convolutions

    Returns:
        MobileNetV3 model
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large",
        width_mult=width_mult,
        reduced_tail=reduced_tail,
        dilated=dilated,
    )
    return MobileNetV3(inverted_residual_setting, last_channel, num_classes=num_classes, **kwargs)


def mobilenet_v3_small(
    num_classes: int = 1000,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs,
) -> MobileNetV3:
    """Constructs a small MobileNetV3 architecture.

    Args:
        num_classes: Number of classes
        width_mult: Width multiplier
        reduced_tail: Whether to reduce the tail
        dilated: Whether to use dilated convolutions

    Returns:
        MobileNetV3 model
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_small",
        width_mult=width_mult,
        reduced_tail=reduced_tail,
        dilated=dilated,
    )
    return MobileNetV3(inverted_residual_setting, last_channel, num_classes=num_classes, **kwargs)
