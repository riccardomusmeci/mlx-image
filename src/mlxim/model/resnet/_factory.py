from .._config import HFWeights, Metrics, ModelConfig, Transform
from .resnet import BasicBlock, Bottleneck, ResNet

resnet_configs = {
    "resnet18": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.64353, accuracy_at_5=0.85725),
        transform=Transform(img_size=224),
        weights=HFWeights(repo_id="mlx-vision/resnet18-mlxim", filename="model.safetensors"),
    ),
    "resnet34": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.69064, accuracy_at_5=0.88848),
        transform=Transform(img_size=224),
        weights=HFWeights(repo_id="mlx-vision/resnet34-mlxim", filename="model.safetensors"),
    ),
    "resnet50": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.78143, accuracy_at_5=0.9407),
        transform=Transform(img_size=224),
        weights=HFWeights(repo_id="mlx-vision/resnet50-mlxim", filename="model.safetensors"),
    ),
    "resnet101": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.79916, accuracy_at_5=0.9486),
        transform=Transform(img_size=224),
        weights=HFWeights(repo_id="mlx-vision/resnet101-mlxim", filename="model.safetensors"),
    ),
    "resnet152": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80624, accuracy_at_5=0.950482),
        transform=Transform(img_size=224),
        weights=HFWeights(repo_id="mlx-vision/resnet152-mlxim", filename="model.safetensors"),
    ),
    "wide_resnet50_2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.79635, accuracy_at_5=0.94678),
        transform=Transform(img_size=224),
        weights=HFWeights(
            repo_id="mlx-vision/wide_resnet50_2-mlxim", filename="model.safetensors"
        ),
    ),
    "wide_resnet101_2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80948, accuracy_at_5=0.95222),
        transform=Transform(img_size=224),
        weights=HFWeights(
            repo_id="mlx-vision/wide_resnet101_2-mlxim", filename="model.safetensors"
        ),
    ),
    # TODO: waiting for groups and dilation support
    # "resnext50_32x4d": ModelConfig(
    #     metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81198, accuracy_at_5=0.95340),
    #     transform=Transform(crop=224, resize=232),
    #     weights=HFWeights(repo_id="mlx-vision/resnext50_32x4d-mlxim", filename="model.safetensors")
    # ),
    # "resnext101_32x8d": ModelConfig(
    #     metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.82834, accuracy_at_5=0.96228),
    #     transform=Transform(crop=224, resize=232),
    #     weights=HFWeights(
    #         repo_id="mlx-vision/resnext101_32x8d-mlxim", filename="model.safetensors"
    #     ),
    # ),
    # "resnext101_64x4d": ModelConfig(
    #     metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.83246, accuracy_at_5=0.96454),
    #     transform=Transform(crop=224, resize=232),
    #     weights=HFWeights(
    #         repo_id="mlx-vision/resnext101_64x4d-mlxim", filename="model.safetensors"
    #     ),
    # )
}


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes: int = 1000) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes: int = 1000) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes)


def wide_resnet50_2(num_classes: int = 1000) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, width_per_group=64 * 2)


def wide_resnet101_2(num_classes: int = 1000) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, width_per_group=64 * 2)


# TODO: waiting for groups and dilation support
# def resnext50_32x4d(num_classes: int = 1000) -> ResNet:
#     return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)

# def resnext101_32x8d(num_classes: int = 1000) -> ResNet:
#     return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, groups=32, width_per_group=8)

# def resnext101_64x4d(num_classes: int = 1000) -> ResNet:
#     return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, groups=64, width_per_group=4)
