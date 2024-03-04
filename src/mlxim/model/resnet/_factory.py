from .._config import HFWeights, Metrics, ModelConfig, Transform
from .resnet import BasicBlock, Bottleneck, ResNet

resnet_configs = {
    "resnet18": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.69758, accuracy_at_5=0.89078),
        transform=Transform(crop=224),
        weights=HFWeights(repo_id="mlx-vision/resnet18-mlxim", filename="resnet18-IMAGENET1K_V1-mlx.npz"),
    ),
    "resnet34": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.73314, accuracy_at_5=0.91420),
        transform=Transform(crop=224),
        weights=HFWeights(repo_id="mlx-vision/resnet34-mlxim", filename="resnet34-IMAGENET1K-mlx.npz"),
    ),
    "resnet50": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.80858, accuracy_at_5=0.94258),
        transform=Transform(crop=224, resize=232),
        weights=HFWeights(repo_id="mlx-vision/resnet50-mlxim", filename="resnet50-IMAGENET1K_V2-mlx.npz"),
    ),
    "resnet101": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81886, accuracy_at_5=0.95780),
        transform=Transform(crop=224, resize=232),
        weights=HFWeights(repo_id="mlx-vision/resnet101-mlxim", filename="resnet101-IMAGENET1K_V2-mlx.npz"),
    ),
    "resnet152": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.82284, accuracy_at_5=0.96002),
        transform=Transform(crop=224, resize=232),
        weights=HFWeights(repo_id="mlx-vision/resnet152-mlxim", filename="resnet152-IMAGENET1K_V2-mlx.npz"),
    ),
    "wide_resnet50_2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81602, accuracy_at_5=0.95758),
        transform=Transform(crop=224, resize=232),
        weights=HFWeights(repo_id="mlx-vision/wide_resnet50_2-mlxim", filename="wide_resnet50_2-IMAGENET1K_V2-mlx.npz"),
    ),
    "wide_resnet101_2": ModelConfig(
        metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.82510, accuracy_at_5=0.96020),
        transform=Transform(crop=224, resize=232),
        weights=HFWeights(
            repo_id="mlx-vision/wide_resnet101_2-mlxim", filename="wide_resnet101_2-IMAGENET1K_V2-mlx.npz"
        ),
    ),
    # TODO: waiting for groups and dilation support
    # "resnext50_32x4d": ModelConfig(
    #     metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.81198, accuracy_at_5=0.95340),
    #     transform=Transform(crop=224, resize=232),
    #     weights=HFWeights(repo_id="mlx-vision/resnext50_32x4d-mlxim", filename="resnext50_32x4d-IMAGENET1K_V2-mlx.npz")
    # ),
    # "resnext101_32x8d": ModelConfig(
    #     metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.82834, accuracy_at_5=0.96228),
    #     transform=Transform(crop=224, resize=232),
    #     weights=HFWeights(
    #         repo_id="mlx-vision/resnext101_32x8d-mlxim", filename="resnext101_32x8d-IMAGENET1K_V2-mlx.npz"
    #     ),
    # ),
    # "resnext101_64x4d": ModelConfig(
    #     metrics=Metrics(dataset="ImageNet-1K", accuracy_at_1=0.83246, accuracy_at_5=0.96454),
    #     transform=Transform(crop=224, resize=232),
    #     weights=HFWeights(
    #         repo_id="mlx-vision/resnext101_64x4d-mlxim", filename="resnext101_64x4d-IMAGENET1K_V1-mlx.npz"
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
