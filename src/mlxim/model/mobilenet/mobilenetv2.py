"""MobileNetV2 implementation for MLX.

Based on the paper: MobileNetV2: Inverted Residuals and Linear Bottlenecks (https://arxiv.org/abs/1801.04381)

Original implementation: torchvision (https://github.com/pytorch/vision)
"""
import mlx.core as mx
import mlx.nn as nn
from typing import Callable, Optional

from ..layers import Conv2dNormActivation
from ..layers.utils import _make_divisible

# Necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1,  norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._in_cn = stride > 1

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self, num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[list[list[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multipler - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for MobileNet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # Building the first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels=3, out_channels=input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]

        # Building the inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )

        # Make it nn.Sequential
        self.features = nn.Sequential(*features)

        # Building the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.he_uniform(m.weight)
            elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
                nn.init.constant(1, m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.he_uniform(m.weight)

        self.num_classes = num_classes

    def get_features(self, x: mx.array) -> mx.array:
        """Extract only features.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output features
        """
        x = self.features(x)
        # Global average pooling
        x = x.mean(axis=(1, 2))
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """MobileNetV2 call.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output array
        """
        x = self.get_features(x)
        x = self.classifier(x)
        return x