"""MobileNetV3 implementation for MLX.

Based on the paper: Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)

Original implementation: torchvision (https://github.com/pytorch/vision)
"""
from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..layers import Conv2dNormActivation, SqueezeExcitation
from ..layers.misc import HardSigmoid
from ..layers.utils import _make_divisible


class InvertedResidualConfig:
    """Stores information for building an InvertedResidual block.

    This configuration matches the structure in Tables 1 and 2 of the MobileNetV3 paper.
    """

    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"  # True = Hardswish, False = ReLU
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """Inverted Residual block for MobileNetV3.

    Implemented as described in section 5 of the MobileNetV3 paper.

    Args:
        cnf: Configuration for this block
        norm_layer: Normalization layer
        se_layer: Squeeze-Excitation layer
    """

    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        se_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        if norm_layer is None:
            norm_layer = nn.BatchNorm

        if se_layer is None:
            se_layer = SqueezeExcitation

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # Expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # Squeeze-Excitation
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels, scale_activation=HardSigmoid))

        # Project (no activation)
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def __call__(self, x: mx.array) -> mx.array:
        result = self.block(x)
        if self.use_res_connect:
            result = result + x
        return result


class MobileNetV3(nn.Module):
    """MobileNetV3 main class.

    Args:
        inverted_residual_setting: Network structure
        last_channel: The number of channels on the penultimate layer
        num_classes: Number of classes
        block: Module specifying inverted residual building block for mobilenet
        norm_layer: Module specifying the normalization layer to use
        dropout: The dropout probability
    """

    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not all(isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            # MobileNetV3 uses different BatchNorm settings than MobileNetV2
            norm_layer = nn.BatchNorm

        layers: List[nn.Module] = []

        # Building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # Building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # Building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.he_uniform(m.weight)
            elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
                nn.init.constant(1, m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.he_uniform(m.weight)

        self.num_classes = num_classes
        self.last_channel = last_channel

    def get_features(self, x: mx.array) -> mx.array:
        """Extract features before the classifier.

        Args:
            x: Input array

        Returns:
            Output features
        """
        x = self.features(x)
        # Global average pooling
        x = x.mean(axis=(1, 2))
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input array

        Returns:
            Output logits
        """
        x = self.get_features(x)
        x = self.classifier(x)
        return x


def _mobilenet_v3_conf(
    arch: str,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
):
    """Configuration for MobileNetV3 variants.

    Args:
        arch: Architecture name ('mobilenet_v3_large' or 'mobilenet_v3_small')
        width_mult: Width multiplier
        reduced_tail: Whether to reduce the tail (for faster inference)
        dilated: Whether to use dilated convolutions

    Returns:
        Tuple of (inverted_residual_setting, last_channel)
    """
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    def bneck_conf(
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
    ):
        return InvertedResidualConfig(
            input_channels, kernel, expanded_channels, out_channels,
            use_se, activation, stride, dilation, width_mult
        )

    def adjust_channels(channels: int):
        return InvertedResidualConfig.adjust_channels(channels, width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel
