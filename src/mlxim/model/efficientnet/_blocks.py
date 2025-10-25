"""MBConv blocks for EfficientNet."""
import math
from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from ..layers import Conv2dNormActivation, SqueezeExcitation, StochasticDepth
from ..layers.utils import _make_divisible


@dataclass
class MBConvConfig:
    """Configuration for MBConv block.

    Args:
        expand_ratio (float): Expansion ratio for the hidden dimension
        kernel (int): Kernel size for depthwise convolution
        stride (int): Stride for depthwise convolution
        input_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_layers (int): Number of layers in this stage
        width_mult (float): Width multiplier for channels
        depth_mult (float): Depth multiplier for number of layers
    """
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    width_mult: float = 1.0
    depth_mult: float = 1.0

    def __post_init__(self):
        """Adjust channels and depth based on multipliers."""
        self.input_channels = self.adjust_channels(self.input_channels, self.width_mult)
        self.out_channels = self.adjust_channels(self.out_channels, self.width_mult)
        self.num_layers = self.adjust_depth(self.num_layers, self.depth_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        """Adjust number of channels based on width multiplier."""
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float) -> int:
        """Adjust depth (number of layers) based on depth multiplier."""
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    This block is the core building block of EfficientNet. It consists of:
    1. Expansion: 1x1 conv to expand channels (if expand_ratio != 1)
    2. Depthwise: Depthwise conv with kernel_size and stride
    3. Squeeze-and-Excitation: Channel attention
    4. Projection: 1x1 conv to project back to output channels
    5. Skip connection: Add input to output (if stride == 1 and input == output channels)

    Args:
        cnf (MBConvConfig): Configuration for this block
        stochastic_depth_prob (float): Probability for stochastic depth
        norm_layer (Callable): Normalization layer (default: nn.BatchNorm)
        se_layer (Callable): Squeeze-and-Excitation layer (default: SqueezeExcitation)
    """

    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("Illegal stride value")

        if norm_layer is None:
            norm_layer = nn.BatchNorm

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = nn.SiLU

        # Expansion phase (1x1 pointwise conv)
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise convolution phase
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,  # Depthwise
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # Squeeze and Excitation phase
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=nn.SiLU,
                scale_activation=nn.Sigmoid,
            )
        )

        # Projection phase (1x1 pointwise conv, no activation)
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,  # No activation
            )
        )

        self.block = layers
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the MBConv block.

        Args:
            x (mx.array): Input tensor

        Returns:
            mx.array: Output tensor
        """
        result = x
        for layer in self.block:
            result = layer(result)

        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = result + x

        return result
