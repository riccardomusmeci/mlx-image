import warnings
from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ._ops import stochastic_depth

from .pool import AdaptiveAvgPool2d


class ConvNormActivation(nn.Module):
    """Configurable block used for Convolution-Normalization-Activation blocks.

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): size of the convolving kernel. Defaults to 3.
        stride (int, optional): stride of the convolution. Defaults to 1.
        padding (int, tuple or str, optional): padding added to all four sides of the input. Defaults to None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): number of blocked connections from input channels to output channels. Defaults to 1.
        norm_layer (Callable[..., nn.Module], optional): norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Defaults to ``nn.BatchNorm``
        activation_layer (Callable[..., nn.Module], optional): activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Defaults to ``nn.ReLU``
        dilation (int): spacing between kernel elements. Defaults to 1.
        inplace (bool): parameter for the activation layer, which can optionally do the operation in-place. Defaults to ``True``
        bias (bool, optional): whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
        conv_layer (Callable[..., nn.Module]): convolution layer. Defaults to ``nn.Conv2d``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = None,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        self.layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                # groups=groups, # TODO: remove since mlx.nn does not have this parameter
                bias=bias,
            )
        ]

        if norm_layer is not None:
            self.layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            # TODO: removed inplace
            # params = {} if inplace is None else {"inplace": inplace}
            params = {}  # type: ignore
            self.layers.append(activation_layer(**params))
        self.out_channels = out_channels

    def __call__(self, x: mx.array) -> mx.array:
        for l in self.layers:
            x = l(x)
        return x


class Conv2dNormActivation(ConvNormActivation):
    """Configurable block used for Convolution-Normalization-Activation blocks.

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): size of the convolving kernel. Defaults to 3.
        stride (int, optional): stride of the convolution. Defaults to 1.
        padding (int, tuple or str, optional): padding added to all four sides of the input. Defaults to None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): number of blocked connections from input channels to output channels. Defaults to 1.
        norm_layer (Callable[..., nn.Module], optional): norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Defaults to ``nn.BatchNorm``
        activation_layer (Callable[..., nn.Module], optional): activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Defaults to ``nn.ReLU``
        dilation (int): spacing between kernel elements. Defaults to 1.
        inplace (bool): parameter for the activation layer, which can optionally do the operation in-place. Defaults to ``True``
        bias (bool, optional): whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = None,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            nn.Conv2d,
        )


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., nn.Module], optional): ``delta`` activation. Default: ``nn.ReLU``
        scale_activation (Callable[..., nn.Module]): ``sigma`` activation. Default: ``nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: mx.array) -> mx.array:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def __call__(self, input: mx.array) -> mx.array:
        scale = self._scale(input)
        return scale * input


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()

    def __call__(self, input: mx.array) -> mx.array:
        return mx.clip(input / 6 + 0.5, 0, 1)


class StochasticDepth(nn.Module):
    """Stochastic Depth from `"Deep Networks with Stochastic Depth"

    Args:
        p: probability of the x to be zeroed.
        mode: ``"batch"`` or ``"row"``.
              ``"batch"`` randomly zeroes the entire x, ``"row"`` zeroes
              randomly selected rows from the batch.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def __call__(self, input: mx.array) -> mx.array:
        return stochastic_depth(input, self.p, self.mode, self.training)
