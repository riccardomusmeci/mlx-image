import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..layers.utils import _make_divisible
from ..layers.misc import Conv2dNormActivation, SqueezeExcitation
from ..layers.pool import AdaptiveAvgPool2d


class SimpleStemIN(Conv2dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1.

    Args:
        width_in (int): input width
        width_out (int): output width
        stride (int): stride
        norm_layer (Callable[..., nn.Module]): normalization layer
        activation_layer (Callable[..., nn.Module]): activation layer
        group_width (int): group width
        bottleneck_multiplier (float): bottleneck multiplier
        se_ratio (Optional[float]): squeeze-and-excitation reduction ratio
    """

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        super().__init__()

        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        self.a = Conv2dNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        self.b = Conv2dNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )
        self.se = None
        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            self.se = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        self.c = Conv2dNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.a(x)

        x = self.b(x)
        if self.se:
            x = self.se(x)
        x = self.c(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform.

    Args:
        width_in (int): input width
        width_out (int): output width
        stride (int): stride
        norm_layer (Callable[..., nn.Module]): normalization layer
        activation_layer (Callable[..., nn.Module]): activation layer
        group_width (int): group width
        bottleneck_multiplier (float): bottleneck multiplier
        se_ratio (Optional[float]): squeeze-and-excitation reduction ratio
    """

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        # self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        else:
            self.proj = None

        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )

        # TODO: removed inplace=True
        self.activation = activation_layer()
        # self.activation = activation_layer(inplace=True)

    def __call__(self, x: mx.array) -> mx.array:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape).

    Args:
        width_in (int): input width
        width_out (int): output width
        stride (int): stride
        depth (int): number of blocks
        block_constructor (Callable[..., nn.Module]): block constructor
        norm_layer (Callable[..., nn.Module]): normalization layer
        activation_layer (Callable[..., nn.Module]): activation layer
        group_width (int): group width
        bottleneck_multiplier (float): bottleneck multiplier
        se_ratio (Optional[float]): squeeze-and-excitation reduction ratio. Defaults to None.
        stage_index (int): stage index. Defaults to 0.
    """

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                block_constructor(
                    width_in if i == 0 else width_out,
                    width_out,
                    stride if i == 0 else 1,
                    norm_layer,
                    activation_layer,
                    group_width,
                    bottleneck_multiplier,
                    se_ratio,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        for m in self.blocks:
            x = m(x)
        return x


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = np.arange(depth) * w_a + w_0
        block_capacity = np.round(np.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (np.round(np.divide(w_0 * np.power(w_m, block_capacity), QUANT)) * QUANT).astype(int).tolist()

        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        # stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        stage_depths = np.diff([d for d, t in enumerate(splits) if t]).tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                AnyStage(
                    width_in=current_width,
                    width_out=width_out,
                    stride=stride,
                    depth=depth,
                    block_constructor=block_type,
                    norm_layer=norm_layer,
                    activation_layer=activation,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                    se_ratio=block_params.se_ratio,
                    stage_index=i + 1,
                ),
            )

            current_width = width_out

        self.trunk_output = blocks

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        if num_classes > 0:
            self.fc = nn.Linear(current_width, num_classes)
        else:
            self.fc = nn.Identity()

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.he_uniform(m.weight)  # type: ignore
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm):
                nn.init.constant(1, m.weight)
                nn.init.constant(0, m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.he_uniform(m.weight)  # type: ignore
                nn.init.constant(0, m.bias)

    def features(self, x: mx.array) -> mx.array:
        x = self.stem(x)
        for m in self.trunk_output:
            x = m(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))  # flatten operation
        return x

    def __call__(self, x: mx.array) -> mx.array:
        x = self.features(x)
        x = self.fc(x)
        return x
