from pathlib import Path
from typing import Any, Callable, List, Optional, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ..layers import AdaptiveAvgPool2d


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding.

    Args:
        in_planes (int): convolution in_channels
        out_planes (int): convolution out_channels
        stride (int, optional): convolution stridel. Defaults to 1.
        dilation (int, optional): convolution padding. Defaults to 1.

    Returns:
        nn.Conv2d: 3x3 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution.

    Args:
        in_planes (int): convolution in_channels
        out_planes (int): convolution out_channels
        stride (int, optional): convolution stride. Defaults to 1.

    Returns:
        nn.Conv2d: 1x1 convolution
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """BasicBlock constructor.

    Args:
        inplanes (int): in features of the block
        planes (int): out features of the block
        stride (int, optional): convolution stride. Defaults to 1.
        downsample (nn.Module, optional): downsample layer. Defaults to None.
        groups (int, optional): convolution groups. Defaults to 1.
        base_width (int, optional): convolution base width. Defaults to 64.
        dilation (int, optional): convolution dilation. Defaults to 1.
        norm_layer (nn.Module, optional): norm layer. If None, defaults to nn.BatchNorm. Defaults to None.

    Raises:
        ValueError: BasicBlock only supports groups=1 and base_width=64
        NotImplementedError: Dilation > 1 not supported in BasicBlock
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm  # type: ignore
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)  # type: ignore
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)  # type: ignore
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        """BasicBlock call.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output array
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck constructor.

    Args:
        inplanes (int): in features
        planes (int): out features
        stride (int, optional): convolution stride. Defaults to 1.
        downsample (Optional[nn.Module], optional): downsample layer. Defaults to None.
        groups (int, optional): convolution groups. Defaults to 1.
        base_width (int, optional): convolution base width. Defaults to 64.
        dilation (int, optional): convolution dilation. Defaults to 1.
        norm_layer (Optional[nn.Module], optional): norm layer. If None, defaults to nn.BatchNorm. Defaults to None.
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm  # type: ignore
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)  # type: ignore
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)  # type: ignore
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)  # type: ignore
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        """Bottleneck call.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output array
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model.

    Args:
        block (Union[BasicBlock, Bottleneck]): resnet base block type
        layers (List[int]): layers for each block
        num_classes (int, optional): number of classes. Defaults to 1000.
        zero_init_residual (bool, optional): zero init residual. Defaults to False.
        groups (int, optional): groups. Defaults to 1.
        width_per_group (int, optional): width per group. Defaults to 64.
        replace_stride_with_dilation (Optional[List[bool]], optional): replace stride with dilation. Defaults to None.
        norm_layer (Optional[nn.Module], optional): norm layer. If None, defaults to nn.BatchNorm. Defaults to None.
    """

    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm  # type: ignore
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)  # type: ignore
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = AdaptiveAvgPool2d(1)
        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Identity()  # type: ignore

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.he_uniform(m.weight)  # type: ignore
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
                nn.init.constant(1, m.weight)

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Make ResNet layer.

        Args:
            block (Union[BasicBlock, Bottleneck]): type of block
            planes (int): output features
            blocks (int): number of blocks
            stride (int, optional): stride. Defaults to 1.
            dilate (bool, optional): dilate. Defaults to False.

        Returns:
            nn.Sequential: ResNet layer
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(in_planes=self.inplanes, out_planes=planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),  # type: ignore
            )
        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def features(self, x: mx.array) -> mx.array:
        """Extact only features.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output features
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))  # flatten operation

        return x

    def __call__(self, x: mx.array) -> mx.array:
        """ResNet call.

        Args:
            x (mx.array): input array

        Returns:
            mx.array: output array
        """
        x = self.features(x)
        x = self.fc(x)
        return x
