from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
from dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    """Mixin for forward pass on list of arrays."""
    def forward(self, x: mx.array):
        """Forward pass on array."""
        raise NotImplementedError

    def forward_list(self, x_list: List[mx.array]) -> List[mx.array]:
        """Forward pass on list of arrays."""
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    """MLP block.
    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features.
        out_features: Number of output features.
        act_layer: Activation layer.
        drop: Dropout rate.
        bias: Whether to use bias.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input array.
        Returns:
            Output array.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    """SwiGLU FFN block.
    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features.
        out_features: Number of output features.
        act_layer: Activation layer.
        drop: Dropout rate.
        bias: Whether to use bias.
        align_to: Alignment to use for hidden features.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Input array.
        Returns:
            Output array.
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = mx.multiply(nn.SiLU()(x1), x2)
        return self.w3(hidden)
