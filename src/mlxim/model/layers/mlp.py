import mlx.nn as nn
from typing import List, Optional, Callable
import mlx.core as mx


class MLP(nn.Module):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``nn.ReLU``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            self.layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                self.layers.append(norm_layer(hidden_dim))
            self.layers.append(activation_layer())
            self.layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        self.layers.append(nn.Dropout(dropout))

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x
