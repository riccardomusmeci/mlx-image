# from ..layers.patch_embed import PatchEmbed
# from ..layers.mlp import Mlp
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn


class MLPBlock(nn.Module):
    """Transformer MLP block.

    Args:
        in_dim (int): input dimension
        mlp_dim (int): mlp dimension
        dropout (float): dropout
    """

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        self.linear_1 = nn.Linear(in_dim, mlp_dim, bias=True)
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim, bias=True)
        self.dropout_2 = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array of shape (batch_size, seq_length, hidden_dim)

        Returns:
            mx.array: output mx.array of shape (batch_size, seq_length, hidden_dim)
        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block.

    Args:
        num_heads (int): number of heads
        hidden_dim (int): hidden dimension
        mlp_dim (int): mlp dimension
        dropout (float): dropout
        norm_layer (Callable[..., nn.Module], optional): normalization layer. Defaults to nn.LayerNorm.
        bias (bool, optional): attention bias. Defaults to True.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiHeadAttention(hidden_dim, num_heads, bias=bias)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array of shape (batch_size, seq_length, hidden_dim)

        Returns:
            mx.array: Output mx.array of shape (batch_size, seq_length, hidden_dim)
        """
        _x = x
        assert _x.ndim == 3, f"Expected (batch_size, seq_length, hidden_dim) got {_x.shape}"
        x = self.ln_1(_x)
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + _x

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Args:
        seq_length (int): sequence length
        num_layers (int): number of layers
        num_heads (int): number of heads
        hidden_dim (int): hidden dimension
        mlp_dim (int): mlp dimension
        dropout (float): dropout
        norm_layer (Callable[..., nn.Module], optional): normalization layer. Defaults to nn.LayerNorm.
        eps (float, optional): epsilon for normalization layer. Defaults to 1e-6.
    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        eps: float = 1e-6,
        attn_bias: bool = True,
    ):
        super().__init__()
        self.pos_embedding = mx.zeros((1, seq_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        layers = []
        for _i in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    bias=attn_bias,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.ln = norm_layer(hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array of shape (batch_size, seq_length, hidden_dim)

        Returns:
            mx.array: output mx.array of shape (batch_size, seq_length, hidden_dim)
        """
        assert x.ndim == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}"
        x = x + self.pos_embedding
        x_ = self.layers(self.dropout(x))
        return self.ln(x_)


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_bias: bool = True,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = mx.zeros((1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            norm_layer=norm_layer,
            attn_bias=attn_bias,
        )
        self.seq_length = seq_length

        heads_layers = []
        if representation_size is None:
            heads_layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            heads_layers.append(nn.Linear(hidden_dim, representation_size))
            heads_layers.append(nn.Tanh())  # type: ignore
            heads_layers.append(nn.Linear(representation_size, num_classes))
        if self.num_classes > 0:
            self.heads = nn.Sequential(*heads_layers)
        else:
            self.heads = nn.Identity()  # type: ignore

    def _process_input(self, x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        p = self.patch_size
        assert h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
        assert w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!"
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(n, n_h * n_w, self.hidden_dim)
        return x

    def features(self, x: mx.array) -> mx.array:
        """Forward pass for feature

        Args:
            x (mx.array): input mx.array of shape (batch_size, image_size, image_size, 3)

        Returns:
            mx.array: output mx.array of shape (batch_size, hidden_dim)
        """
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = mx.repeat(self.class_token, n, 0)
        x = mx.concatenate([batch_class_token, x], axis=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x (mx.array): input mx.array of shape (batch_size, image_size, image_size, 3)

        Returns:
            mx.array: output mx.array of shape (batch_size, num_classes)
        """
        # Reshape and permute the input tensor
        x = self.features(x)
        x = self.heads(x)

        return x
