import math
from functools import partial
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    """Multi-head attention module.

    Args:
        dims (int): input dimensions
        num_heads (int): number of heads
        query_input_dims (Optional[int], optional): query input dimensions. Defaults to None.
        key_input_dims (Optional[int], optional): key input dimensions. Defaults to None.
        value_input_dims (Optional[int], optional): value input dimensions. Defaults to None.
        value_dims (Optional[int], optional): value dimensions. Defaults to None.
        value_output_dims (Optional[int], optional): value output dimensions. Defaults to None.
        bias (bool, optional): attention bias. Defaults to False.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        self.query_proj = nn.Linear(query_input_dims, dims, bias=bias)
        self.key_proj = nn.Linear(key_input_dims, dims, bias=bias)
        self.value_proj = nn.Linear(value_input_dims, value_dims, bias=bias)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=bias)

    def __call__(
        self, queries: mx.array, keys: mx.array, values: mx.array, mask: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass

        Args:
            queries (mx.array): attention queries
            keys (mx.array): attention keys
            values (mx.array): attention values
            mask (Optional[mx.array], optional): attention mask. Defaults to None.

        Returns:
            Tuple[mx.array, mx.array]: attention output and attention mask
        """
        # compute queries, keys and values
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape

        # reshape queries, keys and values
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 3, 1)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        # define scale
        scale = math.sqrt(1 / queries.shape[-1])
        attn = (queries * scale) @ keys
        if mask is not None:
            attn = attn + mask.astype(attn.dtype)
        attn = mx.softmax(attn, axis=-1)
        values_hat = (attn @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), attn


class MLPBlock(nn.Module):
    """Transformer MLP block.

    Args:
        in_dim (int): input dimension
        mlp_dim (int): mlp dimension
        dropout (float): dropout
    """

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()

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
        init_values (Optional[float]): initial values for layer scale. Defaults to None.
        norm_layer (Callable[..., nn.Module], optional): normalization layer. Defaults to nn.LayerNorm.
        bias (bool, optional): attention bias. Defaults to True.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        init_values: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = Attention(dims=hidden_dim, num_heads=num_heads, bias=bias)
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
        x, attn_mask = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + _x

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y, attn_mask


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Args:
        seq_length (int): sequence length
        num_layers (int): number of layers
        num_heads (int): number of heads
        hidden_dim (int): hidden dimension
        mlp_dim (int): mlp dimension
        dropout (float): dropout
        init_values (Optional[float]): initial values for layer scale. Defaults to None.
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
        init_values: Optional[float] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        eps: float = 1e-6,
        attn_bias: bool = True,
    ):
        super().__init__()
        self.pos_embedding = mx.zeros((1, seq_length, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.layers = []
        for _i in range(num_layers):
            self.layers.append(
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    init_values=init_values,
                    norm_layer=norm_layer,
                    bias=attn_bias,
                )
            )
        self.ln = norm_layer(hidden_dim)

    def get_intermediate_layers(self, x: mx.array, blocks_to_take: Iterable) -> Tuple[List[mx.array], List[mx.array]]:
        """Get intermediate layers outputs

        Args:
            x (mx.array): input mx.array of shape (batch_size, seq_length, hidden_dim)
            blocks_to_take (Iterable): blocks to take

        Returns:
            Tuple[List[mx.array], List[mx.array]]: list of intermediate layers outputs and attention matrices
        """
        assert x.ndim == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}"
        x = x + self.pos_embedding

        output, attn_mat = [], []
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            if i in blocks_to_take:
                output.append(x)
                attn_mat.append(attn)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output, attn_mat

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array of shape (batch_size, seq_length, hidden_dim)

        Returns:
            mx.array: output mx.array of shape (batch_size, seq_length, hidden_dim)
        """
        assert x.ndim == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}"
        x = x + self.pos_embedding

        attn_masks = []
        for layer in self.layers:
            x, attn_mask = layer(x)
            attn_masks.append(attn_mask)
        return self.ln(x), attn_masks


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: Optional[int] = None,  # usually hidden_dimx4
        dropout: float = 0.0,
        num_classes: int = 1000,
        init_values: Optional[float] = None,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_bias: bool = True,
        head_type: Literal["linear", "dino"] = "linear",
        n_last_blocks: int = 4,  # only for DINO
        avgpool: bool = False,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else 4 * hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.head_type = head_type
        self.n_last_blocks = n_last_blocks
        self.avgpool = avgpool

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
            init_values=init_values,
            norm_layer=norm_layer,
            attn_bias=attn_bias,
        )
        self.seq_length = seq_length

        heads_layers = []
        if self.head_type == "dino":
            heads_layers.append(
                DINOHead(in_dim=hidden_dim * (self.n_last_blocks + int(self.avgpool)), out_dim=num_classes)
            )
        elif self.head_type == "linear":
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

    def _prepare_tokens(self, x: mx.array) -> mx.array:
        """Prepare tokens

        Args:
            x (mx.array): input mx.array of shape (batch_size, seq_length, hidden_dim)

        Returns:
            mx.array: output mx.array of shape (batch_size, seq_length, hidden_dim)
        """
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = mx.repeat(self.class_token, n, 0)
        x = mx.concatenate([batch_class_token, x], axis=1)

        return x

    def get_intermediate_layers(
        self,
        x: mx.array,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[mx.array, Tuple[mx.array, mx.array]]]:
        """Get intermediate layers outputs

        Args:
            x (mx.array): input mx.array of shape (batch_size, image_size, image_size, 3)
            n (Union[int, Sequence], optional): number of layers to take. Defaults to 1.
            return_class_token (bool, optional): return class token. Defaults to False.
            norm (bool, optional): apply normalization. Defaults to True.

        Returns:
            Tuple[Union[mx.array, Tuple[mx.array]]]: intermediate layers outputs
        """

        x = self._prepare_tokens(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.encoder.layers)  # noqa: F841
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

        outputs, attn_mat = self.encoder.get_intermediate_layers(x, blocks_to_take)

        if norm:
            outputs = [self.encoder.ln(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, w, h, _ = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens)), attn_mat
        return tuple(outputs), attn_mat

    def get_features(self, x: mx.array) -> mx.array:
        """Forward pass for feature

        Args:
            x (mx.array): input mx.array of shape (batch_size, image_size, image_size, 3)

        Returns:
            mx.array: output mx.array of shape (batch_size, hidden_dim)
        """

        x = self._prepare_tokens(x)
        x, attn_masks = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x, attn_masks

    def __call__(self, x: mx.array, attn_masks: bool = False) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]:
        """Forward pass.

        Args:
            x (mx.array): input mx.array of shape (batch_size, image_size, image_size, 3)
            attn_masks (bool, optional): whether to return attention masks. Defaults to False.

        Returns:
            Union[mx.array, Tuple[mx.array, List[mx.array]]]: output mx.array of shape (batch_size, num_classes) or Tuple of output mx.array of shape (batch_size, num_classes) and attention masks from each layer
        """
        # Reshape and permute the input tensor

        if self.head_type == "dino":
            intermediate_x, attn = self.get_intermediate_layers(x, n=self.n_last_blocks, return_class_token=False)
            x = mx.concatenate([x[:, 0] for x in intermediate_x], axis=1)
            if self.avgpool:
                x = mx.concatenate(
                    [x[:, 0] for x in intermediate_x] + [mx.mean(intermediate_x[-1][:, 1:], axis=1)], axis=1
                )
            else:
                x = mx.concatenate([x[:, 0] for x in intermediate_x], axis=1)

        else:
            x, attn = self.get_features(x)

        x = self.heads(x)
        if attn_masks:
            return x, attn
        else:
            return x


class DINOHead(nn.Module):
    """DINO Head for Vision Transformer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        use_relu: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, use_bias)
        self.bn = nn.BatchNorm(out_dim) if use_bn else nn.Identity()
        self.relu = nn.ReLU() if use_relu else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass

        Args:
            x (mx.array): input mx.array of shape (batch_size, hidden_dim)

        Returns:
            mx.array: output mx.array of shape (batch_size, hidden_dim)
        """
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
