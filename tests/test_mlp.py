import mlx.core as mx
import mlx.nn as nn

from mlxim.model.layers.mlp import MLP


def test_mlp_forward_shape():
    mlp = MLP(in_channels=8, hidden_channels=[16, 4])
    x = mx.ones((2, 8))
    out = mlp(x)
    assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"


def test_mlp_with_norm_layer():
    mlp = MLP(in_channels=8, hidden_channels=[16, 4], norm_layer=nn.LayerNorm)
    x = mx.ones((2, 8))
    out = mlp(x)
    assert out.shape == (2, 4)
    # Check that norm layers exist
    has_norm = any(isinstance(layer, nn.LayerNorm) for layer in mlp.layers)
    assert has_norm


def test_mlp_single_layer():
    mlp = MLP(in_channels=4, hidden_channels=[2])
    x = mx.ones((3, 4))
    out = mlp(x)
    assert out.shape == (3, 2)


def test_mlp_multi_layer():
    mlp = MLP(in_channels=4, hidden_channels=[8, 16, 2])
    x = mx.ones((1, 4))
    out = mlp(x)
    assert out.shape == (1, 2)
