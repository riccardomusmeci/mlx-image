import mlx.core as mx
import mlx.nn as nn

from mlxim.model.layers.misc import Conv2dNormActivation, ConvNormActivation, HardSigmoid, SqueezeExcitation


def test_conv_norm_activation():
    block = ConvNormActivation(3, 16, kernel_size=3, stride=1)
    x = mx.ones((1, 8, 8, 3))
    out = block(x)
    assert out.shape[0] == 1
    assert out.shape[-1] == 16


def test_conv2d_norm_activation():
    block = Conv2dNormActivation(3, 32, kernel_size=3)
    x = mx.ones((1, 8, 8, 3))
    out = block(x)
    assert out.shape[-1] == 32


def test_conv_norm_activation_no_norm():
    block = ConvNormActivation(3, 16, kernel_size=1, norm_layer=None)
    x = mx.ones((1, 4, 4, 3))
    out = block(x)
    assert out.shape[-1] == 16


def test_conv_norm_activation_no_activation():
    block = ConvNormActivation(3, 16, kernel_size=1, activation_layer=None)
    x = mx.ones((1, 4, 4, 3))
    out = block(x)
    assert out.shape[-1] == 16


def test_squeeze_excitation():
    se = SqueezeExcitation(input_channels=16, squeeze_channels=4)
    x = mx.ones((1, 4, 4, 16))
    out = se(x)
    assert out.shape == (1, 4, 4, 16)


def test_hard_sigmoid():
    hs = HardSigmoid()
    x = mx.array([-10.0, -3.0, 0.0, 3.0, 10.0])
    out = hs(x)
    assert mx.all(out >= 0.0)
    assert mx.all(out <= 1.0)
    # At x=0: 0/6 + 0.5 = 0.5
    assert mx.isclose(out[2], mx.array(0.5))
