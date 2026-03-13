import mlx.core as mx

from mlxim.model.layers.pool import AdaptiveAvgPool2d


def test_pool_1x1():
    pool = AdaptiveAvgPool2d((1, 1))
    x = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4, 1)
    out = pool(x)
    assert out.shape == (1, 1, 1, 1)
    expected_mean = mx.mean(x)
    assert mx.allclose(out.reshape(-1), expected_mean.reshape(-1))


def test_pool_same_size():
    pool = AdaptiveAvgPool2d((4, 4))
    x = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4, 1)
    out = pool(x)
    assert out.shape == (1, 4, 4, 1)
    assert mx.array_equal(out, x)


def test_pool_multi_channel():
    pool = AdaptiveAvgPool2d((2, 2))
    x = mx.arange(48, dtype=mx.float32).reshape(1, 4, 4, 3)
    out = pool(x)
    assert out.shape == (1, 2, 2, 3)
