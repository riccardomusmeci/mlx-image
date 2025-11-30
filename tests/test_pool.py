import mlx.core as mx

from mlxim.model.layers.pool import AdaptiveAvgPool2d


def test_adaptive_avg_pool_2d_contiguous_patches():
    pool = AdaptiveAvgPool2d((2, 2))
    x = mx.arange(16, dtype=mx.float32).reshape(1, 4, 4, 1)

    out = pool(x)

    expected = mx.array([
        [
            [[2.5], [4.5]],
            [[10.5], [12.5]],
        ]
    ], dtype=mx.float32)

    assert out.shape == (1, 2, 2, 1)
    assert mx.allclose(out, expected)
