import mlx.core as mx
import pytest

from mlxim.model.layers._ops import _make_divisible, stochastic_depth


def test_make_divisible_basic():
    assert _make_divisible(7, 8) == 8
    assert _make_divisible(16, 8) == 16
    assert _make_divisible(17, 8) == 16


def test_make_divisible_min_value():
    assert _make_divisible(1, 8, min_value=16) == 16


def test_make_divisible_round_down_protection():
    # v=9, divisor=8 → int(9 + 4) // 8 * 8 = 8
    # But 8 < 0.9 * 9 = 8.1 → bump up to 16
    assert _make_divisible(9, 8) == 16


def test_stochastic_depth_not_training():
    x = mx.ones((2, 4))
    result = stochastic_depth(x, p=0.5, mode="batch", training=False)
    assert mx.array_equal(result, x)


def test_stochastic_depth_p_zero():
    x = mx.ones((2, 4))
    result = stochastic_depth(x, p=0.0, mode="batch", training=True)
    assert mx.array_equal(result, x)


def test_stochastic_depth_batch_mode():
    mx.random.seed(0)
    x = mx.ones((4, 8))
    result = stochastic_depth(x, p=0.5, mode="batch", training=True)
    assert result.shape == x.shape


def test_stochastic_depth_row_mode():
    mx.random.seed(0)
    x = mx.ones((4, 8))
    result = stochastic_depth(x, p=0.5, mode="row", training=True)
    assert result.shape == x.shape


def test_stochastic_depth_invalid_p():
    x = mx.ones((2, 4))
    with pytest.raises(ValueError, match="Drop probability"):
        stochastic_depth(x, p=1.5, mode="batch")


def test_stochastic_depth_invalid_mode():
    x = mx.ones((2, 4))
    with pytest.raises(ValueError, match="mode"):
        stochastic_depth(x, p=0.5, mode="invalid")
