import mlx.core as mx
import numpy as np

from mlxim.model.layers.functional import dropout, normalize, roll


def test_roll_single_axis():
    # roll splits at position `shift` and reverses halves: [shift:] + [:shift]
    x = mx.array([[1, 2, 3], [4, 5, 6]])
    result = roll(x, shifts=[1], axes=[1])
    expected = mx.array([[2, 3, 1], [5, 6, 4]])
    assert mx.array_equal(result, expected)


def test_roll_multi_axis():
    x = mx.array([[1, 2, 3], [4, 5, 6]])
    # First roll axis=0 shift=1: [[4,5,6],[1,2,3]]
    # Then roll axis=1 shift=1: [[5,6,4],[2,3,1]]
    result = roll(x, shifts=[1, 1], axes=[0, 1])
    expected = mx.array([[5, 6, 4], [2, 3, 1]])
    assert mx.array_equal(result, expected)


def test_roll_zero_shift():
    x = mx.array([1, 2, 3, 4])
    result = roll(x, shifts=[0], axes=[0])
    assert mx.array_equal(result, x)


def test_roll_negative_shift():
    x = mx.array([1, 2, 3, 4])
    # shift=-1 → converted to shape+shift = 4+(-1)=3, then %4=3
    # splits at [3]: [1,2,3] + [4] → reversed: [4] + [1,2,3] = [4,1,2,3]
    result = roll(x, shifts=[-1], axes=[0])
    expected = mx.array([4, 1, 2, 3])
    assert mx.array_equal(result, expected)


def test_normalize_unit_norm():
    x = mx.array([[3.0, 4.0], [1.0, 0.0]])
    result = normalize(x, axis=1)
    norms = mx.sqrt(mx.sum(result * result, axis=1))
    assert mx.allclose(norms, mx.ones(2), atol=1e-4)


def test_normalize_zero_input():
    x = mx.zeros((2, 3))
    result = normalize(x, axis=1)
    # Should not produce NaN due to eps
    assert not mx.any(mx.isnan(result))


def test_dropout_training_false():
    x = mx.ones((4, 4))
    result = dropout(x, p=0.5, training=False)
    assert mx.array_equal(result, x)


def test_dropout_p_zero():
    x = mx.ones((4, 4))
    result = dropout(x, p=0.0, training=True)
    assert mx.array_equal(result, x)


def test_dropout_training_true():
    mx.random.seed(42)
    x = mx.ones((100, 100))
    result = dropout(x, p=0.5, training=True)
    # Some values should be zero
    num_zeros = mx.sum(mx.array(result == 0)).item()
    assert num_zeros > 0, "Expected some values to be zeroed by dropout"
    # Non-zero values should be scaled by 1/(1-p) = 2.0
    non_zero_mask = result > 0
    non_zero_vals = result * non_zero_mask
    # All non-zero values should be 2.0 (= 1.0 / (1 - 0.5))
    scaled_expected = mx.where(non_zero_mask, mx.full(result.shape, 2.0), mx.zeros(result.shape))
    assert mx.allclose(non_zero_vals, scaled_expected, atol=1e-5)
