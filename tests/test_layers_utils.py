from mlxim.model.layers.utils import _make_divisible, to_1tuple, to_2tuple, to_3tuple, to_4tuple, to_ntuple


def test_to_1tuple():
    assert to_1tuple(5) == (5,)
    assert to_1tuple((5,)) == (5,)


def test_to_2tuple():
    assert to_2tuple(5) == (5, 5)
    assert to_2tuple((3, 4)) == (3, 4)


def test_to_3tuple():
    assert to_3tuple(1) == (1, 1, 1)


def test_to_4tuple():
    assert to_4tuple(2) == (2, 2, 2, 2)


def test_to_ntuple():
    to_5tuple = to_ntuple(5)
    assert to_5tuple(7) == (7, 7, 7, 7, 7)


def test_make_divisible():
    assert _make_divisible(7, 8) == 8
    assert _make_divisible(16, 8) == 16


def test_make_divisible_min_value():
    assert _make_divisible(1, 8, min_value=16) == 16
