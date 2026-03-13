import mlx.core as mx

from mlxim.data._utils import _default_collate_fn


def test_collate_with_labels():
    batch = [
        (mx.ones((3, 3, 3)), 0),
        (mx.ones((3, 3, 3)), 1),
    ]
    inputs, targets = _default_collate_fn(batch)
    assert inputs.shape == (2, 3, 3, 3)
    assert targets.shape == (2,)
    assert targets[0].item() == 0
    assert targets[1].item() == 1


def test_collate_without_labels():
    batch = [mx.ones((3, 3, 3)), mx.ones((3, 3, 3))]
    result = _default_collate_fn(batch)
    assert result.shape == (2, 3, 3, 3)
