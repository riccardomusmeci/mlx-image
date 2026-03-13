"""Tests for model utility functions: num_params, get_weights, save_weights, load_weights."""

import os
import tempfile

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxim.model._factory import create_model
from mlxim.model._utils import get_weights, load_weights, num_params, save_weights


def test_num_params():
    model = create_model("resnet18", weights=False, num_classes=10)
    n = num_params(model)
    assert isinstance(n, int)
    assert n > 0


def test_get_weights_returns_dict():
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)
    assert isinstance(weights, dict)
    assert len(weights) > 0
    for k, v in weights.items():
        assert isinstance(k, str)
        assert isinstance(v, mx.array)


def test_save_and_load_weights_npz():
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "weights.npz")
        save_weights(weights, path)
        assert os.path.exists(path)

        # Load back into fresh model
        model2 = create_model("resnet18", weights=False, num_classes=10)
        model2 = load_weights(model2, path, strict=True)
        weights2 = get_weights(model2)

        for k in weights:
            assert k in weights2
            assert mx.array_equal(weights[k], weights2[k])


def test_save_and_load_weights_safetensors():
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.safetensors")
        save_weights(weights, path)
        assert os.path.exists(path)

        model2 = create_model("resnet18", weights=False, num_classes=10)
        model2 = load_weights(model2, path, strict=False)
        weights2 = get_weights(model2)
        assert len(weights2) > 0


def test_load_weights_strict_missing_key():
    """strict=True should raise when weights file is missing keys."""
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)
    # Remove a key to simulate missing weights
    first_key = next(iter(weights))
    del weights[first_key]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "partial.npz")
        save_weights(weights, path)

        model2 = create_model("resnet18", weights=False, num_classes=10)
        with pytest.raises(ValueError, match="Missing keys"):
            load_weights(model2, path, strict=True)


def test_load_weights_strict_extra_key():
    """strict=True should raise when weights file has extra keys."""
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)
    # Add extra key
    weights["nonexistent.layer.weight"] = mx.zeros((3, 3))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "extra.npz")
        save_weights(weights, path)

        model2 = create_model("resnet18", weights=False, num_classes=10)
        with pytest.raises(ValueError, match="extra keys"):
            load_weights(model2, path, strict=True)


def test_load_weights_non_strict_missing():
    """strict=False should tolerate missing keys."""
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)
    first_key = next(iter(weights))
    del weights[first_key]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "partial.npz")
        save_weights(weights, path)

        model2 = create_model("resnet18", weights=False, num_classes=10)
        # Should not raise
        model2 = load_weights(model2, path, strict=False, verbose=True)


def test_load_weights_verbose(capsys):
    model = create_model("resnet18", weights=False, num_classes=10)
    weights = get_weights(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "weights.npz")
        save_weights(weights, path)

        model2 = create_model("resnet18", weights=False, num_classes=10)
        load_weights(model2, path, strict=True, verbose=True)
        captured = capsys.readouterr()
        assert "Loading weights" in captured.out


def test_save_weights_creates_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "nested", "weights.npz")
        weights = {"layer.weight": mx.zeros((3, 3))}
        save_weights(weights, path)
        assert os.path.exists(path)
