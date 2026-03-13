"""Tests for load_weights Conv2d weight transposition fix.

Verifies that PyTorch-format Conv2d weights (O,I,H,W) are automatically
transposed to MLX format (O,H,W,I) when shapes mismatch.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlxim.model._utils import get_weights, load_weights, save_weights


class _TinyConvModel(nn.Module):
    """Minimal model with a single Conv2d for testing weight loading.

    Uses in_channels=8, kernel_size=5 so that MLX shape (O,5,5,8) differs
    from PyTorch shape (O,8,5,5) — making transposition testable.
    """

    def __init__(self):
        super().__init__()
        # MLX Conv2d stores weights as (O, H, W, I)
        self.conv = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm(16)

    def __call__(self, x):
        return self.bn(self.conv(x))


class TestLoadWeightsTransposition:
    """Test automatic Conv2d weight transposition during load."""

    def test_matching_shapes_no_transpose(self, tmp_path):
        """Weights already in MLX format should load directly."""
        model = _TinyConvModel()
        original = get_weights(model)

        # Save weights in MLX format
        weights_path = str(tmp_path / "weights.safetensors")
        save_weights(original, weights_path)

        # Create fresh model and load
        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)

        # Verify shapes match
        loaded = get_weights(model2)
        for k in original:
            assert loaded[k].shape == original[k].shape, f"Shape mismatch for {k}"

    def test_pytorch_conv2d_transposed(self, tmp_path):
        """PyTorch-format Conv2d weights (O,I,H,W) should be auto-transposed to (O,H,W,I)."""
        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        # Get the conv weight shape in MLX format
        conv_key = "conv.weight"
        assert conv_key in mlx_weights
        mlx_shape = mlx_weights[conv_key].shape
        assert len(mlx_shape) == 4  # (O, H, W, I)

        # Create PyTorch-format weights: transpose (O,H,W,I) -> (O,I,H,W)
        pt_conv_weight = mx.transpose(mlx_weights[conv_key], (0, 3, 1, 2))
        assert pt_conv_weight.shape != mlx_shape  # Confirm they differ

        # Build a weights dict with PyTorch-format conv weight
        fake_weights = dict(mlx_weights)
        fake_weights[conv_key] = pt_conv_weight

        # Save these "PyTorch-format" weights
        weights_path = str(tmp_path / "pt_weights.safetensors")
        save_weights(fake_weights, weights_path)

        # Load into fresh model — should auto-transpose
        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)

        loaded = get_weights(model2)
        assert loaded[conv_key].shape == mlx_shape, f"Expected {mlx_shape}, got {loaded[conv_key].shape}"

    def test_transpose_preserves_values(self, tmp_path):
        """Transposed weights should have correct values, not random init."""
        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        conv_key = "conv.weight"
        original_mlx = mlx_weights[conv_key]

        # Transpose to "PyTorch" format and save
        pt_format = mx.transpose(original_mlx, (0, 3, 1, 2))
        fake_weights = dict(mlx_weights)
        fake_weights[conv_key] = pt_format

        weights_path = str(tmp_path / "pt_values.safetensors")
        save_weights(fake_weights, weights_path)

        # Load and verify values match original
        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)
        loaded = get_weights(model2)

        # After auto-transpose, values should match original MLX weights
        assert mx.allclose(loaded[conv_key], original_mlx, atol=1e-6).item(), (
            "Transposed weight values don't match original"
        )

    def test_untransposable_4d_stays_random(self, tmp_path):
        """4D weights that can't be fixed by transposition should keep model init."""
        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        conv_key = "conv.weight"
        original_shape = mlx_weights[conv_key].shape  # e.g. (16, 3, 3, 3)

        # Create a 4D weight with incompatible shape (no transpose will fix it)
        bad_weight = mx.ones((99, 99, 99, 99))
        fake_weights = dict(mlx_weights)
        fake_weights[conv_key] = bad_weight

        weights_path = str(tmp_path / "bad_weights.safetensors")
        save_weights(fake_weights, weights_path)

        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)
        loaded = get_weights(model2)

        # Should keep original model shape (random init)
        assert loaded[conv_key].shape == original_shape

    def test_strict_raises_on_mismatch(self, tmp_path):
        """In strict mode, untransposable shape mismatch should raise ValueError."""
        import pytest

        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        conv_key = "conv.weight"
        fake_weights = dict(mlx_weights)
        fake_weights[conv_key] = mx.ones((99, 99, 99, 99))

        weights_path = str(tmp_path / "strict_weights.safetensors")
        save_weights(fake_weights, weights_path)

        model2 = _TinyConvModel()
        with pytest.raises(ValueError, match="Expected shape"):
            load_weights(model2, weights_path, strict=True)

    def test_1d_shape_mismatch(self, tmp_path):
        """1D weights with different sizes should fall back to model init."""
        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        # Find a 1D weight (bias or batch norm)
        key_1d = None
        for k, v in mlx_weights.items():
            if len(v.shape) == 1:
                key_1d = k
                break

        if key_1d is None:
            return  # No 1D weights to test

        original_shape = mlx_weights[key_1d].shape
        fake_weights = dict(mlx_weights)
        fake_weights[key_1d] = mx.ones((original_shape[0] + 10,))

        weights_path = str(tmp_path / "1d_weights.safetensors")
        save_weights(fake_weights, weights_path)

        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)
        loaded = get_weights(model2)
        assert loaded[key_1d].shape == original_shape

    def test_missing_keys_non_strict(self, tmp_path):
        """Missing keys in non-strict mode should not crash."""
        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        # Remove some keys
        partial = {k: v for i, (k, v) in enumerate(mlx_weights.items()) if i == 0}

        weights_path = str(tmp_path / "partial.safetensors")
        save_weights(partial, weights_path)

        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)
        # Should not crash

    def test_extra_keys_non_strict(self, tmp_path):
        """Extra keys in weights file in non-strict mode should be ignored."""
        model = _TinyConvModel()
        mlx_weights = get_weights(model)

        # Add extra key
        mlx_weights["extra.bogus.weight"] = mx.ones((5, 5))

        weights_path = str(tmp_path / "extra.safetensors")
        save_weights(mlx_weights, weights_path)

        model2 = _TinyConvModel()
        model2 = load_weights(model2, weights_path, strict=False, verbose=True)
        # Should not crash
