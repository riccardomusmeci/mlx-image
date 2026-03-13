import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


@pytest.fixture
def tmp_image():
    """Create a small random test image as numpy array (H, W, 3)."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def tmp_image_path(tmp_path, tmp_image):
    """Save a test image to tmp_path and return its path."""
    import cv2

    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(tmp_image, cv2.COLOR_RGB2BGR))
    return str(path)


@pytest.fixture
def sample_logits():
    """Sample logits for 4 samples, 10 classes."""
    return mx.array(
        [
            [0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # pred: class 2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],  # pred: class 9
            [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # pred: class 1
            [0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # pred: class 3
        ]
    )


@pytest.fixture
def sample_targets():
    """Targets matching sample_logits: classes 2, 9, 1, 3 → all correct."""
    return mx.array([2, 9, 1, 3])


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, in_features=4, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def __call__(self, x):
        return self.linear(x)


@pytest.fixture
def tiny_model():
    return TinyModel()
