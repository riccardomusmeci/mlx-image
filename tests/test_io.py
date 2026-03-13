import os

import numpy as np
import pytest

from mlxim.io.image import read_rgb, resize_rgb, save_image


def test_read_rgb_cv2_engine(tmp_image_path):
    img = read_rgb(tmp_image_path, engine="cv2")
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_read_rgb_invalid_engine_fallback(tmp_image_path):
    img = read_rgb(tmp_image_path, engine="invalid_engine")
    assert img.shape[2] == 3


def test_read_rgb_missing_file():
    with pytest.raises(FileNotFoundError):
        read_rgb("/nonexistent/image.jpg")


def test_save_image(tmp_path, tmp_image):
    output_path = str(tmp_path / "subdir" / "output.jpg")
    save_image(tmp_image, output_path)
    assert os.path.exists(output_path)


def test_resize_rgb(tmp_image):
    resized = resize_rgb(tmp_image, w=32, h=48)
    assert resized.shape == (48, 32, 3)
