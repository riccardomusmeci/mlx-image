import numpy as np

from mlxim.transform import ImageNetTransform


def test_train_transform_shape(tmp_image):
    transform = ImageNetTransform(train=True, img_size=224)
    result = transform(tmp_image)
    assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"


def test_eval_transform_shape(tmp_image):
    transform = ImageNetTransform(train=False, img_size=224)
    result = transform(tmp_image)
    assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"


def test_eval_transform_squash_mode(tmp_image):
    transform = ImageNetTransform(train=False, img_size=224, crop_mode="squash")
    result = transform(tmp_image)
    assert result.shape == (224, 224, 3)


def test_eval_transform_border_mode(tmp_image):
    transform = ImageNetTransform(train=False, img_size=224, crop_mode="border")
    result = transform(tmp_image)
    assert result.shape == (224, 224, 3)


def test_transform_normalization(tmp_image):
    transform = ImageNetTransform(train=False, img_size=224)
    result = transform(tmp_image)
    # Normalized output should have values roughly centered around 0
    assert result.dtype == np.float32
    assert np.abs(result.mean()) < 2.0, "Mean should be roughly near 0 after normalization"


def test_transform_tuple_img_size(tmp_image):
    transform = ImageNetTransform(train=False, img_size=(128, 128))
    result = transform(tmp_image)
    assert result.shape == (128, 128, 3)
