import mlx.core as mx

from mlxim.data import DataLoader, LabelFolderDataset
from mlxim.io import read_rgb
from mlxim.model import create_model
from mlxim.model._registry import MODEL_CONFIG
from mlxim.transform import ImageNetTransform
from mlxim.utils.imagenet import IMAGENET2012_CLASSES


def test_download_resnet():
    x = mx.random.uniform(shape=(1, 224, 224, 3))
    model = create_model("resnet18", strict=True)
    model.eval()
    out = model(x)
    assert out.shape == (1, 1000), f"Expected shape (1, 1000), got {out.shape}"


def test_download_vit():
    x = mx.random.uniform(shape=(1, 224, 224, 3))
    model = create_model("vit_base_patch16_224", strict=True)
    model.eval()
    out = model(x)
    assert out.shape == (1, 1000), f"Expected shape (1, 1000), got {out.shape}"


def test_inference_resnet():
    class_map = {0: "n01440764", 1: "n01514668", 2: "n01667778"}
    dataset = LabelFolderDataset(
        "tests/data",
        class_map=class_map,
        transform=ImageNetTransform(train=False, img_size=224, interpolation="bicubic"),
    )
    loader = DataLoader(dataset=dataset)
    model = create_model("resnet34", weights=False, strict=True)
    model.eval()
    for batch in loader:
        x, target = batch
        logits = model(x)

        assert logits.shape == (1, 1000), f"Expected shape (1, 1000), got {logits.shape}"


def test_inference_vit():
    class_map = {0: "n01440764", 1: "n01514668", 2: "n01667778"}
    dataset = LabelFolderDataset(
        "tests/data",
        class_map=class_map,
        transform=ImageNetTransform(train=False, img_size=512, interpolation="bicubic"),
    )
    loader = DataLoader(dataset=dataset)
    model = create_model("vit_large_patch16_512.swag_e2e", weights=False, strict=True)
    model.eval()
    for batch in loader:
        x, target = batch
        logits = model(x)
        assert logits.shape == (1, 1000), f"Expected shape (1, 1000), got {logits.shape}"


def test_features_resnet():
    x = mx.random.uniform(shape=(1, 224, 224, 3))
    model = create_model("resnet18", strict=True)
    model.eval()
    out = model.get_features(x)
    assert out.shape == (1, 512), f"Expected shape (1, 512), got {out.shape}"
