from mlxim.data import DataLoader, FolderDataset, LabelFolderDataset
from mlxim.io import read_rgb
from mlxim.transform import ImageNetTransform


def test_read_rgb():
    img = read_rgb("tests/data/n01440764/ILSVRC2012_val_00000293.JPEG")
    assert img.shape == (375, 500, 3), f"Expected shape (375, 500, 3), got {img.shape}"


def test_data_loader_no_labels():
    dataset = FolderDataset(
        "tests/data/n01440764", transform=ImageNetTransform(train=False, img_size=224, interpolation="bicubic")
    )
    loader = DataLoader(dataset=dataset)
    for batch in loader:
        assert len(batch) == 1, f"Expected length 1, got {len(batch)}"
        assert batch.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {batch.shape}"


def test_data_loader_with_labels():
    dataset = LabelFolderDataset(
        "tests/data",
        class_map={0: "n01440764", 1: "n01514668", 2: "n01667778"},
        transform=ImageNetTransform(train=False, img_size=224, interpolation="bicubic"),
    )
    loader = DataLoader(dataset=dataset)
    for batch in loader:
        assert len(batch) == 2, f"Expected length 2, got {len(batch)}"
        x, target = batch
        assert x.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {x.shape}"
        assert target.shape == (1,), f"Expected shape (1,), got {target.shape}"
        assert target[0] in [0, 1, 2], f"Expected target in [0, 1, 2], got {target[0]}"
