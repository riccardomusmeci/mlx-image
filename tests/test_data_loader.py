from mlxim.data import DataLoader, FolderDataset, LabelFolderDataset
from mlxim.transform import ImageNetTransform


def _make_loader(batch_size=1, drop_last=False, shuffle=False):
    dataset = LabelFolderDataset(
        "tests/data",
        class_map={0: "n01440764", 1: "n01514668", 2: "n01667778"},
        transform=ImageNetTransform(train=False, img_size=224, interpolation="bicubic"),
    )
    return DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)


def test_loader_drop_last():
    loader = _make_loader(batch_size=2, drop_last=True)
    batches = list(loader)
    for x, _target in batches:
        assert x.shape[0] == 2, "All batches should have batch_size=2 with drop_last=True"


def test_loader_shuffle():
    loader = _make_loader(shuffle=True)
    batches = list(loader)
    assert len(batches) > 0, "Should produce batches even with shuffle=True"


def test_loader_len():
    loader = _make_loader(batch_size=2, drop_last=False)
    expected = (len(loader.indices) + 1) // 2  # ceil division
    assert len(loader) == expected


def test_loader_batch_size():
    loader = _make_loader(batch_size=3)
    for x, _target in loader:
        assert x.shape[0] <= 3
