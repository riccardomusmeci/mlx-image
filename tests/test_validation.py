"""Tests for ValidationResults utility class."""

import os
import tempfile

import pandas as pd

from mlxim.utils.validation import ValidationResults


def _make_csv(tmpdir, rows=None):
    """Create a temp CSV file with optional initial rows."""
    path = os.path.join(tmpdir, "results.csv")
    if rows is None:
        rows = []
    cols = ["model", "acc@1", "acc@5", "param_count", "img_size", "crop_pct", "interpolation", "engine"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)
    return path


def test_init_loads_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_csv(tmpdir)
        vr = ValidationResults(path=path)
        assert isinstance(vr.results, pd.DataFrame)
        assert len(vr.results) == 0


def test_update_adds_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_csv(tmpdir)
        vr = ValidationResults(path=path)
        vr.update(
            model_name="resnet18",
            acc_1=0.80634,
            acc_5=0.95012,
            param_count=11.7,
            img_size=224,
            crop_pct=0.875,
            interpolation="bilinear",
            engine="pil",
        )
        assert len(vr.results) == 1
        assert vr.results.iloc[0]["model"] == "resnet18"
        assert vr.results.iloc[0]["acc@1"] == 0.80634


def test_update_multiple_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_csv(tmpdir)
        vr = ValidationResults(path=path)
        vr.update("resnet18", 0.80, 0.95, 11.7, 224, 0.875, "bilinear", "pil")
        vr.update("resnet50", 0.82, 0.96, 25.6, 224, 0.875, "bicubic", "cv2")
        assert len(vr.results) == 2


def test_save_writes_sorted_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_csv(tmpdir)
        vr = ValidationResults(path=path)
        vr.update("resnet18", 0.80, 0.95, 11.7, 224, 0.875, "bilinear", "pil")
        vr.update("resnet50", 0.82, 0.96, 25.6, 224, 0.875, "bicubic", "cv2")
        vr.save()

        # Re-read and verify sorted by acc@1 descending
        df = pd.read_csv(path)
        assert len(df) == 2
        assert df.iloc[0]["model"] == "resnet50"  # higher acc@1 first


def test_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_csv(tmpdir)
        vr = ValidationResults(path=path)
        vr.update("vit_base", 0.81, 0.95, 86.6, 224, 0.875, "bicubic", "pil")
        vr.save()

        vr2 = ValidationResults(path=path)
        assert len(vr2.results) == 1
        assert vr2.results.iloc[0]["model"] == "vit_base"
