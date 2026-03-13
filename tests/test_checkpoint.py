import os
from unittest.mock import patch

import mlx.core as mx

from mlxim.callbacks.checkpoint import ModelCheckpoint


def test_init_creates_directory(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path / "output"))
    assert os.path.isdir(cp.output_dir)


def test_best_val_empty(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path))
    assert cp.best_val is None


def test_update_history_within_capacity(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), save_top_k=3, mode="max")
    cp._update_history(val=0.5, epoch=0)
    cp._update_history(val=0.7, epoch=1)
    assert len(cp.history) == 2
    assert cp.best_val == 0.7


def test_update_history_full_replaces_worst(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), save_top_k=2, mode="max")
    cp._update_history(val=0.5, epoch=0)
    cp._update_history(val=0.7, epoch=1)
    # Now full, adding 0.9 should evict 0.5
    cp._update_history(val=0.9, epoch=2)
    assert len(cp.history) == 2
    vals = [h[0] for h in cp.history]
    assert 0.5 not in vals
    assert 0.9 in vals


def test_patience_increments_on_no_improvement(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), save_top_k=1, mode="max", patience=5)
    cp._update_history(val=0.9, epoch=0)
    assert cp.patience_count == 0
    # Worse value — should increment patience
    cp._update_history(val=0.1, epoch=1)
    assert cp.patience_count == 1


def test_patience_over(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), save_top_k=1, mode="max", patience=2)
    cp._update_history(val=0.9, epoch=0)
    cp._update_history(val=0.1, epoch=1)
    assert not cp.patience_over
    cp._update_history(val=0.1, epoch=2)
    assert cp.patience_over


def test_create_filename(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path))
    filename = cp._create_filename(epoch=5, metrics={"val_loss": 0.1234, "val_acc": 0.9876})
    assert filename.startswith("epoch=5-")
    assert "val_acc=0.9876" in filename
    assert "val_loss=0.1234" in filename
    assert filename.endswith(".npz")


def test_save_writes_file(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), save_top_k=3, mode="max")
    weights = {"layer.weight": mx.ones((2, 2))}
    metrics = {"val_acc": 0.95}
    cp._update_history(val=0.95, epoch=0)
    cp.save(epoch=0, metrics=metrics, weights=weights)
    files = os.listdir(cp.output_dir)
    assert len(files) == 1
    assert files[0].startswith("epoch=0")


def test_step_integration(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), monitor="val_acc", save_top_k=2, mode="max")
    weights = {"layer.weight": mx.ones((2, 2))}

    cp.step(epoch=0, metrics={"val_acc": 0.8}, weights=weights)
    cp.step(epoch=1, metrics={"val_acc": 0.9}, weights=weights)
    assert len(cp.history) == 2

    files = os.listdir(cp.output_dir)
    assert len(files) == 2


def test_mode_min(tmp_path):
    cp = ModelCheckpoint(output_dir=str(tmp_path), save_top_k=2, mode="min")
    cp._update_history(val=0.5, epoch=0)
    cp._update_history(val=0.3, epoch=1)
    # In min mode, best_val should be the lowest
    assert cp.best_val == 0.3
    # History sorted ascending (reverse=False for min)
    assert cp.history[0][0] == 0.3
