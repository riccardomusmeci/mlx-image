"""Tests for Trainer class — exercises init, train_step, val_step without full training."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxim.data import DataLoader
from mlxim.data._base import Dataset
from mlxim.trainer.trainer import Trainer


class _TinyModel(nn.Module):
    """Minimal model for testing trainer mechanics."""

    def __init__(self, in_features=8, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def __call__(self, x):
        return self.fc(x)


class _FakeDataset(Dataset):
    """Minimal dataset that returns (features, labels)."""

    def __init__(self, n=16, in_features=8, num_classes=4):
        self.data = [(mx.random.normal((in_features,)), mx.array(i % num_classes)) for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


def _make_trainer(with_val=False, max_epochs=1):
    model = _TinyModel()
    optimizer = optim.SGD(learning_rate=0.01)
    loss_fn = nn.losses.cross_entropy

    train_ds = _FakeDataset(n=16)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    val_loader = None
    if with_val:
        val_ds = _FakeDataset(n=8)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        log_every=1.0,
        device=mx.cpu,
        top_k=(1,),
    )


def test_trainer_init():
    trainer = _make_trainer()
    assert trainer.max_epochs == 1
    assert trainer.val_loader is None
    assert trainer.model is not None


def test_trainer_train_step():
    trainer = _make_trainer()
    x = mx.random.normal((4, 8))
    target = mx.array([0, 1, 2, 3])
    loss = trainer._train_step(x, target)
    mx.eval(loss)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_trainer_val_step():
    trainer = _make_trainer(with_val=True)
    x = mx.random.normal((4, 8))
    target = mx.array([0, 1, 2, 3])
    loss = trainer._val_step(x, target)
    mx.eval(loss)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_trainer_train_epoch():
    trainer = _make_trainer()
    mean_loss, acc_dict, throughput = trainer.train_epoch()
    assert isinstance(mean_loss, float)
    assert isinstance(acc_dict, dict)
    assert "acc@1" in acc_dict
    assert throughput > 0


def test_trainer_val_epoch():
    trainer = _make_trainer(with_val=True)
    mean_loss, acc_dict, throughput = trainer.val_epoch()
    assert isinstance(mean_loss, float)
    assert isinstance(acc_dict, dict)
    assert throughput > 0


def test_trainer_full_train_loop():
    trainer = _make_trainer(with_val=True, max_epochs=2)
    trainer.train()
    # If we get here without error, the full loop works
