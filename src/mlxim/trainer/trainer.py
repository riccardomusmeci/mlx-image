import time
from typing import Callable, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from mlxim.callbacks import ModelCheckpoint
from mlxim.data import DataLoader

from ..metrics.classification import Accuracy
from ..model import get_weights


class Trainer:
    """Trainer class for training a model.

    Args:
        model (nn.Module): model to be trained
        optimizer (mlx.optimizers.Optimizer): optimizer to be used
        criterion (Callable): loss function
        train_loader (DataLoader): training data loader
        val_loader (Optional[DataLoader], optional): validation data loader. Defaults to None.
        log_every (int, optional): log every n iterations. Defaults to 100.
        device (mx.DeviceType, optional): device to be used. Defaults to mx.gpu.
        max_epochs (int, optional): maximum number of epochs. Defaults to 10.
        model_checkpoint (Optional[ModelCheckpoint], optional): model checkpoint. Defaults to None.
        top_k (Tuple[int, int], optional): top k accuracy. Defaults to (1, 5).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        train_loader: DataLoader,
        loss_fn_args: Optional[dict] = None,
        val_loader: Optional[DataLoader] = None,
        log_every: float = 0.1,
        device: mx.DeviceType = mx.gpu,
        max_epochs: int = 10,
        model_checkpoint: Optional[ModelCheckpoint] = None,
        top_k: Tuple[int, int] = (1, 5),
    ) -> None:
        mx.set_default_device(device)  # type: ignore

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_fn_args = loss_fn_args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_every = int(log_every * len(self.train_loader))
        self.max_epochs = max_epochs
        self.model_checkpoint = model_checkpoint
        self.top_k = top_k
        self.train_acc = Accuracy(top_k=top_k)
        self.val_acc = Accuracy(top_k=top_k)

        # things to update
        self.state = [self.model.state, self.optimizer.state]

    def _train_step(self, x: mx.array, target: mx.array) -> mx.array:
        """Perform a single training step.

        Args:
            x (mx.array): input data
            target (mx.array): target data

        Returns:
            mx.array: loss
        """
        logits = self.model(x)
        loss = mx.mean(self.loss_fn(logits, target, **self.loss_fn_args))  # type: ignore
        self.train_acc.update(logits, target)
        return loss

    def _val_step(self, x: mx.array, target: mx.array) -> mx.array:
        """Perform a single validation step.

        Args:
            x (mx.array): input data
            target (mx.array): target data

        Returns:
            mx.array: loss
        """
        logits = self.model(x)
        loss = mx.mean(self.loss_fn(logits, target, **self.loss_fn_args))  # type: ignore
        self.val_acc.update(logits, target)
        return loss

    def train_epoch(self) -> Tuple[float, Dict, float]:
        """Train the model for a single epoch.

        Args:
            epoch (int): current epoch

        Returns:
            Tuple[float, Dict, float]: mean loss, accuracy dict, mean throughput
        """
        self.model.train()
        epoch_loss = []
        throughput = []
        for batch_idx, batch in enumerate(self.train_loader):
            x, target = batch

            tic = time.perf_counter()
            train_step_fn = nn.value_and_grad(self.model, self._train_step)
            loss, grads = train_step_fn(x, target)
            self.optimizer.update(self.model, grads)
            mx.eval(self.state)
            toc = time.perf_counter()

            epoch_loss.append(loss.item())
            throughput.append(x.shape[0] / (toc - tic))

            if batch_idx % self.log_every == 0:
                print(
                    " | ".join(
                        [
                            f"> iter=[{batch_idx}/{len(self.train_loader)}]",
                            f"train_loss={np.mean(epoch_loss[-1]):.3f}",
                            f"train_throughput={np.mean(throughput[-1]):.2f} images/second",
                            f"lr={self.optimizer.state['learning_rate'].item():.5f}",
                        ]
                    )
                )

        epoch_acc = self.train_acc.compute()
        self.train_acc.reset()
        return np.mean(epoch_loss), epoch_acc, np.mean(throughput)

    def val_epoch(self) -> Tuple[float, Dict, float]:
        """Run the validation step for a single epoch.

        Returns:
            Tuple[float, float, float]: mean loss, mean accuracy, mean throughput
        """
        self.model.eval()
        epoch_loss = []
        throughput = []
        for _, batch in tqdm(enumerate(self.val_loader), total=len(self.val_loader)):  # type: ignore
            x, target = batch
            tic = time.perf_counter()
            loss = self._val_step(x, target)
            toc = time.perf_counter()
            epoch_loss.append(loss.item())
            throughput.append(x.shape[0] / (toc - tic))

        epoch_acc = self.val_acc.compute()
        self.val_acc.reset()
        return np.mean(epoch_loss), epoch_acc, np.mean(throughput)

    def train(self) -> None:
        """Train the model for the specified number of epochs."""

        for epoch in range(self.max_epochs):
            print(f"\n******** epoch {epoch}/{self.max_epochs} ********\n")
            tic = time.perf_counter()
            train_loss, train_acc, train_throughput = self.train_epoch()
            toc = time.perf_counter()
            print("training recap:")
            print(
                "> "
                + " | ".join(
                    [
                        f"epoch_time={toc-tic:.2f}s",
                        f"train_loss={train_loss:.3f}",
                        f"train_throughput={train_throughput:.2f} images/second",
                    ]
                    + [f"train_acc@{k}={train_acc[f'acc@{k}']}" for k in self.top_k]
                )
            )

            if self.val_loader is not None:
                print("running validation...")
                val_loss, val_acc, val_throughput = self.val_epoch()
                print(
                    "> "
                    + " | ".join(
                        [f"val_loss={val_loss:.3f}", f"val_throughput={val_throughput:.2f} images/second"]
                        + [f"val_acc@{k}={val_acc[f'acc@{k}']}" for k in self.top_k]
                    )
                )

                if self.model_checkpoint:
                    val_metrics = {"val_loss": val_loss} | {f"val_acc@{k}": val_acc[f"acc@{k}"] for k in self.top_k}
                    self.model_checkpoint.step(
                        epoch=epoch,
                        metrics=val_metrics,
                        weights=get_weights(self.model),
                    )
                    if self.model_checkpoint.patience_over:
                        print("\n*******************\n")
                        print(f"Early stopping at epoch {epoch}")
                        quit()
            else:
                if self.model_checkpoint:
                    train_metrics = {"train_loss": train_loss} | {
                        f"train_acc@{k}": train_acc[f"acc@{k}"] for k in self.top_k
                    }
                    self.model_checkpoint.step(
                        epoch=epoch,
                        metrics=train_metrics,
                        weights=get_weights(self.model),
                    )
                    if self.model_checkpoint.patience_over:
                        print("\n*******************\n")
                        print(f"Early stopping at epoch {epoch}")
                        quit()

            print("\n*******************\n")
