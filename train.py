import argparse
import json
import os
import time
from functools import partial
from shutil import copy2

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxim.callbacks import ModelCheckpoint
from mlxim.data import DataLoader, LabelFolderDataset
from mlxim.io.config import load_config
from mlxim.metrics.classification import Accuracy
from mlxim.model import create_model
from mlxim.trainer.trainer import Trainer
from mlxim.transform import ImageNetTransform
from mlxim.utils.time import now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config", type=str, default="config/train.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    mx.random.seed(config["seed"])

    # creating output dir + copying config
    output_dir = os.path.join(config["output"], now())
    os.makedirs(output_dir, exist_ok=True)
    copy2(args.config, output_dir)

    # define datasets
    train_dataset = LabelFolderDataset(
        root_dir=os.path.join(config["data_dir"], "train"),
        transform=ImageNetTransform(train=True, **config["transform"]),
        **config["dataset"],
    )

    val_dataset = LabelFolderDataset(
        root_dir=os.path.join(config["data_dir"], "val"),
        transform=ImageNetTransform(train=False, **config["transform"]),
        **config["dataset"],
    )

    # define loader + optimizer

    # define loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        **config["loader"],
    )

    config["loader"]["shuffle"] = False
    val_loader = DataLoader(dataset=val_dataset, **config["loader"])

    # define model checkpoint
    model_checkpoint = ModelCheckpoint(output_dir=os.path.join(config["output"], now()), **config["model_checkpoint"])

    model = create_model(
        num_classes=len(train_dataset.class_map),
        **config["model"],
    )

    decay_steps = len(train_loader) * config["trainer"]["max_epochs"]
    lr_schedule = optim.cosine_decay(init=1e-3, decay_steps=decay_steps)
    optimizer = optim.SGD(learning_rate=lr_schedule)

    # trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.losses.cross_entropy,
        train_loader=train_loader,
        val_loader=val_loader,
        model_checkpoint=model_checkpoint,
        **config["trainer"]
    )

    trainer.train()

