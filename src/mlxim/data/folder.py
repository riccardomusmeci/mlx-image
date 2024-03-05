import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from ..io import read_rgb
from ._base import EXTENSIONS, Dataset


class FolderDataset(Dataset):
    """FolderDataset is used to load images from a folder

    Args:
        root_dir (str): data dir
        transform (Callable, optional): set of data transformations. Defaults to None.
        engine (str, optional): image processing engine (pil or cv2). Defaults to "pil".
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    def __init__(
        self, root_dir: Union[Path, str], transform: Optional[Callable], engine: str = "pil", verbose: bool = True
    ) -> None:
        super().__init__()
        assert os.path.exists(root_dir), f"Folder with images {root_dir} does not exist."
        self.root_dir = root_dir
        self.images = [
            os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.split(".")[-1].lower() in EXTENSIONS
        ]
        self.transform = transform
        self.engine = engine
        if verbose:
            self.stats()

    def stats(self) -> None:
        """Print dataset stats"""
        print(" ------- FolderDataset stats -------")
        print(f"> root_dir: {self.root_dir}")
        print(f"> num_images: {len(self.images)}")
        print(" -------------------------------------")

    def __getitem__(self, index: int) -> mx.array:
        """Return image at index

        Args:
            index (int): image index

        Returns:
            mx.array: image
        """
        img_path = self.images[index]
        img = read_rgb(img_path, engine=self.engine)

        if self.transform is not None:
            img = self.transform(img)

        img = mx.array(img)

        return img

    def __len__(self) -> int:
        """Return number of samples in the dataset.

        Returns:
            int: number of samples in the dataset
        """
        return len(self.images)


class LabelFolderDataset(Dataset):
    """LabelFolderDataset is used to load images from a label folder structure.

    Args:
        root_dir (Union[Path, str]): data directory
        class_map (Union[Dict[int, Union[str, List[str]]], str, Path]): class map {e.g. {0: 'class_a', 1: ['class_b', 'class_c']}}
        transform (Optional[Callable], optional): set of data transformations. Defaults to None.
        engine (str, optional): image processing engine (pil or cv2). Defaults to "pil".
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        class_map: Union[Dict[int, Union[str, List[str]]], str, Path],
        transform: Optional[Callable] = None,
        engine: str = "pil",
        verbose: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(class_map, dict):
            with open(class_map) as f:
                self.class_map = json.load(f)
                self.class_map = {int(k): v for k, v in self.class_map.items()}
        else:
            self.class_map = class_map
        self.verbose = verbose
        # checking structure
        try:
            self._sanity_check(root_dir=root_dir)
        except Exception as e:
            raise e
        self.root_dir = root_dir
        self.engine = engine
        self.images, self.targets = self._load_samples()
        self.transform = transform
        if verbose:
            self.stats()

    def _sanity_check(self, root_dir: Union[Path, str]) -> None:
        """Check dataset structure.

        Args:
            root_dir (Union[Path, str]): data directory

        Raises:
            FileNotFoundError: if the data folder is not right based on the structure in class_map
            FileExistsError: if some label does not have images in its folder
        """
        for _k, labels in self.class_map.items():
            if not isinstance(labels, list):
                labels = [labels]
            for l in labels:
                label_dir = os.path.join(root_dir, l)
                if not (os.path.exists(label_dir)):
                    raise FileNotFoundError(f"Folder {label_dir} does not exist")
                if len(os.listdir(label_dir)) == 0:
                    raise FileExistsError(f"Folder {label_dir} is empty.")

        if self.verbose:
            print("> [INFO] dataset sanity check OK")

    def _load_samples(self) -> Tuple[List[str], List[int]]:
        """Load samples and targets.

        Returns:
            Tuple[List[str], List[int]]: images + targets
        """
        paths = []
        targets = []
        for c, labels in self.class_map.items():
            if isinstance(labels, str):
                labels = [labels]
            c_images, c_targets = [], []
            for label in labels:
                label_dir = os.path.join(self.root_dir, label)
                c_images += [
                    os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.split(".")[-1].lower() in EXTENSIONS
                ]
            c_targets += [int(c)] * len(c_images)
            paths += c_images
            targets += c_targets

        return paths, targets

    def stats(self) -> None:
        """Print stats of the dataset."""
        print(" ------- LabelFolderDataset stats -------")
        unique, counts = np.unique(self.targets, return_counts=True)
        num_samples = len(self.targets)
        for k in range(len(unique)):
            classes = self.class_map[k]
            if isinstance(classes, str):
                classes = [classes]
            print(f"\t- label {k} - {classes} - {counts[k]}/{num_samples} -> {100 * counts[k] / num_samples:.3f}%")
        print(" -------------------------------------")

    def __getitem__(self, index: int) -> Tuple[mx.array, int]:
        """Return a tuple (image, target) given an index.

        Args:
            index (int): sample index

        Returns:
            Tuple[mx.array, int]: image, target
        """
        img_path = self.images[index]
        target = self.targets[index]

        img = read_rgb(img_path, engine=self.engine)

        if self.transform is not None:
            img = self.transform(img)

        img = mx.array(img)

        return img, target

    def __len__(self) -> int:
        """Return number of samples in the dataset.

        Returns:
            int: number of samples in the dataset
        """
        return len(self.images)
