import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from ..io import read_rgb
from ._base import Dataset


class ImageFolderDataset(Dataset):
    """Image Classification Dataset init (image folder dataset)

    Args:
        root_dir (str): data dir
        class_map (Dict[int, Union[str, List[str]]], optional): class map {e.g. {0: 'class_a', 1: ['class_b', 'class_c']}}
        max_samples_per_class (int, optional): max number of samples for each class in the dataset. Defaults to None.
        transform (Callable, optional): set of data transformations. Defaults to None.
        verbose (bool, optional): verbose mode. Defaults to True.

    Raises:
        e: if some error occurs while checking dataset structure
    """

    EXTENSIONS = (
        "jpg",
        "jpeg",
        "png",
        "ppm",
        "bmp",
        "pgm",
        "tif",
        "tiff",
        "webp",
    )

    def __init__(
        self,
        root_dir: Union[Path, str],
        class_map: Dict[int, Union[str, List[str]]],
        transform: Optional[Callable] = None,
        engine: str = "pil",
        verbose: bool = True,
    ) -> None:
        super().__init__()

        assert isinstance(class_map, dict), "class_map must be a dict (e.g. {0: [car, trunk], 1: airplane})"
        self.verbose = verbose
        # checking structure
        try:
            self._sanity_check(root_dir=root_dir, class_map=class_map)
        except Exception as e:
            raise e
        self.root_dir = root_dir
        self.class_map = class_map
        self.engine = engine
        self.images, self.targets = self._load_samples()
        self.transform = transform
        self.stats()

    def _sanity_check(self, root_dir: Union[Path, str], class_map: Dict[int, Union[str, List[str]]]) -> None:
        """Check dataset structure.

        Args:
            root_dir (Union[Path, str]): data directory
            class_map (Dict[int, Union[str, List[str]]]): class map {e.g. {0: 'class_a', 1: ['class_b', 'class_c']}}

        Raises:
            FileNotFoundError: if the data folder is not right based on the structure in class_map
            FileExistsError: if some label does not have images in its folder
        """
        for _k, labels in class_map.items():
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
                    os.path.join(label_dir, f)
                    for f in os.listdir(label_dir)
                    if f.split(".")[-1].lower() in self.EXTENSIONS
                ]
            c_targets += [c] * len(c_images)
            paths += c_images
            targets += c_targets

        return paths, targets

    def stats(self) -> None:
        """Print stats of the dataset."""
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
