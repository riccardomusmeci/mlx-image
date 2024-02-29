from queue import Queue
from threading import Thread
from typing import Any, Callable

import numpy as np
import torch

from ._base import Dataset
from ._utils import _default_collate_fn


# TODO: must be improved: add sampler support and other features
class DataLoader:
    """custom DataLoader (similar to PyTorch but easier to read).

    Args:
        dataset: the custom Dataset class
        batch_size: the batch size. Defaults to 1.
        shuffle: whether to shuffle the dataset. Defaults to False.
        num_workers: the number of worker threads. Defaults to 0.
        collate_fn: the collate function to be used. Defaults to _default_collate_fn.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Callable = _default_collate_fn,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_queue = Queue()  # type: ignore
        self.index_queue = Queue()  # type: ignore
        self.stop_token = object()
        self.collate_fn = collate_fn

        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def start_workers(self) -> None:
        """Start the worker threads."""
        for _ in range(self.num_workers):
            t = Thread(target=self._process_data)
            t.daemon = True
            t.start()

    def _process_data(self) -> None:
        """Process the data and put it into the data queue."""
        while True:
            index = self.index_queue.get()
            if index is self.stop_token:
                break
            data = self.dataset[index]
            self.data_queue.put(data)

    def __iter__(self) -> Any:
        """Iterate over the DataLoader.

        Yields:
            Any: the batch of data
        """
        self.batch_indices = []  # type: ignore
        self.start_workers()

        for idx in self.indices:
            self.index_queue.put(idx)
            if len(self.batch_indices) < self.batch_size:
                self.batch_indices.append(idx)
            if len(self.batch_indices) == self.batch_size:
                yield self.collate_fn([self.data_queue.get() for _ in self.batch_indices])
                self.batch_indices = []

        # Add stop tokens to stop the worker threads
        for _ in range(self.num_workers):
            self.index_queue.put(self.stop_token)

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size
