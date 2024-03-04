from typing import List, Tuple, Union

import mlx.core as mx


def _default_collate_fn(batch: List[Tuple[mx.array, int]]) -> Union[Tuple[mx.array, mx.array], mx.array]:
    """Default collate function for the DataLoader.

    Args:
        batch: a single batch to be collated

    Returns:
        A tuple of input and target tensors
    """

    if isinstance(batch[0], tuple):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return mx.stack(inputs), mx.array(targets)
    elif isinstance(batch[0], mx.array):  # type: ignore
        return mx.stack(batch)
