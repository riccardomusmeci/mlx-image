from typing import Any


class Dataset:
    r"""An abstract class representing a :class:`Dataset`."""

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
