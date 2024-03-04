from typing import Any

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


class Dataset:
    r"""An abstract class representing a :class:`Dataset`."""

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
