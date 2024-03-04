import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


def read_rgb(file_path: Union[str, Path], engine: str = "pil") -> np.array:  # type: ignore
    """Load an image from file_path as a numpy array.

    Args:
        file_path (Union[str, Path]): path to image
        engine (str, optional): image loading engine. Defaults to "pil".

    Raises:
        FileNotFoundError: if file is not found

    Returns:
        np.array: image
    """
    if engine.lower() not in ["pil", "cv2"]:
        print(f"[WARNING] Loading image engine {engine} is not supported. Using PIL instead.")
        engine = "pil"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")

    if engine == "pil":
        image = Image.open(file_path).convert("RGB")
        image = np.array(image)
    else:
        image = cv2.imread(file_path)  # type: ignore
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(image: np.array, output_path: Union[Path, str]) -> None:  # type: ignore
    """Save an image at given path making sure the folder exists.

    Args:
        image (np.array): image to save
        Union[Path, str] (str): output path
    """
    output_dir = output_path.replace(os.path.basename(output_path), "")  # type: ignore
    os.makedirs(output_dir, exist_ok=True)

    if len(image.shape) > 2:  # type: ignore
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        cv2.imwrite(output_path, image)  # type: ignore
    except Exception as e:
        print(f"[ERROR] While saving image at path {output_path} found an error - {e}")


def resize_rgb(image: np.array, w: int, h: int) -> np.array:  # type: ignore
    """Resize image to w x h.

    Args:
        image (np.array): image
        w (int): width
        h (int): height

    Returns:
        np.array: resized image
    """
    image = cv2.resize(image, (w, h))
    return image
