import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_rgb(file_path: str, engine: str = "pil") -> np.ndarray:
    """Load an image from file_path as a numpy array.

    Args:
        file_path (str): path to image
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
        return np.array(image)
    else:
        bgr = cv2.imread(file_path)
        assert bgr is not None, f"cv2.imread failed for {file_path}"
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, output_path: Path | str) -> None:
    """Save an image at given path making sure the folder exists.

    Args:
        image (np.array): image to save
        output_path: Union[Path, str] (str): output path
    """
    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        cv2.imwrite(str(output_path), image)
    except Exception as e:
        print(f"[ERROR] While saving image at path {output_path} found an error - {e}")


def resize_rgb(image: np.ndarray, w: int, h: int) -> np.ndarray:
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
