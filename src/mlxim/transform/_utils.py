from PIL import Image
from PIL.Image import Resampling

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

PIL_INTERP_TO_STR = {
    Resampling.NEAREST: "nearest",
    Resampling.BILINEAR: "bilinear",
    Resampling.BICUBIC: "bicubic",
    Resampling.BOX: "box",
    Resampling.HAMMING: "hamming",
    Resampling.LANCZOS: "lanczos",
}

STR_TO_PIL_INTERP = {v: k for k, v in PIL_INTERP_TO_STR.items()}
