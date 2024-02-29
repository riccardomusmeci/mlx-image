from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

PIL_INTERP_TO_STR = {
    Image.NEAREST: "nearest",
    Image.BILINEAR: "bilinear",
    Image.BICUBIC: "bicubic",
    Image.BOX: "box",
    Image.HAMMING: "hamming",
    Image.LANCZOS: "lanczos",
}

STR_TO_PIL_INTERP = {v: k for k, v in PIL_INTERP_TO_STR.items()}
