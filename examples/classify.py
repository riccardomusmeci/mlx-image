"""Classify an image using a pretrained model.

Usage:
    python examples/classify.py --image path/to/image.jpg
    python examples/classify.py --image path/to/image.jpg --model resnet50
"""

import argparse

import mlx.core as mx
import numpy as np
from PIL import Image

from mlxim.model import create_model, list_models
from mlxim.model._registry import MODEL_CONFIG
from mlxim.transform import ImageNetTransform
from mlxim.utils.imagenet import IMAGENET2012_CLASSES


def classify(image_path: str, model_name: str = "resnet18", top_k: int = 5) -> None:
    if model_name not in MODEL_CONFIG:
        print(f"Model '{model_name}' not found. Available models:")
        for name in sorted(list_models()):
            print(f"  {name}")
        return

    config = MODEL_CONFIG[model_name]
    transform = ImageNetTransform(
        train=False,
        img_size=config.transform.img_size,
        crop_pct=config.transform.crop_pct,
        mean=config.transform.mean,
        std=config.transform.std,
        interpolation=config.transform.interpolation,
    )

    print(f"Loading {model_name}...")
    model = create_model(model_name, weights=True)
    model.eval()

    image = np.array(Image.open(image_path).convert("RGB"))
    image = transform(image)
    # HWC -> 1CHW -> 1HWC (mlx uses channels-last but transform outputs HWC already)
    x = mx.array(image)[None]

    logits = model(x)
    probs = mx.softmax(logits, axis=1)

    top_indices = mx.argsort(probs, axis=1)[0, -top_k:][::-1]
    top_probs = probs[0]

    labels = list(IMAGENET2012_CLASSES.values())

    print(f"\nTop-{top_k} predictions:")
    for idx in np.array(top_indices).tolist():
        print(f"  {labels[idx]:40s} {top_probs[idx].item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image with a pretrained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name (default: resnet18)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions (default: 5)")
    args = parser.parse_args()

    classify(args.image, args.model, args.top_k)
