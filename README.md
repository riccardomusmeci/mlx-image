# mlx-image

Apple MLX image models library -- converts and runs timm/torchvision vision models natively on Apple Silicon.

| | |
|---|---|
| **Stack** | Python 3.13+, Apple MLX, safetensors |
| **Runs on** | Apple Silicon (M1/M2/M3/M4) |
| **Weights** | [mlx-vision](https://huggingface.co/mlx-vision) on HuggingFace |

## What mlx-image Provides

When a developer needs to run image classification, feature extraction, or fine-tuning on Apple Silicon, they import `mlxim` and call `create_model("resnet18")` to get a ready-to-use model with pre-trained weights downloaded from HuggingFace. The library handles weight conversion from PyTorch `.pth` format to MLX-native `.safetensors`, so no manual conversion step is needed. Seven model families are supported -- ResNet, ViT (including DINO v1/v2/v3), Swin Transformer, RegNet, EfficientNet, MobileNet, and DINOv3 -- covering a wide range of classification and representation learning architectures.

Beyond inference, the library provides a PyTorch-familiar training pipeline. `DataLoader` supports custom `collate_fn` and multi-worker data loading, `LabelFolderDataset` mirrors PyTorch's `ImageFolder`, and `ModelCheckpoint` tracks the best model with early stopping suggestions. A developer who has trained models in PyTorch can move to MLX with minimal code changes: the optimizer, loss, and gradient APIs follow the same patterns.

The weights are not retrained -- they are direct conversions from the original PyTorch implementations, validated against ImageNet-1K to confirm that accuracy is preserved. Each model is published to the `mlx-vision` HuggingFace community so that `create_model()` can fetch weights automatically. The library also supports loading weights from local `.safetensors` files for offline or custom-trained models.

Internally, the model registry maps model names to factory functions and configuration dicts. Each architecture family (resnet, vit, swin, etc.) has its own submodule with a `_factory.py` that registers supported variants. A shared `layers` module provides common building blocks -- attention, MLP, patch embedding, pooling, RoPE position encoding, RMS normalization, and weight initialization -- so that new architectures can be added by composing existing layers rather than reimplementing them.

## Features

- **7 model families** -- ResNet, ViT, Swin Transformer, RegNet, EfficientNet, MobileNet, and DINOv3 cover classification, self-supervised representation learning, and efficient mobile inference
- **Automatic weight download** -- `create_model("resnet18")` fetches pre-trained weights from HuggingFace so models are ready to use with zero manual setup
- **PyTorch-compatible training API** -- DataLoader, Dataset, and ModelCheckpoint mirror PyTorch conventions so developers can migrate training code with minimal changes
- **Multi-worker data loading** -- DataLoader supports `num_workers` and custom `collate_fn` for efficient batch preparation on Apple Silicon
- **ImageNet-validated conversions** -- every converted model is validated against ImageNet-1K to ensure accuracy matches the original PyTorch implementation
- **Shared layer library** -- attention, MLP, patch embedding, pooling, RoPE, RMS norm, and weight init modules let new architectures compose from proven building blocks
- **Local and remote weights** -- load from HuggingFace automatically or from a local `.safetensors` file for offline and custom-trained models
- **Classification metrics** -- built-in top-k accuracy and per-class metrics for evaluation without external dependencies
- **Model checkpoint callbacks** -- track best model during training with configurable metric monitoring and early stopping suggestions

## Quick Start

```bash
uv sync
uv run pytest tests/ -v --cov=src/mlxim
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## Models

Model weights are available on the [`mlx-vision`](https://huggingface.co/mlx-vision) community on HuggingFace.

To load a model with pre-trained weights:
```python
from mlxim.model import create_model

# loading weights from HuggingFace (https://huggingface.co/mlx-vision/resnet18-mlxim)
model = create_model("resnet18") # pretrained weights loaded from HF

# loading weights from local file
model = create_model("resnet18", weights="path/to/resnet18/model.safetensors")
```

To list all available models:

```python
from mlxim.model import list_models
list_models()
```

### Supported models

List of all models available in `mlx-image`:

* **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
* **ViT**:
    * **supervised**: vit_base_patch16_224, vit_base_patch16_224.swag_lin, vit_base_patch16_384.swag_e2e, vit_base_patch32_224, vit_large_patch16_224, vit_large_patch16_224, vit_large_patch16_224.swag_lin, vit_large_patch16_512.swag_e2e, vit_huge_patch14_224.swag_lin, vit_huge_patch14_518.swag_e2e
    * **DINO v1**: vit_base_patch16_224.dino, vit_small_patch16_224.dino, vit_small_patch8_224.dino, vit_base_patch8_224.dino

    * **DINO v2**: vit_small_patch14_518.dinov2, vit_base_patch14_518.dinov2, vit_large_patch14_518.dinov2
* **Swin**: swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_v2_tiny_patch4_window8_256, swin_v2_small_patch4_window8_256, swin_v2_base_patch4_window8_256
* **RegNet**: regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, regnet_x_8gf, regnet_x_16gf, regnet_x_32gf, regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, regnet_y_128gf
* **EfficientNet**: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
* **MobileNet**: mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

> **Warning**: The `regnet_y_128gf` model couldn't be tested due to computational limitations.

## ImageNet-1K Results

Go to [results-imagenet-1k.csv](https://github.com/riccardomusmeci/mlx-image/blob/main/results/results-imagenet-1k.csv) to check every model converted to `mlx-image` and its performance on ImageNet-1K with different settings.

> **TL;DR** performance is comparable to the original models from PyTorch implementations.

## Similarity to PyTorch and other familiar tools

`mlx-image` tries to be as close as possible to PyTorch:
- `DataLoader` -> you can define your own `collate_fn` and also use `num_workers` to speed up data loading
- `Dataset` -> `mlx-image` already supports `LabelFolderDataset` (the good and old PyTorch `ImageFolder`) and `FolderDataset` (a generic folder with images in it)
- `ModelCheckpoint` -> keeps track of the best model and saves it to disk (similar to PyTorchLightning). It also suggests early stopping

## Training

Training is similar to PyTorch. Here's an example of how to train a model:

```python
import mlx.nn as nn
import mlx.optimizers as optim
from mlxim.model import create_model
from mlxim.data import LabelFolderDataset, DataLoader

train_dataset = LabelFolderDataset(
    root_dir="path/to/train",
    class_map={0: "class_0", 1: "class_1", 2: ["class_2", "class_3"]}
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
model = create_model("resnet18") # pretrained weights loaded from HF
optimizer = optim.Adam(learning_rate=1e-3)

def train_step(model, inputs, targets):
    logits = model(inputs)
    loss = mx.mean(nn.losses.cross_entropy(logits, target))
    return loss

model.train()
for epoch in range(10):
    for batch in train_loader:
        x, target = batch
        train_step_fn = nn.value_and_grad(model, train_step)
        loss, grads = train_step_fn(x, target)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
```

## Validation

The `validation.py` script is run every time a pth model is converted to mlx and it's used to check if the model performs similarly to the original one on ImageNet-1K.

I use the configuration file `config/validation.yaml` to set the parameters for the validation script.

You can download the ImageNet-1K validation set from mlx-vision space on HuggingFace at this [link](https://huggingface.co/datasets/mlx-vision/imagenet-1k).

## Architecture

```
src/mlxim/
├── __init__.py                              # Package version
├── callbacks/
│   └── checkpoint.py                        # ModelCheckpoint with best-model tracking and early stopping
├── data/
│   ├── _base.py                             # Base dataset class
│   ├── _utils.py                            # Data utilities (collation, batching)
│   ├── folder.py                            # LabelFolderDataset and FolderDataset
│   └── loader.py                            # DataLoader with multi-worker support
├── io/
│   ├── config.py                            # YAML config loading for train/validation
│   └── image.py                             # Image read/write helpers
├── metrics/
│   └── classification.py                    # Top-k accuracy and per-class metrics
├── model/
│   ├── _config.py                           # Model configuration dataclass
│   ├── _factory.py                          # create_model() and list_models() entry points
│   ├── _registry.py                         # Global model entrypoint and config registry
│   ├── _utils.py                            # Weight loading and conversion utilities
│   ├── layers/                              # 13 shared layer modules
│   │   ├── _ops.py                          # Custom operations
│   │   ├── attention.py                     # Multi-head attention
│   │   ├── functional.py                    # Activation functions
│   │   ├── layer_scale.py                   # Layer scale (CaiT/DeiT)
│   │   ├── misc.py                          # Miscellaneous layers
│   │   ├── mlp.py                           # MLP blocks
│   │   ├── patch_embed.py                   # Patch embedding for ViT/Swin
│   │   ├── pool.py                          # Pooling layers
│   │   ├── rms_norm.py                      # RMS normalization
│   │   ├── rope_position_encoding.py        # Rotary position encoding
│   │   ├── utils.py                         # Layer utilities
│   │   └── weight_init.py                   # Weight initialization
│   ├── resnet/                              # ResNet family (7 variants)
│   ├── vit/                                 # Vision Transformer family
│   ├── swin_transformer/                    # Swin Transformer family (6 variants)
│   ├── regnet/                              # RegNet family (15 variants)
│   ├── efficientnet/                        # EfficientNet family (8 variants)
│   ├── mobilenet/                           # MobileNet v2/v3 (3 variants)
│   └── dinov3/                              # DINOv3 with RoPE and SwiGLU FFN
├── trainer/
│   └── trainer.py                           # Training loop orchestration
├── transform/
│   ├── _utils.py                            # Transform utilities
│   └── imagenet.py                          # ImageNet normalization and augmentation
└── utils/
    ├── imagenet.py                          # ImageNet class labels and mappings
    ├── time.py                              # Timing utilities
    └── validation.py                        # Validation loop helpers
```

## Modules

| Module | What It Does |
|--------|-------------|
| `model` | Registry-based model factory -- `create_model()` instantiates any supported architecture by name and loads weights from HuggingFace or local files |
| `model.layers` | 13 shared building blocks (attention, MLP, patch embed, pooling, RoPE, RMS norm, weight init) composed by all architecture families |
| `data` | PyTorch-style DataLoader and Dataset classes with multi-worker loading, custom collation, and folder-based datasets |
| `callbacks` | ModelCheckpoint callback for tracking best model during training with early stopping suggestions |
| `metrics` | Classification metrics including top-k accuracy and per-class evaluation |
| `trainer` | Training loop orchestration tying together model, optimizer, data loader, and callbacks |
| `transform` | ImageNet normalization and augmentation pipelines for preprocessing |
| `io` | YAML config loading and image read/write helpers |
| `utils` | ImageNet class labels, timing utilities, and validation loop helpers |

## Testing

```bash
uv run pytest tests/ -v --cov=src/mlxim          # All tests (27 test files)
uv run ruff check src/ tests/                     # Lint
uv run ruff format --check src/ tests/            # Format check
uv run ty check src/                              # Type check
```

Tests require Apple Silicon (MLX framework). Local CI skips tests on non-Apple hardware (`[tool.local-ci] skip_tests = true`).

## Dependencies

- **mlx** (>=0.31.0) -- Apple's ML framework for array operations, autograd, and neural network primitives
- **safetensors** (>=0.4) -- fast and safe tensor serialization for model weights
- **huggingface_hub** (>=0.23) -- downloads pre-trained weights from the mlx-vision HuggingFace community
- **numpy** (>=2.0) -- array operations and data manipulation
- **Pillow** (>=12.1.1) -- image loading and basic transforms
- **opencv-python** (>=4.10) -- image preprocessing and augmentation
- **albumentations** (>=2.0) -- advanced image augmentation pipelines
- **pandas** (>=2.2) -- results tracking and ImageNet label mapping
- **matplotlib** (>=3.9) -- visualization of training metrics and predictions
- **tqdm** (>=4.66) -- progress bars for training and validation loops

## Contributing

This is a work in progress, so any help is appreciated.

I am working on it in my spare time, so I can't guarantee frequent updates.

If you love coding and want to contribute, follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## Additional Resources

* [mlx-vision community](https://huggingface.co/mlx-vision)
* [HuggingFace doc](https://huggingface.co/docs/hub/main/en/mlx-image)

## Known Issues

- **No CI pipeline** -- no `.github/workflows` directory exists; tests run manually on Apple Silicon machines only
- **DenseNet not yet implemented** -- listed as a to-do but not started
- **regnet_y_128gf untested** -- model is too large to validate on available hardware
