# **mlx-image**
Image models based on [Apple MLX framework](https://github.com/ml-explore/mlx) for Apple Silicon machines.

## **Why?**

Apple MLX framework is a great tool to run machine learning models on Apple Silicon machines.

This repository is meant to convert image models from timm/torchvision to Apple MLX framework. The weights are just converted from .pth to .npz/.safetensors and the models **are not trained again**.

## How to install

```bash
pip install mlx-image
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

## **Validation**

The `validation.py` script is run every time a pth model is converted to mlx and it's used to check if the model performs similarly to the original one on ImageNet-1K.

I use the configuration file `config/validation.yaml` to set the parameters for the validation script.

You can download the ImageNet-1K validation set from mlx-vision space on HuggingFace at this [link](https://huggingface.co/datasets/mlx-vision/imagenet-1k).

## **Contributing**

This is a work in progress, so any help is appreciated.

I am working on it in my spare time, so I can't guarantee frequent updates.

If you love coding and want to contribute, follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## Additional Resources

* [mlx-vision community](https://huggingface.co/mlx-vision)
* [HuggingFace doc](https://huggingface.co/docs/hub/main/en/mlx-image)

## **To-Dos**

[ ] inference script (similar to train/validation)

[ ] DenseNet

[ ] MobileNet

## Contact

If you have any questions, please email `riccardomusmeci92@gmail.com`.