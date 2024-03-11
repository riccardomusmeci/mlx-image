# **mlxim**
Image models based on [Apple MLX framework](https://github.com/ml-explore/mlx) for Apple Silicon machines.

## **Why? 💡**

Apple MLX framework is a great tool to run machine learning models on Apple Silicon machines.

This repository is meant to convert image models from timm/torchvision to Apple MLX framework. The weights are just converted from .pth to .npz/.safetensors and the models **are not trained again**.

I don't have enough compute power (and time) to train all the models from scratch (**someone buy me a maxed-out Mac, please**).

## **How to install**

```
pip install mlx-image
```

## **Models**

Model weights are available on the [`mlx-vision`](https://huggingface.co/mlx-vision) community on HuggingFace.

To load a model with pre-trained weights:
```python
from mlxim.model import create_model

# loading weights from HuggingFace (https://huggingface.co/mlx-vision/resnet18-mlxim)
model = create_model("resnet18") # pretrained weights loaded from HF

# loading weights from another HuggingFace model
model = create_model("resnet18", weights="hf://repo_id/filename")

# loading weights from local file
model = create_model("resnet18", weights="path/to/resnet18/model.safetensors")
```

To list all available models:

```python
from mlxim.model import list_models
list_models()
```
> [!WARNING]
> As of today (2024-03-08) mlx does not support `group` param for nn.Conv2d. Therefore, architectures such as `resnext`, `regnet` or `efficientnet` are not yet supported in `mlxim`.

## **ImageNet-1K Results**

Go to [results-imagenet-1k.csv](results/results-imagenet-1k.csv) to check every model converted and its performance on ImageNet-1K with different settings.

> **TL;DR** performance is comparable to the original models from PyTorch implementations.

## **Training**

Training is similar to PyTorch, thanks to some tools `mlx-im` provides. Here's an example of how to train a model with `mlx-im`:

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

## **Similarity to PyTorch and other familiar tools**

`mlxim` tries to be as close as possible to PyTorch:
- `DataLoader` -> you can define your own `collate_fn` and also use `num_workers` to speed up data loading
- `Dataset` -> `mlxim` already supports `LabelFolderDataset` (the good and old PyTorch `ImageFolder`) and `FolderDataset` (a generic folder with images in it)
- `ModelCheckpoint` -> keeps track of the best model and saves it to disk (similar to PyTorchLightning). It also suggests early stopping

## **Contributing**

This is a work in progress, so any help is appreciated.

I am working on it in my spare time, so I can't guarantee frequent updates.

If you love coding and want to contribute, follow the instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## **To-Do**

[ ] add register_model (similar to timm)

[ ] inference script (similar to train/validation)

[ ] SwinTransformer

[ ] DenseNet

[ ] MobileNet (waiting for nn.Conv2d group)

[ ] RegNet (waiting for nn.Conv2d group)


## Contact

If you have any questions, please email `riccardomusmeci92@gmail.com`.
