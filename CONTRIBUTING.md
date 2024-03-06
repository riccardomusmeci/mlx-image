# How to contribute

I want to make contributing to this project as easy as possible.

## Pull Requests

1. Fork and submit pull requests to the repo.
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests and at least one review.
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   Install the dev dependencies in pyproject.toml using `pip install .`.
   Before committing, always run `make format` to ensure the code is formatted correctly.

## Adding models to mlx-vision

1. Add the architecture to `src/mlxim/model/` like `ViT` and `ResNet`. Using the `_factory.py` is mandatory: keep track of model entry points and configs.
2. Test the model on ImageNet-1K using the `validation.py` script. Download the dataset from mlx-vision space on HuggingFace at this [link](https://huggingface.co/datasets/mlx-vision/imagenet-1k).
3. Create the model card and upload it to HuggingFace.
4. PR the model and the results under `results/results-imagenet-1k.csv` to the repo.

## Issues

Let's use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to mlx-image, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
