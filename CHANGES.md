# Changes since v0.1.10

All changes relative to upstream `main` at commit `ffcfb57` (feat: mobilenetv2/3 release).

---

## New Model: DINOv3 Vision Transformer with RoPE Attention

Self-supervised DINOv3 vision transformer with Rotary Position Embeddings (RoPE),
ported from [etornam45's PR #22](https://github.com/riccardomusmeci/mlx-image/pull/22)
and modernized for Python 3.13+.

**3 pretrained variants** (weights from `mlx-vision` on HuggingFace Hub):

| Model | Params | Embed Dim | HuggingFace Repo |
|-------|--------|-----------|------------------|
| `vit_small_patch16_224.dinov3` | 21.6M | 384 | `mlx-vision/dinov3-vit-small-mlx` |
| `vit_small_plus_patch16_224.dinov3` | 28.7M | 384 | `mlx-vision/dinov3-vit-small-plus-mlx` |
| `vit_base_patch16_224.dinov3` | 85.7M | 768 | `mlx-vision/dinov3-vit-base-mlx` |

**New modules added:**

- `src/mlxim/model/dinov3/` — DINOv3 model, factory, FFN layers
- `src/mlxim/model/layers/attention.py` — `RoPESelfAttention`, `CausalSelfAttention`, `LinearKMaskedBias`
- `src/mlxim/model/layers/rope_position_encoding.py` — `RopePositionEmbedding` (axial RoPE, no learnable weights)
- `src/mlxim/model/layers/layer_scale.py` — `LayerScale` per-channel scaling
- `src/mlxim/model/layers/patch_embed.py` — `PatchEmbed` Conv2d-based patch embedding
- `src/mlxim/model/layers/rms_norm.py` — `RMSNorm`
- `src/mlxim/model/layers/utils.py` — `_randperm`, `cat_keep_shapes`, `uncat_with_shapes`, `named_apply`
- `src/mlxim/model/dinov3/ffn_layers.py` — `Mlp` and `SwiGLUFFN` with `ListForwardMixin`

**Key features:**

- `model.get_features(x)` returns CLS token embeddings
- `model.forward_features(x)` returns dict with `x_norm_clstoken`, `x_norm_patchtokens`, `x_storage_tokens`, `x_prenorm`, `masks`
- `model.get_intermediate_layers(x, n=K)` extracts features from last K transformer blocks
- `model(x, is_training=True)` returns full feature dict for self-supervised training
- Batch inference supported
- Same-class images produce higher cosine similarity than cross-class (verified on ImageNet)

---

## Build System: Poetry to uv + hatchling

- **Replaced Poetry** with [uv](https://github.com/astral-sh/uv) for dependency management and [hatchling](https://hatch.pypa.io/) as build backend
- `pyproject.toml` fully rewritten — no more `[tool.poetry]` sections
- `uv.lock` committed for reproducible installs
- `.python-version` file added (3.14)

---

## Python Version and Dependencies

**Minimum Python raised from 3.10 to 3.13.**

| Dependency | Old (v0.1.10) | New |
|-----------|---------------|-----|
| Python | ^3.10 | >=3.13 |
| mlx | 0.24.2 (pinned) | >=0.31.0 |
| numpy | 1.26.2 (pinned) | >=2.0 |
| albumentations | 1.4.1 (pinned) | >=2.0 |
| opencv-python | 4.9.0.80 (pinned) | >=4.10 |
| Pillow | 10.4.0 (pinned) | >=12.1.1 |
| pandas | 2.2.1 (pinned) | >=2.2 |
| huggingface_hub | 0.24.0 (pinned) | >=0.23 |
| safetensors | 0.4.2 (pinned) | >=0.4 |
| ruff | 0.0.264 | >=0.11 |

All exact version pins replaced with minimum version ranges.

---

## Typing Modernization

All 28 source files modernized to Python 3.13+ typing syntax:

- `List[X]` → `list[X]`
- `Dict[K, V]` → `dict[K, V]`
- `Tuple[X, Y]` → `tuple[X, Y]`
- `Optional[X]` → `X | None`
- `Union[X, Y]` → `X | Y`
- All `from typing import ...` imports for these removed
- All `# type: ignore` comments removed (zero remaining)

---

## Bug Fixes

### Model bugs

- **ViT `get_features()` return type** — was annotated as `mx.array`, actually returns `tuple[mx.array, list[mx.array]]`. Fixed annotation to match implementation.
- **ViT `EncoderBlock.__call__` and `Encoder.__call__`** — return type annotations were `mx.array` but the methods return tuples. Fixed.
- **ViT `.permute()` call** — MLX arrays don't have `.permute()`. Changed to `.transpose()`.
- **Swin Transformer `nn.init.normal` usage** — was calling `nn.init.normal(self.relative_position_bias_table, std=0.02)` passing the array as first arg. MLX `nn.init.normal` returns a callable initializer. Fixed to `nn.init.normal(std=0.02)(self.relative_position_bias_table)`.
- **`load_weights` type narrowing** — `mx.load()` returns a union type. Added `isinstance` assertion and typed dict comprehension to satisfy the type checker.
- **Conv2d weight transposition** — `load_weights` now transposes 4D weights from PyTorch OIHW layout to MLX OHWI layout when shapes are transposable.

### API bugs

- **`list_models()` returned `None`** — was printing to stdout and returning nothing. Now returns `list[str]` of model names.

### Training bugs

- **`Trainer` division by zero** — `self.log_every = int(log_every * len(self.train_loader))` produced 0 for small datasets, causing modulo-by-zero. Fixed with `max(1, ...)`.
- **`Trainer` device type** — `mx.set_default_device()` expects `mx.Device`, not `mx.DeviceType`. Added isinstance check.

### I/O bugs

- **`read_rgb` with cv2** — added `assert bgr is not None` after `cv2.imread()` for type safety.

---

## albumentations 2.0 Migration

All transform code updated for albumentations 2.0 breaking changes:

- `height=`/`width=` parameters → `size=(h, w)` tuple
- `value=` fill parameter → `fill=`
- `always_apply=True` → `p=1.0`
- `border_mode=cv2.BORDER_CONSTANT` → `border_mode=0`
- `A.Normalize(mean, std)` → positional args still work, verified

---

## Developer Tooling

### New tools added

- **ruff** (>=0.11) — linter and formatter, replaces black + old ruff. Config in `pyproject.toml` with `line-length = 120`, `target-version = "py313"`, pyupgrade rules enabled.
- **ty** (>=0.0.15) — Astral's type checker, configured in `pyproject.toml` under `[tool.ty]`. All source passes with 0 errors.
- **detect-secrets** — secret scanning with `.secrets.baseline`
- **pytest** (>=9.0) with `pytest-cov`, `pytest-sugar`, `pytest-xdist` — test runner with coverage

### Pre-commit hooks

`.pre-commit-config.yaml` updated with ruff lint + format hooks.

### Makefile

Updated targets for `uv run` commands instead of Poetry.

### Setup script

`scripts/setup-dev.sh` — sets up dev environment with uv.

---

## Test Suite

**241 tests total** (up from ~6 in v0.1.10), **88% code coverage**.

### New test files (24 files added)

| File | Tests | What it covers |
|------|-------|----------------|
| `test_arch_init.py` | 26 | Init + forward pass for all model families |
| `test_checkpoint.py` | 10 | ModelCheckpoint callbacks |
| `test_collate.py` | 2 | Batch collation |
| `test_config.py` | 3 | YAML config loading |
| `test_data_loader.py` | 4 | DataLoader batching, shuffle, drop_last |
| `test_dinov3.py` | 49 | DINOv3 init, pretrained inference, features, RoPE |
| `test_functional.py` | 9 | roll, normalize, dropout |
| `test_integration.py` | 38 | End-to-end: HF download → transform → inference → accuracy |
| `test_io.py` | 5 | read_rgb, save_image, resize |
| `test_layers_misc.py` | 6 | ConvNormActivation, SqueezeExcitation |
| `test_layers_ops.py` | 6 | StochasticDepth, make_divisible |
| `test_layers_utils.py` | 7 | ntuple helpers |
| `test_load_weights.py` | 8 | Weight loading, transposition, strict mode |
| `test_metrics.py` | 6 | Accuracy metric |
| `test_mlp.py` | 4 | MLP forward shapes |
| `test_model_factory.py` | 4 | create_model, list_models |
| `test_model_utils.py` | 9 | num_params, save/load weights |
| `test_pool_extended.py` | 3 | AdaptiveAvgPool2d edge cases |
| `test_time.py` | 1 | Timestamp formatting |
| `test_trainer.py` | 6 | Trainer init, train/val steps, full loop |
| `test_transform.py` | 6 | Train/eval transforms, crop modes |
| `test_validation.py` | 4 | Validation CSV tracking |
| `conftest.py` | — | Shared fixtures (test images, models) |

### Integration tests

- Download and run inference with real pretrained weights from HuggingFace
- End-to-end: JPEG → `read_rgb` → `ImageNetTransform` → model → prediction
- Models tested: ResNet18, ViT-B/16, EfficientNet-B0, MobileNet-V2, MobileNet-V3-S, Swin-Tiny, RegNet-Y-400MF, all 3 DINOv3 variants
- Weights roundtrip (save npz/safetensors → load → verify identical)
- Training loop convergence test

### DINOv3-specific tests

- All 3 variants: init, forward, get_features, forward_features, get_intermediate_layers
- Pretrained inference on 4 real ImageNet validation images
- Feature distinctness: different images produce different features
- Semantic similarity: same-class images (mud turtle) more similar than cross-class (tench vs cock)
- Deterministic inference: same input → same output
- Weight loading consistency: two loads produce identical features
- RoPE position embedding: shapes, sin/cos range, multi-resolution

---

## Model Count

**58 pretrained models** across 10 families:

| Family | Count | New in this release |
|--------|-------|---------------------|
| ResNet | 7 | — |
| ViT | 3 | — |
| ViT-DINO | 4 | — |
| ViT-DINOv2 | 3 | — |
| ViT-DINOv3 | 3 | Yes |
| ViT-SWAG | 6 | — |
| Swin / Swin V2 | 6 | — |
| EfficientNet | 8 | — |
| MobileNet V2/V3 | 3 | — |
| RegNet X/Y | 15 | — |

---

## Open PRs incorporated

This release covers changes from all open PRs against the upstream repo:

- **#13** — numpy version unpin (covered by full dependency upgrade to numpy >=2.0)
- **#22** — DINOv3 model (incorporated with full modernization)
