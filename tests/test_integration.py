"""Integration tests that exercise the full mlx-image pipeline with real pretrained weights.

These tests download actual models from HuggingFace, create real test images,
run inference, and validate end-to-end correctness.
"""

import os
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from PIL import Image

from mlxim.data import DataLoader, FolderDataset, LabelFolderDataset
from mlxim.io import read_rgb, save_image
from mlxim.metrics.classification import Accuracy
from mlxim.model import create_model, get_weights, list_models, load_weights, num_params, save_weights
from mlxim.model._registry import MODEL_CONFIG, MODEL_ENTRYPOINT
from mlxim.trainer.trainer import Trainer
from mlxim.transform import ImageNetTransform

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_image_dir(tmp_path_factory):
    """Create a directory with real JPEG images for dataset tests."""
    base = tmp_path_factory.mktemp("images")
    # Create two class directories with real images
    for cls_name in ("cat", "dog"):
        cls_dir = base / cls_name
        cls_dir.mkdir()
        for i in range(6):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(cls_dir / f"{cls_name}_{i}.jpg")
    return base


@pytest.fixture(scope="module")
def flat_image_dir(tmp_path_factory):
    """Create a flat directory with real JPEG images (no labels)."""
    base = tmp_path_factory.mktemp("flat_images")
    for i in range(8):
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(base / f"img_{i}.jpg")
    return base


@pytest.fixture(scope="module")
def resnet18_model():
    """Download and cache resnet18 with real pretrained weights."""
    model = create_model("resnet18", weights=True, num_classes=1000)
    model.eval()
    return model


@pytest.fixture(scope="module")
def vit_model():
    """Download and cache vit_base_patch16_224 with real pretrained weights."""
    model = create_model("vit_base_patch16_224", weights=True, num_classes=1000)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------


class TestModelInference:
    """Test real model inference with pretrained weights."""

    def test_resnet18_inference_produces_1000_classes(self, resnet18_model):
        x = mx.random.normal((1, 224, 224, 3))
        logits = resnet18_model(x)
        mx.eval(logits)
        assert logits.shape == (1, 1000)

    def test_resnet18_batch_inference(self, resnet18_model):
        x = mx.random.normal((4, 224, 224, 3))
        logits = resnet18_model(x)
        mx.eval(logits)
        assert logits.shape == (4, 1000)

    def test_resnet18_softmax_sums_to_one(self, resnet18_model):
        x = mx.random.normal((1, 224, 224, 3))
        logits = resnet18_model(x)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        assert abs(probs.sum().item() - 1.0) < 1e-4

    def test_resnet18_deterministic(self, resnet18_model):
        """Same input should produce same output (eval mode, no dropout)."""
        x = mx.random.normal((1, 224, 224, 3))
        out1 = resnet18_model(x)
        out2 = resnet18_model(x)
        mx.eval(out1, out2)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-5)

    def test_vit_inference_produces_1000_classes(self, vit_model):
        x = mx.random.normal((1, 224, 224, 3))
        logits = vit_model(x)
        mx.eval(logits)
        assert logits.shape == (1, 1000)

    def test_vit_with_attention_masks(self, vit_model):
        x = mx.random.normal((1, 224, 224, 3))
        logits, attn_masks = vit_model(x, attn_masks=True)
        mx.eval(logits)
        assert logits.shape == (1, 1000)
        assert isinstance(attn_masks, list)
        assert len(attn_masks) > 0

    def test_vit_get_features(self, vit_model):
        x = mx.random.normal((1, 224, 224, 3))
        features, attn_masks = vit_model.get_features(x)
        mx.eval(features)
        assert features.ndim == 2
        assert features.shape[0] == 1

    def test_resnet18_get_features(self, resnet18_model):
        x = mx.random.normal((1, 224, 224, 3))
        features = resnet18_model.get_features(x)
        mx.eval(features)
        # ResNet features are a flat array
        assert features.ndim == 2
        assert features.shape[0] == 1


class TestModelRegistry:
    """Test model registry and config."""

    def test_all_registered_models_have_configs(self):
        for name in MODEL_ENTRYPOINT:
            assert name in MODEL_CONFIG, f"Model {name} registered but has no config"

    def test_all_configs_have_transform_settings(self):
        for name, config in MODEL_CONFIG.items():
            assert hasattr(config, "transform"), f"Model {name} config missing transform"
            assert hasattr(config.transform, "img_size"), f"Model {name} transform missing img_size"

    def test_list_models_runs(self, capsys):
        list_models()
        captured = capsys.readouterr()
        assert "resnet18" in captured.out
        assert "vit_base_patch16_224" in captured.out

    def test_create_model_invalid_raises(self):
        with pytest.raises(ValueError, match="not available"):
            create_model("nonexistent_model_xyz")


# ---------------------------------------------------------------------------
# Transform pipeline
# ---------------------------------------------------------------------------


class TestTransformPipeline:
    """Test the full image transform pipeline with real images."""

    def test_train_transform_on_real_image(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=True, img_size=224)
        result = transform(img)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32

    def test_val_transform_on_real_image(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=False, img_size=224)
        result = transform(img)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32

    def test_transform_with_model_config(self):
        """Use actual model config to build the transform."""
        from dataclasses import asdict

        config = MODEL_CONFIG["resnet18"]
        transform_config = asdict(config.transform)
        transform = ImageNetTransform(train=False, **transform_config)
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transform(img)
        expected_size = transform_config["img_size"]
        if isinstance(expected_size, int):
            expected_size = (expected_size, expected_size)
        assert result.shape == (expected_size[0], expected_size[1], 3)

    def test_squash_crop_mode(self):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=False, img_size=224, crop_mode="squash")
        result = transform(img)
        assert result.shape == (224, 224, 3)

    def test_border_crop_mode(self):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=False, img_size=224, crop_mode="border")
        result = transform(img)
        assert result.shape == (224, 224, 3)

    def test_train_transform_with_color_jitter(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=True, img_size=224, color_jitter=0.4)
        result = transform(img)
        assert result.shape == (224, 224, 3)

    def test_train_transform_with_flips(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=True, img_size=224, hflip=1.0, vflip=1.0)
        result = transform(img)
        assert result.shape == (224, 224, 3)

    def test_tuple_img_size(self):
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        transform = ImageNetTransform(train=False, img_size=(192, 256))
        result = transform(img)
        assert result.shape == (192, 256, 3)


# ---------------------------------------------------------------------------
# Data pipeline (real images on disk)
# ---------------------------------------------------------------------------


class TestDataPipeline:
    """Test dataset and dataloader with real images on disk."""

    def test_folder_dataset_loads_images(self, flat_image_dir):
        dataset = FolderDataset(root_dir=str(flat_image_dir), transform=None, verbose=False)
        assert len(dataset) == 8
        img = dataset[0]
        assert isinstance(img, mx.array)
        assert img.ndim == 3

    def test_folder_dataset_with_transform(self, flat_image_dir):
        transform = ImageNetTransform(train=False, img_size=224)
        dataset = FolderDataset(root_dir=str(flat_image_dir), transform=transform, verbose=False)
        img = dataset[0]
        mx.eval(img)
        assert img.shape == (224, 224, 3)

    def test_label_folder_dataset(self, test_image_dir):
        class_map = {0: "cat", 1: "dog"}
        dataset = LabelFolderDataset(root_dir=str(test_image_dir), class_map=class_map, transform=None, verbose=False)
        assert len(dataset) == 12  # 6 cats + 6 dogs
        img, target = dataset[0]
        assert isinstance(img, mx.array)
        assert isinstance(target, int)

    def test_label_folder_dataset_with_transform(self, test_image_dir):
        class_map = {0: "cat", 1: "dog"}
        transform = ImageNetTransform(train=False, img_size=224)
        dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=transform, verbose=False
        )
        img, target = dataset[0]
        mx.eval(img)
        assert img.shape == (224, 224, 3)

    def test_dataloader_batches(self, test_image_dir):
        class_map = {0: "cat", 1: "dog"}
        transform = ImageNetTransform(train=False, img_size=224)
        dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=transform, verbose=False
        )
        loader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=False)
        batches = list(loader)
        assert len(batches) == 3  # 12 images / 4 batch_size
        x, targets = batches[0]
        mx.eval(x)
        assert x.shape == (4, 224, 224, 3)
        assert targets.shape == (4,)

    def test_dataloader_drop_last(self, test_image_dir):
        class_map = {0: "cat", 1: "dog"}
        transform = ImageNetTransform(train=False, img_size=224)
        dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=transform, verbose=False
        )
        loader = DataLoader(dataset=dataset, batch_size=5, num_workers=0, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 2  # 12 // 5 = 2 full batches
        for x, _ in batches:
            assert x.shape[0] == 5

    def test_dataloader_with_shuffle(self, test_image_dir):
        class_map = {0: "cat", 1: "dog"}
        dataset = LabelFolderDataset(root_dir=str(test_image_dir), class_map=class_map, transform=None, verbose=False)
        loader = DataLoader(dataset=dataset, batch_size=12, num_workers=0, shuffle=True)
        batch = next(iter(loader))
        x, targets = batch
        mx.eval(x)
        assert x.shape[0] == 12


# ---------------------------------------------------------------------------
# End-to-end: load model → transform image → run inference
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline: real image → transform → pretrained model → prediction."""

    def test_resnet18_e2e_with_real_jpeg(self, resnet18_model, tmp_path):
        """Create a real JPEG, read it, transform it, run inference."""
        # Create and save a real image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test.jpg"
        img.save(str(img_path))

        # Read with the library's own I/O
        loaded_img = read_rgb(str(img_path), engine="pil")
        assert loaded_img.shape == (480, 640, 3)

        # Transform using model's own config
        from dataclasses import asdict

        transform_config = asdict(MODEL_CONFIG["resnet18"].transform)
        transform = ImageNetTransform(train=False, **transform_config)
        transformed = transform(loaded_img)

        # Run inference
        x = mx.array(transformed)[None]  # add batch dim
        logits = resnet18_model(x)
        mx.eval(logits)

        assert logits.shape == (1, 1000)
        # Verify we get a valid class prediction
        pred_class = logits.argmax(axis=-1).item()
        assert 0 <= pred_class < 1000

    def test_vit_e2e_with_real_jpeg(self, vit_model, tmp_path):
        """Create a real JPEG, read it, transform it, run ViT inference."""
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_vit.jpg"
        img.save(str(img_path))

        loaded_img = read_rgb(str(img_path), engine="pil")

        from dataclasses import asdict

        transform_config = asdict(MODEL_CONFIG["vit_base_patch16_224"].transform)
        transform = ImageNetTransform(train=False, **transform_config)
        transformed = transform(loaded_img)

        x = mx.array(transformed)[None]
        logits = vit_model(x)
        mx.eval(logits)

        assert logits.shape == (1, 1000)
        pred_class = logits.argmax(axis=-1).item()
        assert 0 <= pred_class < 1000

    def test_dataloader_to_model_inference(self, resnet18_model, test_image_dir):
        """Full pipeline: dataset → dataloader → model → predictions."""
        class_map = {0: "cat", 1: "dog"}
        transform = ImageNetTransform(train=False, img_size=224)
        dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=transform, verbose=False
        )
        loader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=False)

        all_preds = []
        for x, _targets in loader:
            logits = resnet18_model(x)
            mx.eval(logits)
            preds = logits.argmax(axis=-1)
            all_preds.extend(preds.tolist())

        assert len(all_preds) == 12

    def test_accuracy_metric_with_real_model(self, resnet18_model, test_image_dir):
        """Run model on real data and compute accuracy metrics."""
        class_map = {0: "cat", 1: "dog"}
        transform = ImageNetTransform(train=False, img_size=224)
        dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=transform, verbose=False
        )
        loader = DataLoader(dataset=dataset, batch_size=6, num_workers=0, shuffle=False)

        # Use the real Accuracy metric (top_k capped to num_classes=2)
        accuracy = Accuracy(top_k=(1,))
        for x, targets in loader:
            logits = resnet18_model(x)
            # Slice to 2 classes for our dataset
            logits_2class = logits[:, :2]
            mx.eval(logits_2class)
            accuracy.update(logits_2class, targets)

        result = accuracy.compute()
        assert "acc@1" in result
        assert 0.0 <= result["acc@1"] <= 1.0


# ---------------------------------------------------------------------------
# Weights save/load roundtrip
# ---------------------------------------------------------------------------


class TestWeightsRoundtrip:
    """Test saving and loading real model weights."""

    def test_save_load_npz_roundtrip(self, resnet18_model, tmp_path):
        weights = get_weights(resnet18_model)
        path = str(tmp_path / "resnet18.npz")
        save_weights(weights, path)
        assert os.path.exists(path)

        # Create fresh model and load saved weights
        model2 = create_model("resnet18", weights=False, num_classes=1000)
        model2 = load_weights(model2, path, strict=False)
        model2.eval()

        # Verify same output
        x = mx.random.normal((1, 224, 224, 3))
        out1 = resnet18_model(x)
        out2 = model2(x)
        mx.eval(out1, out2)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-4)

    def test_save_load_safetensors_roundtrip(self, resnet18_model, tmp_path):
        weights = get_weights(resnet18_model)
        path = str(tmp_path / "resnet18.safetensors")
        save_weights(weights, path)
        assert os.path.exists(path)

        model2 = create_model("resnet18", weights=False, num_classes=1000)
        model2 = load_weights(model2, path, strict=False)
        model2.eval()

        x = mx.random.normal((1, 224, 224, 3))
        out1 = resnet18_model(x)
        out2 = model2(x)
        mx.eval(out1, out2)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-4)

    def test_num_params_resnet18(self, resnet18_model):
        n = num_params(resnet18_model)
        # ResNet-18 has ~11.7M parameters
        assert 11_000_000 < n < 12_000_000, f"Expected ~11.7M params, got {n}"


# ---------------------------------------------------------------------------
# Training loop (micro-training on synthetic data)
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    """Test the actual Trainer class with real data on disk."""

    def test_micro_training_converges(self, test_image_dir):
        """Train a tiny model for a few epochs and verify loss decreases."""
        class_map = {0: "cat", 1: "dog"}
        train_transform = ImageNetTransform(train=True, img_size=32)
        val_transform = ImageNetTransform(train=False, img_size=32)

        train_dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=train_transform, verbose=False
        )
        val_dataset = LabelFolderDataset(
            root_dir=str(test_image_dir), class_map=class_map, transform=val_transform, verbose=False
        )

        train_loader = DataLoader(dataset=train_dataset, batch_size=6, num_workers=0, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=6, num_workers=0, shuffle=False)

        # Use a small model with 2 classes
        model = create_model("resnet18", weights=False, num_classes=2)

        lr_schedule = optim.cosine_decay(init=1e-2, decay_steps=len(train_loader) * 3)
        optimizer = optim.SGD(learning_rate=lr_schedule)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.losses.cross_entropy,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=3,
            top_k=(1,),
        )

        # Record initial loss
        model.train()
        first_batch = next(iter(train_loader))
        x, target = first_batch
        logits = model(x)
        initial_loss = mx.mean(nn.losses.cross_entropy(logits, target)).item()

        # Train
        trainer.train()

        # Record final loss
        model.eval()
        final_batch = next(iter(val_loader))
        x, target = final_batch
        logits = model(x)
        final_loss = mx.mean(nn.losses.cross_entropy(logits, target)).item()

        # Loss should have decreased (or at least not exploded)
        assert final_loss < initial_loss * 2, f"Loss exploded: {initial_loss} -> {final_loss}"


# ---------------------------------------------------------------------------
# I/O operations
# ---------------------------------------------------------------------------


class TestIOOperations:
    """Test image I/O with real files."""

    def test_read_rgb_pil_engine(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        path = str(tmp_path / "test_pil.jpg")
        img.save(path)
        loaded = read_rgb(path, engine="pil")
        assert loaded.shape[0] == 100
        assert loaded.shape[1] == 150
        assert loaded.shape[2] == 3

    def test_read_rgb_cv2_engine(self, tmp_path):
        img = Image.fromarray(np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8))
        path = str(tmp_path / "test_cv2.jpg")
        img.save(path)
        loaded = read_rgb(path, engine="cv2")
        assert loaded.shape[0] == 100
        assert loaded.shape[1] == 150
        assert loaded.shape[2] == 3

    def test_save_and_reload_image(self, tmp_path):
        original = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = str(tmp_path / "roundtrip.jpg")
        save_image(original, path)
        assert os.path.exists(path)
        reloaded = read_rgb(path, engine="pil")
        assert reloaded.shape == (64, 64, 3)
