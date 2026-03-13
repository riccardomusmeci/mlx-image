"""DINOv3 model tests — init, pretrained inference, and feature extraction."""

import mlx.core as mx
import numpy as np
import pytest

from mlxim.model import create_model, list_models
from mlxim.model._utils import num_params
from mlxim.model.layers import RopePositionEmbedding

DINOV3_MODELS = [
    "vit_small_patch16_224.dinov3",
    "vit_small_plus_patch16_224.dinov3",
    "vit_base_patch16_224.dinov3",
]

EXPECTED_DIMS = {
    "vit_small_patch16_224.dinov3": 384,
    "vit_small_plus_patch16_224.dinov3": 384,
    "vit_base_patch16_224.dinov3": 768,
}

EXPECTED_PARAMS = {
    "vit_small_patch16_224.dinov3": 21_614_992,
    "vit_small_plus_patch16_224.dinov3": 28_711_312,
    "vit_base_patch16_224.dinov3": 85_697_296,
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestDinov3Registration:
    def test_dinov3_models_in_registry(self):
        models = list_models()
        for name in DINOV3_MODELS:
            assert name in models, f"{name} not in model registry"

    def test_total_model_count_includes_dinov3(self):
        models = list_models()
        assert len(models) >= 58


# ---------------------------------------------------------------------------
# Init (no pretrained weights)
# ---------------------------------------------------------------------------
class TestDinov3Init:
    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_init_no_weights(self, model_name):
        model = create_model(model_name, weights=False)
        assert model is not None

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_forward_random_input(self, model_name):
        model = create_model(model_name, weights=False)
        x = mx.random.normal((1, 224, 224, 3))
        out = model(x)
        dim = EXPECTED_DIMS[model_name]
        assert out.shape == (1, dim), f"Expected (1, {dim}), got {out.shape}"

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_get_features_random_input(self, model_name):
        model = create_model(model_name, weights=False)
        x = mx.random.normal((2, 224, 224, 3))
        features = model.get_features(x)
        dim = EXPECTED_DIMS[model_name]
        assert features.shape == (2, dim)

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_forward_features_returns_dict(self, model_name):
        model = create_model(model_name, weights=False)
        x = mx.random.normal((1, 224, 224, 3))
        ret = model.forward_features(x)
        assert isinstance(ret, dict)
        expected_keys = {
            "x_norm_clstoken",
            "x_storage_tokens",
            "x_norm_patchtokens",
            "x_prenorm",
            "masks",
        }
        assert set(ret.keys()) == expected_keys

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_patch_token_count(self, model_name):
        """224/16 = 14 patches per side → 196 patch tokens."""
        model = create_model(model_name, weights=False)
        x = mx.random.normal((1, 224, 224, 3))
        ret = model.forward_features(x)
        assert ret["x_norm_patchtokens"].shape[1] == 196

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_training_mode_forward(self, model_name):
        model = create_model(model_name, weights=False)
        model.train()
        x = mx.random.normal((2, 224, 224, 3))
        ret = model(x, is_training=True)
        assert isinstance(ret, dict)


# ---------------------------------------------------------------------------
# Pretrained inference on real ImageNet images
# ---------------------------------------------------------------------------
REAL_IMAGES = [
    ("tests/data/n01440764/ILSVRC2012_val_00000293.JPEG", "tench"),
    ("tests/data/n01514668/ILSVRC2012_val_00001114.JPEG", "cock"),
    ("tests/data/n01667778/ILSVRC2012_val_00002832.JPEG", "mud_turtle_1"),
    ("tests/data/n01667778/ILSVRC2012_val_00005799.JPEG", "mud_turtle_2"),
]


@pytest.fixture(scope="module")
def real_images():
    from mlxim.io import read_rgb
    from mlxim.transform import ImageNetTransform

    transform = ImageNetTransform(train=False, img_size=224)
    imgs = {}
    for path, label in REAL_IMAGES:
        img = read_rgb(path)
        imgs[label] = mx.array(np.expand_dims(transform(img), 0))
    return imgs


@pytest.fixture(scope="module")
def real_batch(real_images):
    return mx.concatenate(list(real_images.values()), axis=0)


class TestDinov3Pretrained:
    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_param_count(self, model_name):
        model = create_model(model_name)
        n = num_params(model)
        assert n == EXPECTED_PARAMS[model_name], f"Expected {EXPECTED_PARAMS[model_name]:,} params, got {n:,}"

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_single_image_inference(self, model_name, real_images):
        model = create_model(model_name)
        dim = EXPECTED_DIMS[model_name]
        for label, x in real_images.items():
            out = model(x)
            assert out.shape == (1, dim), f"{label}: expected (1,{dim}), got {out.shape}"
            norm = mx.sqrt(mx.sum(out * out)).item()
            assert norm > 0, f"{label}: feature norm is 0"

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_batch_inference(self, model_name, real_batch):
        model = create_model(model_name)
        dim = EXPECTED_DIMS[model_name]
        out = model(real_batch)
        assert out.shape == (4, dim)

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_features_differ_across_images(self, model_name, real_batch):
        model = create_model(model_name)
        features = np.array(model.get_features(real_batch))
        # All 4 images should produce different features
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.allclose(features[i], features[j], atol=1e-4), (
                    f"Features for image {i} and {j} are too similar"
                )

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_same_class_more_similar(self, model_name, real_batch):
        """Two mud turtle images should be more similar than fish vs bird."""
        model = create_model(model_name)
        features = np.array(model.get_features(real_batch))

        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_cross = cosine(features[0], features[1])  # tench vs cock
        sim_same = cosine(features[2], features[3])  # turtle vs turtle
        assert sim_same > sim_cross, (
            f"Same-class similarity ({sim_same:.4f}) should exceed cross-class ({sim_cross:.4f})"
        )

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_deterministic_inference(self, model_name, real_images):
        model = create_model(model_name)
        x = real_images["tench"]
        f1 = np.array(model.get_features(x))
        f2 = np.array(model.get_features(x))
        np.testing.assert_allclose(f1, f2, atol=1e-5)

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_get_intermediate_layers(self, model_name, real_images):
        model = create_model(model_name)
        x = real_images["tench"]
        intermediates = model.get_intermediate_layers(x, n=2, return_class_token=True)
        assert len(intermediates) == 2
        for patches, cls_token in intermediates:
            assert patches.shape[1] == 196  # 14x14 patches
            dim = EXPECTED_DIMS[model_name]
            assert patches.shape[2] == dim
            assert cls_token.shape[1] == dim

    @pytest.mark.parametrize("model_name", DINOV3_MODELS)
    def test_forward_features_keys(self, model_name, real_images):
        model = create_model(model_name)
        x = real_images["tench"]
        ret = model.forward_features(x)
        assert isinstance(ret, dict)
        assert "x_norm_clstoken" in ret
        assert "x_norm_patchtokens" in ret
        assert "x_prenorm" in ret
        assert "x_storage_tokens" in ret


# ---------------------------------------------------------------------------
# RoPE Position Embedding
# ---------------------------------------------------------------------------
class TestRopePositionEmbedding:
    def test_output_shapes(self):
        rope = RopePositionEmbedding(embed_dim=384, num_heads=6)
        sin, cos = rope(H=14, W=14)
        assert sin.shape == (196, 64)
        assert cos.shape == (196, 64)

    def test_sin_cos_range(self):
        rope = RopePositionEmbedding(embed_dim=384, num_heads=6)
        sin, cos = rope(H=14, W=14)
        assert mx.min(sin).item() >= -1.0
        assert mx.max(sin).item() <= 1.0
        assert mx.min(cos).item() >= -1.0
        assert mx.max(cos).item() <= 1.0

    def test_different_resolutions(self):
        rope = RopePositionEmbedding(embed_dim=768, num_heads=12)
        sin7, cos7 = rope(H=7, W=7)
        sin14, cos14 = rope(H=14, W=14)
        assert sin7.shape == (49, 64)
        assert sin14.shape == (196, 64)

    def test_min_max_period_init(self):
        rope = RopePositionEmbedding(
            embed_dim=384,
            num_heads=6,
            base=None,
            min_period=2.0,
            max_period=100.0,
        )
        sin, cos = rope(H=14, W=14)
        assert sin.shape == (196, 64)


# ---------------------------------------------------------------------------
# Weight loading consistency
# ---------------------------------------------------------------------------
class TestDinov3WeightLoading:
    def test_two_loads_produce_identical_features(self, real_images):
        model1 = create_model("vit_small_patch16_224.dinov3")
        model2 = create_model("vit_small_patch16_224.dinov3")
        x = real_images["tench"]
        f1 = np.array(model1.get_features(x))
        f2 = np.array(model2.get_features(x))
        np.testing.assert_allclose(f1, f2, atol=1e-5)
