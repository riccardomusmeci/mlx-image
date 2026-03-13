"""Test model architecture creation and forward pass without pretrained weights.

These tests exercise the __init__ and forward code paths of each architecture
family, dramatically improving coverage of model/*.py without needing downloads.
"""

import mlx.core as mx

from mlxim.model._factory import create_model


def _forward_test(model_name, img_size=224, num_classes=10):
    """Create model without weights and run a forward pass."""
    model = create_model(model_name, weights=False, num_classes=num_classes)
    model.eval()
    x = mx.random.normal((1, img_size, img_size, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape == (1, num_classes), f"{model_name}: expected (1, {num_classes}), got {out.shape}"


# ResNet family
def test_resnet18_init():
    _forward_test("resnet18")


def test_resnet34_init():
    _forward_test("resnet34")


def test_resnet50_init():
    _forward_test("resnet50")


# ViT family
def test_vit_base_patch16_224_init():
    _forward_test("vit_base_patch16_224", img_size=224)


def test_vit_base_patch32_224_init():
    _forward_test("vit_base_patch32_224", img_size=224)


# EfficientNet family
def test_efficientnet_b0_init():
    _forward_test("efficientnet_b0")


def test_efficientnet_b1_init():
    _forward_test("efficientnet_b1", img_size=240)


# MobileNet family
def test_mobilenet_v2_init():
    _forward_test("mobilenet_v2")


def test_mobilenet_v3_large_init():
    _forward_test("mobilenet_v3_large")


def test_mobilenet_v3_small_init():
    _forward_test("mobilenet_v3_small")


# RegNet family
def test_regnet_y_400mf_init():
    _forward_test("regnet_y_400mf")


def test_regnet_y_800mf_init():
    _forward_test("regnet_y_800mf")


def test_regnet_y_1_6gf_init():
    _forward_test("regnet_y_1_6gf")


# Swin Transformer family (doesn't accept num_classes kwarg, defaults to 1000)
def test_swin_tiny_init():
    _forward_test("swin_tiny_patch4_window7_224", img_size=224, num_classes=1000)


def test_swin_small_init():
    _forward_test("swin_small_patch4_window7_224", img_size=224, num_classes=1000)


# RegNet X family (covers a different branch than Y)
def test_regnet_x_400mf_init():
    _forward_test("regnet_x_400mf")


def test_regnet_x_800mf_init():
    _forward_test("regnet_x_800mf")


# Feature extraction tests
def test_resnet18_get_features():
    model = create_model("resnet18", weights=False, num_classes=10)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    features = model.get_features(x)
    mx.eval(features)
    # ResNet get_features returns a flat tensor (batch, feature_dim)
    assert features.ndim == 2
    assert features.shape[0] == 1


def test_resnet50_get_features():
    model = create_model("resnet50", weights=False, num_classes=10)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    features = model.get_features(x)
    mx.eval(features)
    assert features.ndim == 2
    assert features.shape[0] == 1
    # Bottleneck expansion=4, so 512*4=2048
    assert features.shape[1] == 2048


def test_vit_get_features():
    model = create_model("vit_base_patch32_224", weights=False, num_classes=10)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    features, attn_masks = model.get_features(x)
    mx.eval(features)
    # ViT get_features returns (cls_token_features, attn_masks)
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert features.shape[1] == 768  # hidden_dim for base
    assert isinstance(attn_masks, list)
    assert len(attn_masks) == 12  # num_layers for base


def test_vit_with_attn_masks():
    model = create_model("vit_base_patch32_224", weights=False, num_classes=10)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    out, attn = model(x, attn_masks=True)
    mx.eval(out)
    assert out.shape == (1, 10)
    assert isinstance(attn, list)


def test_swin_get_features():
    model = create_model("swin_tiny_patch4_window7_224", weights=False, num_classes=1000)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    features = model.get_features(x)
    mx.eval(features)
    # Swin get_features returns flat tensor
    assert features.ndim == 2
    assert features.shape[0] == 1


def test_efficientnet_get_features():
    model = create_model("efficientnet_b0", weights=False, num_classes=10)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    features = model.get_features(x)
    mx.eval(features)
    assert features.ndim == 2
    assert features.shape[0] == 1


def test_mobilenet_v2_get_features():
    model = create_model("mobilenet_v2", weights=False, num_classes=10)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    features = model.get_features(x)
    mx.eval(features)
    assert features.ndim == 2
    assert features.shape[0] == 1


# Test num_classes=0 (feature-only mode)
def test_resnet18_no_head():
    model = create_model("resnet18", weights=False, num_classes=0)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    out = model(x)
    mx.eval(out)
    # With num_classes=0, fc is Identity, output is feature dim
    assert out.shape[0] == 1
    assert out.shape[1] == 512  # BasicBlock expansion=1


def test_vit_no_head():
    model = create_model("vit_base_patch32_224", weights=False, num_classes=0)
    model.eval()
    x = mx.random.normal((1, 224, 224, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape[0] == 1
    assert out.shape[1] == 768  # hidden_dim
