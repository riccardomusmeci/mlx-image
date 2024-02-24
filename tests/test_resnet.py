import mlx.core as mx

from mlxim.model import create_model


def test_resnet18():
    model = create_model("resnet18", weights=False, num_classes=5)
    model.eval()
    x = mx.random.uniform(shape=(1, 224, 224, 3))
    out = model(x)
    assert out.shape == (1, 5)
