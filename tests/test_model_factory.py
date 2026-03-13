import pytest

from mlxim.model._factory import create_model, list_models
from mlxim.model._registry import MODEL_ENTRYPOINT


def test_create_model_invalid_name():
    with pytest.raises(ValueError, match="not available"):
        create_model("nonexistent_model_xyz", weights=False)


def test_create_model_no_weights():
    model = create_model("resnet18", weights=False)
    assert model is not None


def test_list_models(capsys):
    list_models()
    captured = capsys.readouterr()
    assert "resnet18" in captured.out


def test_model_registry_not_empty():
    assert len(MODEL_ENTRYPOINT) > 0
    assert "resnet18" in MODEL_ENTRYPOINT
