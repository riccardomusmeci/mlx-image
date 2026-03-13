import pytest

from mlxim.io.config import load_config


def test_load_config_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value\nnested:\n  a: 1\n  b: 2\n")
    result = load_config(str(config_file))
    assert result == {"key": "value", "nested": {"a": 1, "b": 2}}


def test_load_config_missing_file():
    with pytest.raises(FileExistsError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("key: [invalid\n  yaml: broken")
    with pytest.raises(ValueError, match="Failed to parse YAML"):
        load_config(str(config_file))
