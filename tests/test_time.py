import re

from mlxim.utils.time import now


def test_now_format():
    result = now()
    pattern = r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$"
    assert re.match(pattern, result), f"Expected format YYYY-MM-DD-HH-MM-SS, got {result}"
