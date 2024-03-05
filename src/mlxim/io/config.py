import os
from pathlib import Path
from typing import Dict, Optional

import yaml


def load_config(config_path: str) -> Dict:
    """Load a single yml file.

    Args:
        config_path (str): path to yml file

    Raises:
        FileExistsError: if config file does not exist

    Returns:
        Dict: yml dict
    """
    if os.path.exists(path=config_path) is False:
        raise FileExistsError(f"Config file {config_path} does not exist.")

    with open(config_path) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    return params
