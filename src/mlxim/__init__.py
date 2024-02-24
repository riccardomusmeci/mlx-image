import logging
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_version() -> str:
    """Read version from pyproject.toml."""
    pkg = "mlxim"

    try:
        return version(pkg)

    except PackageNotFoundError as e:
        logger.warning("Failed version retrieval using importlib.metadata. Trying with pyproject.toml")

        # For development. It works in local mode
        pyproject_file = Path(__file__).absolute().parents[2] / "pyproject.toml"

        try:
            with pyproject_file.open() as fp:
                pyproject = fp.read()
                match = re.search(r"version\s*=\s*[\"'](\d+\.\d+\.*\d*)[\"']", pyproject)
                version_from_toml = ""
                if match is not None:
                    version_from_toml = match.group(1)
                else:
                    raise PackageNotFoundError(
                        "Can't find information on version."
                        "Cannot retrieve version from 'version' field in pyproject.toml."
                    )
            return version_from_toml
        except Exception:
            raise PackageNotFoundError(
                "Can't find information on version."
                "Either install the package or make sure you are in the repo root folder."
            ) from e


__version__ = _read_version()
__all__ = ["__version__"]
