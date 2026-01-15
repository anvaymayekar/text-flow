from pathlib import Path
from typing import Any
import yaml


def _getconst(key: str) -> str:
    """
    Retrieve a filesystem path string from the 'paths' section of params.yaml.

    Args:
        key (str): The key under 'paths' whose value should be retrieved.

    Returns:
        str: The resolved absolute path as a string.

    Raises:
        FileNotFoundError: If params.yaml does not exist.
        KeyError: If 'paths' section or the requested key is missing.
        ValueError: If the value is not a string.
    """
    yaml_file = Path("model/info.yaml")  # default known location

    if not yaml_file.exists():
        raise FileNotFoundError(f"params.yaml not found at {yaml_file.resolve()}")

    with yaml_file.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if "paths" not in config:
        raise KeyError("missing 'paths' section in params.yaml")

    paths = config["paths"]

    if key not in paths:
        raise KeyError(f"missing key '{key}' in 'paths' section")

    value = paths[key]

    if not isinstance(value, str):
        raise ValueError(f"value for '{key}' must be a string, got {type(value)}")

    return str(Path(value).resolve())
