import logging
import os
import yaml
from typing import Literal
from .constants import _getconst

_configured = False


def log(
    message: str,
    mode: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    """
    Log a message using strictly YAML-defined parameters.

    Args:
        message: Message to log.
        mode: Logging level (default: INFO)

    Raises:
        KeyError: If any required logging parameter is missing in YAML.
    """

    global _configured
    if not _configured:
        # Load YAML
        INFO_PATH = _getconst("info")
        if not os.path.exists(INFO_PATH):
            raise FileNotFoundError(f"params.yaml not found at {INFO_PATH}")

        with open(INFO_PATH, "r") as f:
            config = yaml.safe_load(f)

        if "logging" not in config:
            raise KeyError("Missing 'logging' section in params.yaml")
        log_config = config["logging"]

        required_keys = ["level", "file", "format"]
        for key in required_keys:
            if key not in log_config:
                raise KeyError(f"Missing '{key}' in logging section of info.yaml")

        log_level = getattr(logging, log_config["level"].upper())
        log_file = _getconst("logs")
        log_format = log_config["format"]

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        _configured = True

    # Log the message at requested level
    level = mode.upper()
    if level == "DEBUG":
        logging.debug(message)
    elif level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "CRITICAL":
        logging.critical(message)
    else:
        raise ValueError(f"Invalid logging mode: {mode}")
