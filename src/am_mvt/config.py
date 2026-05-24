from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_environment() -> None:
    """
    Load environment variables from the local .env file.

    The .env file should not be committed to GitHub.
    """
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load project configuration from config.yaml.

    Parameters
    ----------
    config_path:
        Optional path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary.")

    return config


def get_path(*parts: str) -> Path:
    """
    Build an absolute path from the project root.

    Example
    -------
    get_path("data", "processed", "modelling_dataset.csv")
    """
    return PROJECT_ROOT.joinpath(*parts)