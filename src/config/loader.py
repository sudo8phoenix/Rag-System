"""YAML configuration loader with schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.settings import AppConfig


class ConfigLoadError(RuntimeError):
    """Raised when configuration file loading fails."""


class ConfigValidationError(ValueError):
    """Raised when schema validation fails for configuration."""


def load_yaml_file(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dictionary payload."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigLoadError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Unable to read config file {path}: {exc}") from exc

    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ConfigLoadError(
            f"Top-level YAML structure must be a mapping/object in {path}"
        )

    return raw


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    formatted: list[str] = [f"Config validation failed for {path}:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "invalid value")
        formatted.append(f"- {loc}: {msg}")
    return "\n".join(formatted)


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """Load YAML config and return a validated AppConfig object."""
    path = Path(config_path)
    payload = load_yaml_file(path)

    try:
        return AppConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error(path, exc)) from exc
