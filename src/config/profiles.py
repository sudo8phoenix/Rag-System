"""Helpers for saving, loading, and listing named configuration profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from src.config.loader import load_config
from src.config.settings import AppConfig

PROFILE_FILE_PREFIX = "config-"
PROFILE_FILE_SUFFIXES = (".yaml", ".yml")


class ProfileConfigurationError(ValueError):
    """Raised when a profile name or profile path is invalid."""


def _config_directory(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve().parent


def normalize_profile_name(profile_name: str) -> str:
    """Normalize a profile name to a safe filesystem-friendly slug."""

    candidate = profile_name.strip()
    if not candidate:
        raise ProfileConfigurationError("Profile name cannot be empty")

    candidate = Path(candidate).name
    lower_candidate = candidate.lower()
    for suffix in PROFILE_FILE_SUFFIXES:
        if lower_candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            lower_candidate = candidate.lower()
            break

    if lower_candidate.startswith(PROFILE_FILE_PREFIX):
        candidate = candidate[len(PROFILE_FILE_PREFIX) :]

    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate.strip())
    candidate = candidate.strip("-._").lower()
    if not candidate:
        raise ProfileConfigurationError("Profile name cannot be empty")

    return candidate


def profile_path(
    profile_name: str, *, config_path: str | Path = "config/config.yaml"
) -> Path:
    """Return the file path for a profile name in the config directory."""

    normalized_name = normalize_profile_name(profile_name)
    return (
        _config_directory(config_path) / f"{PROFILE_FILE_PREFIX}{normalized_name}.yaml"
    )


def profile_name_from_path(path: str | Path) -> str | None:
    """Return the profile name if the path follows the profile naming pattern."""

    candidate = Path(path)
    lower_name = candidate.name.lower()
    if not lower_name.startswith(PROFILE_FILE_PREFIX):
        return None

    stem = candidate.stem
    if stem.lower().startswith(PROFILE_FILE_PREFIX):
        stem = stem[len(PROFILE_FILE_PREFIX) :]

    try:
        return normalize_profile_name(stem)
    except ProfileConfigurationError:
        return None


def list_profile_paths(*, config_path: str | Path = "config/config.yaml") -> list[Path]:
    """List saved profile files next to the active config file."""

    config_dir = _config_directory(config_path)
    if not config_dir.exists():
        return []

    profiles = [
        path
        for path in sorted(config_dir.glob(f"{PROFILE_FILE_PREFIX}*.yaml"))
        if path.is_file() and path.name != "config.yaml"
    ]
    return profiles


def list_profile_names(*, config_path: str | Path = "config/config.yaml") -> list[str]:
    """List saved profile names sorted alphabetically."""

    names: list[str] = []
    for path in list_profile_paths(config_path=config_path):
        name = profile_name_from_path(path)
        if name is not None:
            names.append(name)
    return names


def save_profile(
    config: AppConfig,
    profile_name: str,
    *,
    config_path: str | Path = "config/config.yaml",
) -> Path:
    """Save a validated config to a named profile file."""

    path = profile_path(profile_name, config_path=config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="python"), handle, sort_keys=False)
    return path


def load_profile(
    profile_name: str, *, config_path: str | Path = "config/config.yaml"
) -> AppConfig:
    """Load a named profile as a validated AppConfig."""

    return load_config(profile_path(profile_name, config_path=config_path))


def load_profile_from_path(profile_file: str | Path) -> AppConfig:
    """Load a profile directly from a file path."""

    return load_config(profile_file)


def profile_exists(
    profile_name: str, *, config_path: str | Path = "config/config.yaml"
) -> bool:
    """Return True when a profile file exists."""

    return profile_path(profile_name, config_path=config_path).exists()


def iter_profile_files(
    *, config_path: str | Path = "config/config.yaml"
) -> Iterable[Path]:
    """Iterate over profile files in the config directory."""

    return iter(list_profile_paths(config_path=config_path))
