"""
src/config.py
─────────────
Load and expose project configuration from config.yaml.
All modules import from here — never read yaml directly elsewhere.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env (FRED_API_KEY lives here)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return parsed config.yaml as a dict. Cached after first call."""
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get(path: str, default: Any = None) -> Any:
    """
    Dot-path accessor for nested config values.

    Example
    -------
    >>> get("features.momentum.windows")
    [21, 63, 252]
    """
    cfg = load_config()
    keys = path.split(".")
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val is None:
            return default
    return val


def fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def data_dir(subdir: str = "") -> Path:
    base = PROJECT_ROOT / get("data.raw_dir", "data/raw")
    return (base.parent / subdir) if subdir else base


def processed_dir() -> Path:
    return PROJECT_ROOT / get("data.processed_dir", "data/processed")


def tickers() -> list[str]:
    raw = get("universe.tickers", [])
    # Config stores tickers with YAML inline list syntax; flatten if nested
    flat: list[str] = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(str(item).strip())
    return [t for t in flat if t and not t.startswith("-")]
