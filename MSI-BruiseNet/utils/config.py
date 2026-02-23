"""Configuration loader with CLI overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply dot-notation overrides like model.attention=se."""
    out = deepcopy(cfg)
    for item in overrides:
        k, v = item.split("=", 1)
        keys = k.split(".")
        ref = out
        for kk in keys[:-1]:
            ref = ref[kk]
        ref[keys[-1]] = _parse_value(v)
    return out


def load_config(path: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    """Load YAML config and apply optional overrides."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides or [])
