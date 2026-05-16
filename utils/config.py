"""Config loading with _base inheritance support.

Each YAML config can declare ``_base: _defaults.yaml`` to inherit from a
shared defaults file.  The specific config only needs to list its unique
overrides; all other values come from the base.

Deep-merge semantics: nested dicts are merged recursively; scalars and lists
in the child config replace the base value entirely.  Setting a key to
``null`` explicitly overrides a non-null default.
"""

import copy
import os

import yaml


def _deep_merge(base, override):
    """Recursively merge *override* into *base* (in place). Override wins."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    """Load a YAML config with optional ``_base`` inheritance.

    If the loaded YAML contains a ``_base`` key (string path relative to the
    config file's directory), the base config is loaded first, then the child
    config is deep-merged on top.

    Returns:
        dict — fully merged config.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    base_ref = cfg.pop("_base", None)
    if base_ref is not None:
        base_path = os.path.join(os.path.dirname(path), base_ref)
        base_cfg = load_config(base_path)  # recursive for multi-level
        _deep_merge(base_cfg, cfg)
        return base_cfg
    return cfg
