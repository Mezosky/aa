from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def resolve_config_path(path: str | Path) -> Path:
    """Keep explicit paths and legacy bare config names working after the move."""
    requested = Path(path)
    if requested.is_file():
        return requested
    if requested.parent == Path("."):
        candidate = Path(__file__).resolve().parent.parent / "configs" / requested.name
        if candidate.is_file():
            return candidate
    return requested


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with resolve_config_path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return cfg


def config_id(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def output_path(cfg: dict[str, Any], *parts: str) -> Path:
    path = Path(cfg["output_dir"]) / config_id(cfg)
    path.mkdir(parents=True, exist_ok=True)
    return path.joinpath(*parts)


def common_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Repeatable dotted config override",
    )
    return parser
