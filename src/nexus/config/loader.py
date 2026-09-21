"""Config loader (YAML + env overrides)."""
from __future__ import annotations
import os
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_default(config_path: str | Path = "configs/default.yaml") -> dict:
    config_path = Path(config_path)
    cfg = load_yaml(config_path) if config_path.exists() else {}
    models_path = config_path.with_name("models.yaml")
    if models_path.exists():
        configured_models = load_yaml(models_path).get("models", {})
        cfg["models"] = {**configured_models, **cfg.get("models", {})}
    cfg.setdefault("max_steps", int(os.getenv("NEXUS_MAX_STEPS", "10")))
    cfg.setdefault("debug", os.getenv("NEXUS_DEBUG", "false").lower() == "true")
    return cfg
