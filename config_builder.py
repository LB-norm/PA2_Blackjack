from copy import deepcopy
from typing import Mapping, Any
from config_schema import Config

def deep_update(base: dict, patch: Mapping[str, Any]) -> dict:
    base = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base

def build_config(preset: dict | None, gui_overrides: dict | None) -> Config:
    cfg = Config().model_dump(mode="python")
    if preset:
        cfg = deep_update(cfg, preset)
    if gui_overrides:
        cfg = deep_update(cfg, gui_overrides)
    # any derived fields could be added here before validation, or attach after:
    return Config(**cfg)  # validates and returns typed model
