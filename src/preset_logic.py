# presets.py
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import yaml
from typing import Any, Mapping, List
from config_schema import Config
from config_builder import build_config, deep_update  

_PRESET_EXT = ".yaml"

def _diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    """Compare 2 mappings and output the diff"""
    out = {}
    for k, va in a.items():
        vb = b.get(k, object())
        if isinstance(va, Mapping) and isinstance(vb, Mapping):
            sub = _diff(va, vb)
            if sub:
                out[k] = sub
        elif k not in b or va != vb:
            out[k] = va
    return out

# --- FULL SNAPSHOT ---

def save_snapshot(cfg: Config, path: Path | str) -> None:
    """Write the entire Config."""
    p = Path(path)
    data = cfg.model_dump(mode="python")
    yaml.safe_dump({"kind": "snapshot", "config": data}, p.open("w"), sort_keys=True)

def load_snapshot(path: Path | str) -> Config:
    """Load a full Config snapshot."""
    p = Path(path)
    doc = yaml.safe_load(p.read_text())
    if doc.get("kind") != "snapshot" or "config" not in doc:
        raise ValueError("Not a snapshot file.")
    return Config.model_validate(doc["config"])


# --- GAME SETUP PATCH (rules + shoe only) ---
def save_game_setup_patch(cfg: Config, path: Path | str) -> None:
    """Save only the differences in rules/shoe vs defaults."""
    p = Path(path)
    defaults = Config().model_dump(mode="python")
    current  = cfg.model_dump(mode="python")
    patch = {
        "rules": _diff(current["rules"], defaults["rules"]),
        "shoe":  _diff(current["shoe"],  defaults["shoe"]),
        "config_version": current.get("config_version", 1),
    }
    yaml.safe_dump({"kind": "game_setup_patch", "preset": patch}, p.open("w"), sort_keys=True)

# --- GUI Commands
def apply_game_setup_patch_to(cfg: Config, path: Path | str) -> Config:
    """
    Apply a game-setup patch file to an existing Config (only rules/shoe).
    Leaves table/bankroll/simulation/reporting/policy untouched.
    """
    p = Path(path)
    doc = yaml.safe_load(p.read_text())
    if doc.get("kind") != "game_setup_patch" or "preset" not in doc:
        # also accept bare {rules:..., shoe:...}
        preset = {"rules": doc.get("rules", {}), "shoe": doc.get("shoe", {})}
    else:
        preset = doc["preset"]

    base = cfg.model_dump(mode="python")
    if "rules" in preset:
        base["rules"] = deep_update(base["rules"], preset["rules"])
    if "shoe" in preset:
        base["shoe"]  = deep_update(base["shoe"],  preset["shoe"])
    return Config(**base)  # validate

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_name(name: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip("_-")
    return s or "preset"

def list_game_setup_presets(presets_dir: str | Path) -> List[Path]:
    d = Path(presets_dir)
    return sorted(d.glob(f"*{_PRESET_EXT}")) if d.exists() else []

def apply_game_setup_preset(cfg: Config, preset_path: str | Path) -> Config:
    """Wrapper: apply rules+shoe patch file to cfg."""
    return apply_game_setup_patch_to(cfg, preset_path)

def save_game_setup_preset_as(cfg: Config, name: str, presets_dir: str | Path) -> Path:
    """Create new preset YAML from cfg.rules/shoe. Error if exists."""
    pdir = Path(presets_dir); _ensure_dir(pdir)
    path = pdir / f"{_safe_name(name)}{_PRESET_EXT}"
    if path.exists():
        raise FileExistsError(f"Preset exists: {path}")
    save_game_setup_patch(cfg, path)
    return path

def overwrite_game_setup_preset(cfg: Config, preset_path: str | Path) -> Path:
    """Overwrite existing preset atomically."""
    p = Path(preset_path)
    if not p.exists():
        raise FileNotFoundError(f"Preset not found: {p}")
    tmp = p.with_suffix(p.suffix + ".tmp")
    save_game_setup_patch(cfg, tmp)
    tmp.replace(p)
    return p

def reset_rules_shoe_to_defaults(cfg: Config) -> Config:
    """Reset only rules and shoe to schema defaults."""
    defaults = Config()
    base = cfg.model_dump(mode="python")
    base["rules"] = defaults.rules.model_dump(mode="python")
    base["shoe"]  = defaults.shoe.model_dump(mode="python")
    return Config(**base)
