"""Configuration loading.

All tunable thresholds live in YAML under ``config/`` so that a business user can read and
challenge every rule without opening Python. Code only supplies the *shape* of the rules.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEED = 42
DEFAULT_AS_OF = date(2026, 8, 31)
CONFIG_FILES = ("pipeline.yaml", "segments.yaml", "audiences.yaml")


def project_root() -> Path:
    """Repository root (two levels above ``src/loyaltyseg``)."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    """Run-wide settings resolved from CLI arguments, environment and YAML files."""

    config_dir: Path
    data_dir: Path
    output_dir: Path
    seed: int = DEFAULT_SEED
    as_of: date = DEFAULT_AS_OF
    pipeline: dict[str, Any] = field(default_factory=dict)
    segments: dict[str, Any] = field(default_factory=dict)
    audiences: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def config_hash(self) -> str:
        """SHA-256 over the raw bytes of every config file.

        Stored in each audience manifest so a campaign upload can always be traced back to
        the exact rules that produced it.
        """
        h = hashlib.sha256()
        for name in CONFIG_FILES:
            p = self.config_dir / name
            if p.exists():
                h.update(name.encode())
                h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()


def load_settings(
    config_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    as_of: date = DEFAULT_AS_OF,
) -> Settings:
    """Resolve settings. Precedence: explicit argument > environment variable > default."""
    root = project_root()
    cfg = Path(config_dir or os.environ.get("LOYALTYSEG_CONFIG_DIR") or root / "config")
    data = Path(data_dir or os.environ.get("LOYALTYSEG_DATA_DIR") or root / "data")
    out = Path(output_dir or os.environ.get("LOYALTYSEG_OUTPUT_DIR") or root / "outputs")
    return Settings(
        config_dir=cfg,
        data_dir=data,
        output_dir=out,
        seed=seed,
        as_of=as_of,
        pipeline=load_yaml(cfg / "pipeline.yaml"),
        segments=load_yaml(cfg / "segments.yaml"),
        audiences=load_yaml(cfg / "audiences.yaml"),
    )
