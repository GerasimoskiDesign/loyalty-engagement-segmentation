"""Campaign audience exports (``python -m loyaltyseg audiences``).

Each audience goes through the same, ordered filter chain and every step is counted in the
run manifest, so "why is my audience smaller than the segment?" has a written answer:

1. **match**        include / exclude rules on the master table
2. **reachable**    keep push-reachable users only (if required)
3. **dedupe**       drop duplicate user_ids and users already claimed by a higher-priority
                    audience in this run (one user should not get three pushes in a day)
4. **exclusion**    drop users pushed in the last N days (contact-history file)
5. **cap**          optional max size, filled by a seeded random sample so the cut is
                    unbiased and reproducible

The export is a generic push-provider upload file (user_id + a few attributes). Phone
numbers are intentionally *not* exported: the push provider only needs the user id.
"""

from __future__ import annotations

import logging
import zlib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loyaltyseg.io import write_csv, write_json

log = logging.getLogger(__name__)
COUNT_STEPS = ("matched", "after_reachability", "after_dedupe", "after_exclusion", "after_cap")


def _rule_mask(df: pd.DataFrame, rules: dict[str, list] | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, values in (rules or {}).items():
        if col not in df.columns:
            raise KeyError(f"audience filter references unknown column {col!r}")
        mask &= df[col].isin(list(values))
    return mask


def load_exclusions(path: Path | None, as_of: date, days: int) -> set[int]:
    """User ids contacted within the last ``days`` days up to and including ``as_of``."""
    if path is None or days <= 0 or not Path(path).exists():
        return set()
    ch = pd.read_csv(path, dtype={"user_id": "int64"})
    sent = pd.to_datetime(ch["sent_date"]).dt.date
    start = as_of - timedelta(days=days - 1)
    recent = (sent >= start) & (sent <= as_of)
    return set(ch.loc[recent, "user_id"].astype(int))


def build_audience(
    pool: pd.DataFrame,
    spec: dict[str, Any],
    defaults: dict[str, Any],
    claimed: set[int],
    excluded: set[int],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the filter chain to one audience. Returns (rows, counts per step)."""
    counts: dict[str, int] = {}
    df = pool[_rule_mask(pool, spec.get("include"))]
    if spec.get("exclude"):
        df = df[~_rule_mask(df, spec.get("exclude"))]
    counts["matched"] = len(df)

    if spec.get("require_reachable", defaults.get("require_reachable", True)):
        df = df[df["reachable"]]
    counts["after_reachability"] = len(df)

    df = df.drop_duplicates("user_id")
    if spec.get("dedupe_across_audiences", defaults.get("dedupe_across_audiences", True)):
        df = df[~df["user_id"].isin(claimed)]
    counts["after_dedupe"] = len(df)

    df = df[~df["user_id"].isin(excluded)]
    counts["after_exclusion"] = len(df)

    cap = spec.get("max_size")
    if cap is not None and len(df) > int(cap):
        # Seed is derived from the run seed and the audience name, so adding a new audience
        # does not change which users another audience keeps.
        rng = np.random.default_rng([seed, zlib.crc32(spec["name"].encode())])
        keep = np.sort(rng.choice(len(df), size=int(cap), replace=False))
        df = df.iloc[keep]
    counts["after_cap"] = len(df)
    return df.sort_values("user_id"), counts


def export_audiences(
    master: pd.DataFrame,
    users: pd.DataFrame,
    cfg: dict[str, Any],
    out_dir: Path,
    as_of: date,
    seed: int,
    config_hash: str,
    exclusion_file: Path | None,
) -> dict[str, Any]:
    """Export every configured audience as CSV and write ``manifest.json``."""
    defaults = cfg.get("defaults", {})
    cols = list(defaults.get("export_columns", ["user_id"]))
    pool = master.merge(
        users[["user_id", "os", "preferred_channel"]],
        on="user_id",
        how="left",
        validate="one_to_one",
    )
    days = int(defaults.get("exclusion_days", 0))
    excluded = load_exclusions(exclusion_file, as_of, days)
    run_dir = out_dir / f"as_of_{as_of.isoformat()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in run_dir.glob("*.csv"):
        old.unlink()

    claimed: set[int] = set()
    entries = []
    for spec in cfg.get("audiences", []):
        rows, counts = build_audience(pool, spec, defaults, claimed, excluded, seed)
        claimed.update(rows["user_id"].tolist())
        export = rows[cols].copy()
        export.insert(1, "audience", spec["name"])
        export.insert(2, "as_of", as_of.isoformat())
        path = run_dir / f"{spec['name']}.csv"
        write_csv(export, path)
        if counts["matched"] == 0:
            log.warning("audience %s matched no users - check its filters", spec["name"])
        entries.append(
            {
                "audience": spec["name"],
                "description": spec.get("description", ""),
                "file": path.name,
                "counts": counts,
                "rows_written": len(export),
                "max_size": spec.get("max_size"),
            }
        )
        log.info("audience %-22s %s", spec["name"], counts)

    manifest = {
        "as_of": as_of.isoformat(),
        "seed": seed,
        "config_hash": config_hash,
        "exclusion_file": exclusion_file.name if exclusion_file else None,
        "exclusion_days": days,
        "excluded_users_in_window": len(excluded),
        "filter_order": list(COUNT_STEPS),
        "audiences": entries,
    }
    write_json(manifest, run_dir / "manifest.json")
    return manifest
