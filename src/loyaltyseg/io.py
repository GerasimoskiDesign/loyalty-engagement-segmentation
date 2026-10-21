"""Reading/writing helpers and dataset fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EVENTS_SUBDIR = "events"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def event_parts(raw_dir: Path) -> list[Path]:
    """Event part files, sorted. Each part holds a contiguous block of user_ids."""
    return sorted((raw_dir / EVENTS_SUBDIR).glob("part-*.parquet"))


def iter_event_parts(raw_dir: Path, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    """Yield events one part at a time.

    Events are partitioned by user block so that every per-user aggregate can be computed
    part by part. That keeps memory flat at 800k+ users instead of loading ~40M rows at once.
    """
    for p in event_parts(raw_dir):
        yield read_parquet(p, columns=columns)


def read_events(raw_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load all events (fine for tests / small samples)."""
    parts = list(iter_event_parts(raw_dir, columns))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame (independent of Parquet writer metadata)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def dataset_fingerprint(raw_dir: Path) -> str:
    """One hash over every generated table, used by the reproducibility test."""
    h = hashlib.sha256()
    for name in ("users", "catalog", "ground_truth"):
        h.update(frame_hash(read_parquet(raw_dir / f"{name}.parquet")).encode())
    for part in iter_event_parts(raw_dir):
        h.update(frame_hash(part).encode())
    ch = raw_dir / "contact_history.csv"
    if ch.exists():
        h.update(ch.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
