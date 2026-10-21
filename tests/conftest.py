"""Shared fixtures. The full pipeline runs once per test session on a small population."""

from __future__ import annotations

import pandas as pd
import pytest

from loyaltyseg.cli import main
from loyaltyseg.config import Settings, load_settings
from loyaltyseg.io import read_events, read_parquet

N_USERS = 12_000


@pytest.fixture(scope="session")
def run(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Run ``python -m loyaltyseg all`` end to end through the real CLI entry point."""
    root = tmp_path_factory.mktemp("pipeline")
    data, out = root / "data", root / "outputs"
    rc = main(
        [
            "all",
            "--users",
            str(N_USERS),
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    return load_settings(data_dir=data, output_dir=out)


@pytest.fixture(scope="session")
def master(run: Settings) -> pd.DataFrame:
    return read_parquet(run.output_dir / "segments" / "master.parquet")


@pytest.fixture(scope="session")
def features(run: Settings) -> pd.DataFrame:
    return read_parquet(run.output_dir / "features" / "user_features.parquet")


@pytest.fixture(scope="session")
def events(run: Settings) -> pd.DataFrame:
    return read_events(run.raw_dir)


@pytest.fixture(scope="session")
def users(run: Settings) -> pd.DataFrame:
    return read_parquet(run.raw_dir / "users.parquet")
