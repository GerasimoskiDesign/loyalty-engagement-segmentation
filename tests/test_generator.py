from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from loyaltyseg.config import Settings
from loyaltyseg.generate import EVENT_TYPES, LOYALTY_EVENTS, generate
from loyaltyseg.io import dataset_fingerprint, read_parquet


def test_same_seed_gives_identical_outputs(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    generate(a, n_users=3_000, seed=42)
    generate(b, n_users=3_000, seed=42)
    generate(c, n_users=3_000, seed=43)
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    assert dataset_fingerprint(a) != dataset_fingerprint(c)


def test_regenerating_into_same_folder_is_idempotent(tmp_path: Path) -> None:
    generate(tmp_path, n_users=3_000, seed=5)
    first = dataset_fingerprint(tmp_path)
    generate(tmp_path, n_users=3_000, seed=5)
    assert dataset_fingerprint(tmp_path) == first


def test_users_table(run: Settings, users: pd.DataFrame) -> None:
    assert users["user_id"].is_unique
    # Fake +999 prefix: synthetic numbers can never be real subscriber numbers.
    assert users["msisdn"].str.match(r"^\+999\d{9}$").all()
    assert users["msisdn"].is_unique
    assert set(users["os"]) <= {"android", "ios"}
    assert (pd.to_datetime(users["signup_date"]) <= pd.Timestamp(run.as_of)).all()


def test_catalog_shape(run: Settings) -> None:
    cat = read_parquet(run.raw_dir / "catalog.parquet")
    assert 140 <= len(cat) <= 160
    assert set(cat["item_type"]) == {"game", "offer", "raffle"}
    assert {"tech", "shopping", "food", "travel", "entertainment", "fitness"} <= set(
        cat["category"]
    )
    assert set(cat["frequency"]) <= {"daily", "weekly", "one-off"}
    assert cat["discount_depth"].between(0, 1).all()


def test_events_within_window_and_known_types(run: Settings, events: pd.DataFrame) -> None:
    assert set(events["event_type"].astype(str)) == set(EVENT_TYPES)
    end = pd.Timestamp(run.as_of) + pd.Timedelta(days=1)
    assert (events["ts"] < end).all()
    assert (events["ts"] >= end - pd.Timedelta(days=367)).all()
    loy = events["event_type"].astype(str).isin(LOYALTY_EVENTS)
    assert (events.loc[loy, "item_id"] > 0).all()
    assert (events.loc[~loy, "item_id"] == -1).all()


def test_browsers_and_reachability_match_spec(master: pd.DataFrame) -> None:
    # ~35% Browsers and ~1/3 push-reachable are spec targets for the generator.
    assert 0.30 <= (master["lifecycle"] == "browser").mean() <= 0.40
    assert 0.28 <= master["reachable"].mean() <= 0.40


def test_ground_truth_is_not_a_feature(run: Settings, features: pd.DataFrame) -> None:
    truth = read_parquet(run.raw_dir / "ground_truth.parquet")
    assert set(truth.columns) == {"user_id", "archetype"}
    assert not any("archetype" in c or "truth" in c for c in features.columns)


def test_as_of_is_respected(tmp_path: Path) -> None:
    generate(tmp_path, n_users=500, seed=1, as_of=date(2025, 3, 31))
    ev = pd.concat(pd.read_parquet(p) for p in (tmp_path / "events").glob("*.parquet"))
    assert ev["ts"].max() < pd.Timestamp("2025-04-01")
