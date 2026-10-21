from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from loyaltyseg.audiences import build_audience, load_exclusions
from loyaltyseg.config import Settings
from loyaltyseg.io import read_json


def audience_dir(run: Settings) -> Path:
    return run.output_dir / "audiences" / f"as_of_{run.as_of.isoformat()}"


def _exports(run: Settings) -> tuple[dict, dict[str, pd.DataFrame]]:
    d = audience_dir(run)
    manifest = read_json(d / "manifest.json")
    files = {a["audience"]: pd.read_csv(d / a["file"]) for a in manifest["audiences"]}
    return manifest, files


def test_manifest_counts_match_files(run: Settings) -> None:
    manifest, files = _exports(run)
    assert manifest["as_of"] == run.as_of.isoformat()
    assert manifest["config_hash"] == run.config_hash()
    for a in manifest["audiences"]:
        c = a["counts"]
        assert a["rows_written"] == len(files[a["audience"]]) == c["after_cap"]
        steps = [c[k] for k in manifest["filter_order"]]
        assert steps == sorted(steps, reverse=True), "each filter can only remove users"


def test_exports_respect_reachability_dedupe_caps_exclusions(
    run: Settings, master: pd.DataFrame
) -> None:
    manifest, files = _exports(run)
    reachable = set(master.loc[master["reachable"], "user_id"])
    excluded = load_exclusions(
        run.raw_dir / "contact_history.csv", run.as_of, manifest["exclusion_days"]
    )
    assert excluded, "fixture should contain recently contacted users"
    seen: set[int] = set()
    for a in manifest["audiences"]:
        ids = files[a["audience"]]["user_id"]
        assert ids.is_unique
        assert set(ids) <= reachable
        assert not set(ids) & excluded
        assert not set(ids) & seen, "user exported to two audiences"
        seen |= set(ids)
        if a["max_size"] is not None:
            assert len(ids) <= a["max_size"]
    # the fixture is big enough that the browser cap and cross-audience dedupe both bite
    by = {a["audience"]: a for a in manifest["audiences"]}
    assert (
        by["high_value_redeemers"]["counts"]["after_dedupe"]
        < by["high_value_redeemers"]["counts"]["after_reachability"]
    )


def test_export_has_no_phone_numbers(run: Settings) -> None:
    _, files = _exports(run)
    for df in files.values():
        assert "msisdn" not in df.columns
        assert {"user_id", "audience", "as_of"} <= set(df.columns)


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
            "segment": ["x"] * 9 + ["y"],
            "reachable": [True, True, True, False, True, True, True, True, True, True],
        }
    )


def test_build_audience_filter_chain() -> None:
    spec = {"name": "t", "include": {"segment": ["x"]}, "max_size": 3}
    rows, counts = build_audience(_pool(), spec, {}, claimed={5}, excluded={6}, seed=1)
    assert counts == {
        "matched": 9,
        "after_reachability": 8,
        "after_dedupe": 6,
        "after_exclusion": 5,
        "after_cap": 3,
    }
    assert len(rows) == 3 and rows["user_id"].is_unique
    assert not {3, 5, 6} & set(rows["user_id"])
    again, _ = build_audience(_pool(), spec, {}, claimed={5}, excluded={6}, seed=1)
    assert list(again["user_id"]) == list(rows["user_id"])  # cap sample is reproducible


def test_build_audience_exclude_and_unreachable_allowed() -> None:
    spec = {
        "name": "t",
        "include": {"segment": ["x"]},
        "exclude": {"user_id": [1]},
        "require_reachable": False,
    }
    rows, counts = build_audience(_pool(), spec, {}, claimed=set(), excluded=set(), seed=1)
    assert counts["matched"] == 8 and counts["after_reachability"] == 8
    assert 3 in set(rows["user_id"]) and 1 not in set(rows["user_id"])


def test_exclusion_window(tmp_path: Path) -> None:
    p = tmp_path / "ch.csv"
    pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "campaign_id": ["a", "b", "c"],
            "sent_date": ["2026-08-31", "2026-08-25", "2026-08-24"],
        }
    ).to_csv(p, index=False)
    assert load_exclusions(p, date(2026, 8, 31), 7) == {1, 2}
    assert load_exclusions(p, date(2026, 8, 31), 0) == set()
    assert load_exclusions(tmp_path / "missing.csv", date(2026, 8, 31), 7) == set()
