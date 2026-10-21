from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loyaltyseg.config import Settings
from loyaltyseg.content import cluster_content, elbow_point, name_cluster, select_k
from loyaltyseg.generate import generate_catalog
from loyaltyseg.io import read_json


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    return generate_catalog(np.random.default_rng(42))


def test_select_k_returns_int_in_range() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.3, size=(40, 2)) for c in ((0, 0), (5, 5), (0, 5))])
    sel = select_k(X, 2, 6)
    assert isinstance(sel.best_k, int)
    assert 2 <= sel.best_k <= 6
    assert sel.best_k == 3  # three well separated blobs
    assert sel.k_values == list(range(2, 7))


def test_select_k_rejects_bad_range() -> None:
    with pytest.raises(ValueError):
        select_k(np.zeros((10, 2)), 5, 3)
    with pytest.raises(ValueError):
        select_k(np.zeros((10, 2)), 1, 3)


def test_elbow_point_finds_the_knee() -> None:
    assert elbow_point([1, 2, 3, 4, 5, 6], [100, 40, 20, 17, 15, 14]) == 3


def test_pipeline_content_k_within_configured_range(run: Settings) -> None:
    k = read_json(run.output_dir / "content" / "k.json")["best_k"]
    cfg = run.pipeline["content"]
    assert isinstance(k, int) and cfg["k_min"] <= k <= cfg["k_max"]


def test_cluster_names_are_unique_and_deterministic(catalog: pd.DataFrame, run: Settings) -> None:
    cfg = run.pipeline["content"]
    items1, summary1, _ = cluster_content(catalog, cfg, seed=42)
    items2, summary2, _ = cluster_content(catalog, cfg, seed=42)
    assert summary1["content_cluster_name"].is_unique
    pd.testing.assert_frame_equal(items1, items2)
    assert len(items1) == len(catalog)


def test_naming_rules() -> None:
    cat = pd.DataFrame(
        {
            "item_type": ["offer"] * 6,
            "category": ["tech"] * 6,
            "reward_value": [5, 10, 20, 80, 90, 100],
            "discount_depth": [0.2] * 6,
            "frequency": ["one-off"] * 6,
            "effort_level": [1] * 6,
        }
    )
    rules = {"deep_discount_min": 0.45, "dominant_share_min": 0.6}
    assert name_cluster(cat.iloc[3:], cat, rules) == "High-value tech offers"
    assert name_cluster(cat.iloc[:2], cat, rules) == "Everyday tech offers"
    deep = cat.iloc[:2].assign(discount_depth=0.6)
    assert name_cluster(deep, cat, rules) == "Deep-discount tech offers"
    games = pd.DataFrame(
        {
            "item_type": ["game"] * 3,
            "category": ["games"] * 3,
            "frequency": ["daily"] * 3,
            "effort_level": [1, 1, 1],
            "reward_value": [1, 2, 3],
            "discount_depth": [0.0] * 3,
        }
    )
    assert name_cluster(games, games, rules) == "Daily casual games"
