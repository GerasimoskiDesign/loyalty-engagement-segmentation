from __future__ import annotations

import pandas as pd

from loyaltyseg.archetypes import choose_k, name_archetype
from loyaltyseg.config import Settings
from loyaltyseg.io import read_json, read_parquet

# ARI threshold for recovering the 7 planted archetypes, justified against baselines on
# this generator (12k users):
#   * Browsers-vs-everyone-else (the trivial rule split)         -> ARI ~0.33
#   * perfect labels but the two big diffuse groups merged
#     (Dormant + Occasional Players)                               -> ARI ~0.87
#   * perfect labels with two pairs merged                         -> ARI ~0.83
#   * this pipeline, three different seeds at 12k users            -> ARI 0.85-0.90
# 0.80 therefore demands that (nearly) every archetype is recovered as its own cluster,
# while leaving room for sampling noise in a small CI-sized population.
ARI_THRESHOLD = 0.80


def test_planted_archetypes_are_recoverable(run: Settings) -> None:
    ev = read_json(run.output_dir / "archetypes" / "evaluation.json")
    assert ev["ari_vs_planted"] >= ARI_THRESHOLD


def test_archetype_k_is_int_in_configured_range(run: Settings) -> None:
    ev = read_json(run.output_dir / "archetypes" / "evaluation.json")
    cfg = run.pipeline["archetypes"]
    k = ev["best_k_engaged"]
    assert isinstance(k, int) and cfg["k_min"] <= k <= cfg["k_max"]
    table = pd.read_csv(run.output_dir / "archetypes" / "k_selection.csv")
    assert list(table["k"]) == list(range(cfg["k_min"], cfg["k_max"] + 1))
    assert table["stability_ari"].between(-1, 1).all()


def test_every_user_has_one_archetype(run: Settings) -> None:
    a = read_parquet(run.output_dir / "archetypes" / "assignments.parquet")
    assert a["user_id"].is_unique and a["archetype"].notna().all()
    feats = read_parquet(run.output_dir / "features" / "user_features.parquet")
    m = a.merge(feats[["user_id", "loyalty_actions", "app_opens"]], on="user_id")
    browsers = m["archetype"] == "Browsers"
    assert ((m["loyalty_actions"] == 0) & (m["app_opens"] > 0)).eq(browsers).all()


def test_choose_k_prefers_stable_then_geometry() -> None:
    t = pd.DataFrame(
        {
            "k": [3, 4, 5, 6],
            "silhouette": [0.30, 0.50, 0.40, 0.45],
            "davies_bouldin": [1.5, 1.0, 1.2, 1.1],
            "stability_ari": [0.95, 0.60, 0.90, 0.92],
        }
    )
    t["stable"] = t["stability_ari"] >= 0.85
    assert choose_k(t) == 6  # k=4 has the best geometry but is unstable
    t["stable"] = False
    assert choose_k(t) == 3  # nothing stable -> most stable


def test_naming_rule_order() -> None:
    rules = {
        "dormant_min_recency_days": 90,
        "occasional_max_days_per_30d": 0.6,
        "game_share_min": 0.5,
        "category_share_min": 0.4,
        "deal_discount_min": 0.4,
    }
    base = {
        "recency_days_filled": 10,
        "loyalty_days_per_30d": 2.0,
        "type_share_game": 0.1,
        "cat_share_tech": 0.1,
        "cat_share_shopping": 0.1,
        "discount_pref": 0.2,
    }
    assert name_archetype(pd.Series({**base, "type_share_game": 0.8}), rules) == "Gamers"
    assert (
        name_archetype(
            pd.Series({**base, "type_share_game": 0.8, "recency_days_filled": 200}), rules
        )
        == "Dormant"
    )
    assert name_archetype(pd.Series({**base, "cat_share_tech": 0.7}), rules) == "Tech Chasers"
    assert name_archetype(pd.Series({**base, "discount_pref": 0.5}), rules) == "Deal Hunters"
    assert name_archetype(pd.Series(base), rules) == "Mixed Engagers"
