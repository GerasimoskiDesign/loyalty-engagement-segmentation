"""Five transparent, rule-based segmentation layers on top of the archetypes.

Why rules on top of clusters? Clusters are good at discovering structure, but campaign teams
need labels that are stable, explainable in one sentence and adjustable without retraining.
The layers below are deliberately simple; every threshold lives in ``config/segments.yaml``.
Each layer function returns exactly one label per user, with no nulls, by construction:
``np.select`` always falls through to a default.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LAYERS = ("lifecycle", "interest", "play_style", "deal_style", "channel")


def is_browser(features: pd.DataFrame) -> pd.Series:
    """The Browser definition, in one place: opens the app, never touched the loyalty hub."""
    return (features["app_opens"] > 0) & (features["loyalty_actions"] == 0)


def reachable(features: pd.DataFrame, users: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Push-reachable: push enabled, in country, and app opened in the last N days.

    A push registration on a device that has not opened the app for a month is likely stale,
    so counting it as reachable would inflate every audience estimate.
    """
    days = int(cfg.get("active_device_days", 30))
    active = features["days_since_last_app_open"].fillna(np.inf) < days
    u = users.set_index("user_id").loc[features["user_id"]]
    return pd.Series(
        u["push_enabled"].to_numpy() & u["in_country"].to_numpy() & active.to_numpy(),
        index=features.index,
    )


def lifecycle(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    rec = features["days_since_last_loyalty"]
    no_loyalty = rec.isna()
    conds = [
        is_browser(features),
        features["days_since_signup"] < cfg["new_max_days_since_signup"],
        no_loyalty | (rec > cfg["dormant_min_days"]),
        rec >= cfg["at_risk_min_days"],
    ]
    labels = ["browser", "new", "dormant", "at-risk"]
    return pd.Series(np.select(conds, labels, default="active"), index=features.index)


def interest(
    features: pd.DataFrame, cluster_names: dict[int, str], cfg: dict[str, Any]
) -> pd.Series:
    cols = sorted(
        (c for c in features.columns if c.startswith("share_c")),
        key=lambda c: int(c.removeprefix("share_c")),
    )
    S = features[cols].to_numpy()
    top = S.argmax(axis=1)
    top_share = S.max(axis=1)
    names = np.array([cluster_names[int(c.removeprefix("share_c"))] for c in cols], dtype=object)
    out = np.where(
        features["loyalty_actions"].to_numpy() == 0,
        "none",
        np.where(top_share < cfg["min_dominant_share"], "mixed", names[top]),
    )
    return pd.Series(out, index=features.index)


def play_style(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    conds = [
        features["plays"] == 0,
        features["play_days"] <= cfg["one_and_done_max_play_days"],
        features["distinct_play_items"] >= cfg["collector_min_distinct_items"],
    ]
    return pd.Series(
        np.select(conds, ["non-player", "one-and-done", "collector"], default="repeat player"),
        index=features.index,
    )


def deal_style(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    mean_val = features["mean_redeemed_value"].fillna(0.0)
    conds = [features["redeems"] == 0, mean_val >= cfg["high_value_min_mean_reward"]]
    return pd.Series(
        np.select(conds, ["non-redeemer", "high-value seeker"], default="any-deal"),
        index=features.index,
    )


def channel(features: pd.DataFrame, is_reachable: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    rate = features["push_open_rate"].fillna(0.0)
    responsive = (features["push_received"] >= cfg["min_push_received"]) & (
        rate >= cfg["responsive_min_open_rate"]
    )
    conds = [~is_reachable, responsive]
    return pd.Series(
        np.select(
            conds, ["not reachable", "reachable-responsive"], default="reachable-unresponsive"
        ),
        index=features.index,
    )


def build_master(
    features: pd.DataFrame,
    users: pd.DataFrame,
    archetypes: pd.DataFrame,
    cluster_names: dict[int, str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Master table: user_id + archetype + one label per layer + reachable flag."""
    f = features.reset_index(drop=True)
    reach = reachable(f, users, cfg.get("reachability", {}))
    master = pd.DataFrame(
        {
            "user_id": f["user_id"].to_numpy(),
            "lifecycle": lifecycle(f, cfg["lifecycle"]).to_numpy(),
            "interest": interest(f, cluster_names, cfg["interest"]).to_numpy(),
            "play_style": play_style(f, cfg["play_style"]).to_numpy(),
            "deal_style": deal_style(f, cfg["deal_style"]).to_numpy(),
            "channel": channel(f, reach, cfg["channel"]).to_numpy(),
            "reachable": reach.to_numpy(),
        }
    )
    master = master.merge(
        archetypes[["user_id", "archetype"]], on="user_id", how="left", validate="one_to_one"
    )
    master = master[["user_id", "archetype", *LAYERS, "reachable"]]
    if master[["archetype", *LAYERS]].isna().any().any():
        raise ValueError("segmentation produced missing labels")
    log.info(
        "master table: %d users, reachable %.1f%%", len(master), 100 * master["reachable"].mean()
    )
    return master
