"""User feature table: one row per user, built from observable events only.

Every feature here is something a CRM team could compute from a real event stream. The
planted archetype (``ground_truth.parquet``) is deliberately never read in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from loyaltyseg.generate import LOYALTY_EVENTS

log = logging.getLogger(__name__)
DAY = 86_400


def as_of_end_ts(as_of: date) -> int:
    """Epoch seconds of the last second of the as-of day."""
    return int(pd.Timestamp(as_of).timestamp()) + DAY - 1


def share_columns(cluster_ids: Iterable[int]) -> list[str]:
    return [f"share_c{int(c)}" for c in cluster_ids]


def _to_epoch_s(ts: pd.Series) -> np.ndarray:
    return ts.to_numpy().astype("datetime64[s]").astype(np.int64)


def _part_features(
    ev: pd.DataFrame,
    items: pd.DataFrame,
    depth: dict[str, float],
    end_ts: int,
    cluster_ids: list[int],
) -> pd.DataFrame:
    """Aggregate one event part (a contiguous block of users) to per-user features."""
    et = ev["event_type"].astype(str).to_numpy()
    uid = ev["user_id"].to_numpy()
    ts = _to_epoch_s(ev["ts"])

    out: dict[str, pd.Series] = {}

    def count(mask: np.ndarray) -> pd.Series:
        return pd.Series(uid[mask]).value_counts()

    out["app_opens"] = count(et == "app_open")
    out["app_opens_30d"] = count((et == "app_open") & (ts > end_ts - 30 * DAY))
    out["push_received"] = count(et == "push_received")
    out["push_opened"] = count(et == "push_opened")
    app = et == "app_open"
    out["last_app_open_ts"] = pd.Series(ts[app]).groupby(uid[app]).max()

    is_loy = np.isin(et, list(LOYALTY_EVENTS))
    loy = pd.DataFrame(
        {
            "user_id": uid[is_loy],
            "ts": ts[is_loy],
            "event_type": et[is_loy],
            "item_id": ev["item_id"].to_numpy()[is_loy],
        }
    )
    loy = loy.merge(items, on="item_id", how="left", validate="many_to_one")
    loy["depth"] = loy["event_type"].map(depth).astype(float)
    loy["day"] = loy["ts"] // DAY
    g = loy.groupby("user_id")
    out["loyalty_actions"] = g.size()
    out["loyalty_active_days"] = g["day"].nunique()
    out["last_loyalty_ts"] = g["ts"].max()

    plays = loy[loy["event_type"].isin(["game_play", "raffle_entry"])]
    out["plays"] = plays.groupby("user_id").size()
    out["distinct_play_items"] = plays.groupby("user_id")["item_id"].nunique()
    out["play_days"] = plays.groupby("user_id")["day"].nunique()
    out["unlocks"] = loy[loy["event_type"] == "offer_unlock"].groupby("user_id").size()
    red = loy[loy["event_type"] == "offer_redeem"]
    out["redeems"] = red.groupby("user_id").size()
    out["reward_value_redeemed"] = red.groupby("user_id")["reward_value"].sum()

    offers = loy[loy["item_type"] == "offer"]
    w = offers["depth"] * offers["discount_depth"]
    out["discount_pref"] = (
        w.groupby(offers["user_id"]).sum() / offers.groupby("user_id")["depth"].sum()
    )

    total_depth = g["depth"].sum()
    out["loyalty_depth"] = total_depth
    feats = pd.DataFrame(out)

    def shares(col: str, prefix: str, keys: list) -> pd.DataFrame:
        pv = loy.pivot_table(
            index="user_id", columns=col, values="depth", aggfunc="sum", fill_value=0.0
        )
        pv = pv.reindex(columns=keys, fill_value=0.0)
        pv = pv.div(total_depth, axis=0)
        pv.columns = [f"{prefix}{k}" for k in keys]
        return pv

    parts = [feats, shares("content_cluster", "share_c", cluster_ids)]
    parts.append(shares("item_type", "type_share_", ["game", "offer", "raffle"]))
    cats = sorted(items["category"].unique())
    parts.append(shares("category", "cat_share_", cats))
    return pd.concat(parts, axis=1)


def build_user_features(
    users: pd.DataFrame,
    event_parts: Iterable[pd.DataFrame],
    catalog: pd.DataFrame,
    item_clusters: pd.DataFrame,
    as_of: date,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build the per-user feature table.

    ``event_parts`` must partition events by user (each user's events in exactly one part),
    which is how the generator writes them. Aggregates are computed part by part so memory
    stays proportional to the largest part, not to the full event history.
    """
    depth = {k: float(v) for k, v in cfg["action_depth"].items()}
    end_ts = as_of_end_ts(as_of)
    items = catalog[["item_id", "item_type", "category", "reward_value", "discount_depth"]]
    items = items.merge(item_clusters[["item_id", "content_cluster"]], on="item_id")
    cluster_ids = sorted(int(c) for c in item_clusters["content_cluster"].unique())

    frames = [_part_features(p, items, depth, end_ts, cluster_ids) for p in event_parts]
    agg = pd.concat(frames) if frames else pd.DataFrame()
    if agg.index.has_duplicates:
        raise ValueError("a user appears in more than one event part")

    f = users[["user_id"]].set_index("user_id").join(agg, how="left")
    count_cols = [
        "app_opens",
        "app_opens_30d",
        "push_received",
        "push_opened",
        "loyalty_actions",
        "loyalty_active_days",
        "plays",
        "play_days",
        "distinct_play_items",
        "unlocks",
        "redeems",
    ]
    for c in count_cols:
        f[c] = f[c].fillna(0).astype(np.int64)
    share_like = [c for c in f.columns if c.startswith(("share_c", "type_share_", "cat_share_"))]
    f[share_like] = f[share_like].fillna(0.0)
    f["reward_value_redeemed"] = f["reward_value_redeemed"].fillna(0.0)
    f["loyalty_depth"] = f["loyalty_depth"].fillna(0.0)
    f["discount_pref"] = f["discount_pref"].fillna(0.0)

    f["days_since_last_loyalty"] = (end_ts - f.pop("last_loyalty_ts")) / DAY
    f["days_since_last_app_open"] = (end_ts - f.pop("last_app_open_ts")) / DAY
    f["recency_days_filled"] = f["days_since_last_loyalty"].fillna(
        float(cfg.get("no_activity_recency_days", 400))
    )
    f["push_open_rate"] = f["push_opened"] / f["push_received"].where(f["push_received"] > 0)
    f["push_open_rate_filled"] = f["push_open_rate"].fillna(0.0)
    denom = (f["loyalty_actions"] + f["app_opens"]).where(lambda s: s > 0)
    f["loyalty_ratio"] = (f["loyalty_actions"] / denom).fillna(0.0)
    f["mean_redeemed_value"] = f["reward_value_redeemed"] / f["redeems"].where(f["redeems"] > 0)

    signup = pd.to_datetime(users.set_index("user_id")["signup_date"])
    f["days_since_signup"] = (pd.Timestamp(as_of) - signup).dt.days

    # Intensity per 30 observable days. A user who joined 40 days ago cannot have 12 months
    # of activity; raw counts would make every recent sign-up look "light" and the
    # clustering would split users by tenure instead of by behaviour.
    observed = np.clip(f["days_since_signup"].to_numpy() + 1, 1, 365)
    f["observed_days"] = observed
    f["loyalty_days_per_30d"] = f["loyalty_active_days"] / observed * 30
    f["app_opens_per_30d"] = f["app_opens"] / observed * 30
    f["reward_value_per_30d"] = f["reward_value_redeemed"] / observed * 30
    log.info("user features: %d users x %d columns", len(f), f.shape[1])
    return f.reset_index()
