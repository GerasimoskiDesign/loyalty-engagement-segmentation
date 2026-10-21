from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from loyaltyseg.features import build_user_features

AS_OF = date(2026, 8, 31)
CFG = {
    "action_depth": {
        "loyalty_tab_view": 1,
        "game_play": 2,
        "raffle_entry": 2,
        "offer_unlock": 3,
        "offer_redeem": 5,
    },
    "no_activity_recency_days": 400,
}


def _ts(days_ago: float) -> pd.Timestamp:
    return pd.Timestamp(AS_OF) + pd.Timedelta(days=1) - pd.Timedelta(days=days_ago)


@pytest.fixture()
def tiny() -> dict[str, pd.DataFrame]:
    catalog = pd.DataFrame(
        {
            "item_id": [1, 2],
            "item_type": ["game", "offer"],
            "category": ["games", "tech"],
            "reward_value": [2.0, 50.0],
            "discount_depth": [0.0, 0.4],
        }
    )
    clusters = pd.DataFrame({"item_id": [1, 2], "content_cluster": [0, 1]})
    users = pd.DataFrame({"user_id": [1, 2, 3], "signup_date": [date(2024, 1, 1)] * 3})
    ev = pd.DataFrame(
        [
            # user 1: one game play session + one redeemed offer
            (1, _ts(10), "app_open", -1),
            (1, _ts(10), "loyalty_tab_view", 1),
            (1, _ts(10), "game_play", 1),
            (1, _ts(5), "loyalty_tab_view", 2),
            (1, _ts(5), "offer_unlock", 2),
            (1, _ts(4), "offer_redeem", 2),
            (1, _ts(3), "push_received", -1),
            (1, _ts(3), "push_opened", -1),
            (1, _ts(1), "push_received", -1),
            # user 2: a Browser
            (2, _ts(2), "app_open", -1),
            (2, _ts(40), "app_open", -1),
            # user 3: no events at all
        ],
        columns=["user_id", "ts", "event_type", "item_id"],
    )
    ev["event_type"] = ev["event_type"].astype("category")
    return {"catalog": catalog, "clusters": clusters, "users": users, "events": ev}


def test_depth_weighted_shares(tiny: dict[str, pd.DataFrame]) -> None:
    f = build_user_features(
        tiny["users"], [tiny["events"]], tiny["catalog"], tiny["clusters"], AS_OF, CFG
    ).set_index("user_id")
    # user 1 depth: game view 1 + play 2 = 3 ; offer view 1 + unlock 3 + redeem 5 = 9
    assert f.loc[1, "share_c0"] == pytest.approx(3 / 12)
    assert f.loc[1, "share_c1"] == pytest.approx(9 / 12)
    assert f.loc[1, "loyalty_actions"] == 5
    assert f.loc[1, "redeems"] == 1
    assert f.loc[1, "reward_value_redeemed"] == pytest.approx(50.0)
    assert f.loc[1, "discount_pref"] == pytest.approx(0.4)
    assert f.loc[1, "push_open_rate"] == pytest.approx(0.5)
    assert f.loc[1, "days_since_last_loyalty"] == pytest.approx(4.0, abs=1e-3)


def test_browser_and_empty_users(tiny: dict[str, pd.DataFrame]) -> None:
    f = build_user_features(
        tiny["users"], [tiny["events"]], tiny["catalog"], tiny["clusters"], AS_OF, CFG
    ).set_index("user_id")
    assert f.loc[2, "app_opens"] == 2 and f.loc[2, "app_opens_30d"] == 1
    assert f.loc[2, "loyalty_actions"] == 0
    assert f.loc[2, ["share_c0", "share_c1"]].sum() == 0
    assert np.isnan(f.loc[2, "days_since_last_loyalty"])
    assert f.loc[2, "recency_days_filled"] == 400
    assert np.isnan(f.loc[2, "push_open_rate"]) and f.loc[2, "push_open_rate_filled"] == 0
    assert f.loc[3, "app_opens"] == 0 and f.loc[3, "loyalty_ratio"] == 0
    assert len(f) == 3


def test_user_split_across_parts_is_rejected(tiny: dict[str, pd.DataFrame]) -> None:
    ev = tiny["events"]
    with pytest.raises(ValueError, match="more than one event part"):
        build_user_features(
            tiny["users"], [ev.iloc[:3], ev.iloc[3:]], tiny["catalog"], tiny["clusters"], AS_OF, CFG
        )


def test_feature_table_has_one_row_per_user(features: pd.DataFrame, users: pd.DataFrame) -> None:
    assert len(features) == len(users)
    assert features["user_id"].is_unique
    shares = features.filter(like="share_c")
    engaged = features["loyalty_actions"] > 0
    assert np.allclose(shares[engaged].sum(axis=1), 1.0)
    assert (shares[~engaged].sum(axis=1) == 0).all()
