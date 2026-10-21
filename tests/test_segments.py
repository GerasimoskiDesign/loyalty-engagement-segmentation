from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loyaltyseg.config import Settings
from loyaltyseg.generate import LOYALTY_EVENTS
from loyaltyseg.segments import LAYERS, deal_style, lifecycle, play_style

EXPECTED_LABELS = {
    "lifecycle": {"new", "active", "at-risk", "dormant", "browser"},
    "play_style": {"repeat player", "collector", "one-and-done", "non-player"},
    "deal_style": {"high-value seeker", "any-deal", "non-redeemer"},
    "channel": {"reachable-responsive", "reachable-unresponsive", "not reachable"},
}


def test_exactly_one_label_per_layer_no_nulls(master: pd.DataFrame, users: pd.DataFrame) -> None:
    assert master["user_id"].is_unique
    assert set(master["user_id"]) == set(users["user_id"])
    assert not master[["archetype", *LAYERS]].isna().any().any()
    for layer, allowed in EXPECTED_LABELS.items():
        assert set(master[layer]) <= allowed, layer
    assert (master[list(LAYERS)].map(lambda v: isinstance(v, str) and v != "")).all().all()


def test_browser_definition_is_exact(master: pd.DataFrame, events: pd.DataFrame) -> None:
    """Recompute 'app_open > 0 and zero loyalty events' straight from raw events."""
    et = events["event_type"].astype(str)
    opens = events.loc[et == "app_open"].groupby("user_id").size()
    loyal = events.loc[et.isin(LOYALTY_EVENTS)].groupby("user_id").size()
    m = master.set_index("user_id")
    expected = (opens.reindex(m.index, fill_value=0) > 0) & (
        loyal.reindex(m.index, fill_value=0) == 0
    )
    actual = m["lifecycle"] == "browser"
    assert (expected == actual).all()
    assert (m.loc[actual, "interest"] == "none").all()


def test_reachable_matches_definition(
    master: pd.DataFrame, users: pd.DataFrame, features: pd.DataFrame
) -> None:
    u = users.set_index("user_id")
    f = features.set_index("user_id")
    m = master.set_index("user_id")
    expected = u["push_enabled"] & u["in_country"] & (f["app_opens_30d"] > 0)
    assert (m["reachable"] == expected.loc[m.index]).all()
    assert (m.loc[~m["reachable"], "channel"] == "not reachable").all()
    assert (m.loc[m["reachable"], "channel"] != "not reachable").all()


def _f(**cols: list) -> pd.DataFrame:
    return pd.DataFrame(cols)


LIFE = {"new_max_days_since_signup": 30, "at_risk_min_days": 30, "dormant_min_days": 90}


@pytest.mark.parametrize(
    ("app", "loy", "signup", "rec", "expected"),
    [
        (5, 0, 400, np.nan, "browser"),
        (5, 0, 3, np.nan, "browser"),  # Browser wins over new by design
        (2, 4, 10, 2.0, "new"),
        (0, 0, 400, np.nan, "dormant"),  # no activity at all
        (3, 4, 400, 90.5, "dormant"),
        (3, 4, 400, 90.0, "at-risk"),  # boundary: exactly 90 days is still at-risk
        (3, 4, 400, 30.0, "at-risk"),
        (3, 4, 400, 29.9, "active"),
    ],
)
def test_lifecycle_rules(app: int, loy: int, signup: int, rec: float, expected: str) -> None:
    f = _f(
        app_opens=[app],
        loyalty_actions=[loy],
        days_since_signup=[signup],
        days_since_last_loyalty=[rec],
    )
    assert lifecycle(f, LIFE).iloc[0] == expected


def test_play_and_deal_style_rules() -> None:
    f = _f(plays=[0, 3, 9, 9], play_days=[0, 1, 4, 6], distinct_play_items=[0, 1, 2, 7])
    cfg = {"one_and_done_max_play_days": 1, "collector_min_distinct_items": 6}
    assert list(play_style(f, cfg)) == ["non-player", "one-and-done", "repeat player", "collector"]
    d = _f(redeems=[0, 2, 2], mean_redeemed_value=[np.nan, 12.0, 60.0])
    assert list(deal_style(d, {"high_value_min_mean_reward": 40})) == [
        "non-redeemer",
        "any-deal",
        "high-value seeker",
    ]


def test_thresholds_come_from_config(run: Settings) -> None:
    seg = run.segments
    for layer in ("lifecycle", "interest", "play_style", "deal_style", "channel"):
        assert layer in seg
