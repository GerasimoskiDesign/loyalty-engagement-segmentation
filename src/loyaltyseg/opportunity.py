"""Opportunity sizing: how big is each segment, and how much of it can we actually reach?

A segment that is 20% of the base but 2% reachable is a product problem (get push opt-in),
not a campaign opportunity. Reporting both numbers side by side prevents that confusion.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from loyaltyseg.segments import LAYERS

KEY_COMBINATIONS: tuple[tuple[str, str], ...] = (
    ("lifecycle", "channel"),
    ("archetype", "lifecycle"),
    ("interest", "lifecycle"),
)


def size_table(master: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Users, share of base, reachable users and reachable share for each group."""
    n = len(master)
    n_reach = int(master["reachable"].sum())
    g = master.groupby(by, observed=True)
    out = pd.DataFrame({"users": g.size(), "reachable_users": g["reachable"].sum()})
    out["share_of_base"] = out["users"] / max(n, 1)
    out["reachable_rate"] = out["reachable_users"] / out["users"]
    out["share_of_reachable"] = out["reachable_users"] / max(n_reach, 1)
    return out.reset_index().sort_values("users", ascending=False, ignore_index=True)


def layer_sizes(master: pd.DataFrame) -> pd.DataFrame:
    """Long table over archetype + all five layers."""
    frames = []
    for layer in ("archetype", *LAYERS):
        t = size_table(master, [layer]).rename(columns={layer: "label"})
        t.insert(0, "layer", layer)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def combination_sizes(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f"{a}_x_{b}": size_table(master, [a, b]) for a, b in KEY_COMBINATIONS}


def browser_highlight(master: pd.DataFrame) -> dict[str, Any]:
    """The headline number: users who open the app but never touched the loyalty hub."""
    b = master["lifecycle"] == "browser"
    reach_b = b & master["reachable"]
    return {
        "browsers": int(b.sum()),
        "browser_share_of_base": float(b.mean()),
        "reachable_browsers": int(reach_b.sum()),
        "browser_reachable_rate": float(reach_b.sum() / max(b.sum(), 1)),
        "browser_share_of_reachable": float(reach_b.sum() / max(master["reachable"].sum(), 1)),
        "base_users": int(len(master)),
        "reachable_users": int(master["reachable"].sum()),
    }
