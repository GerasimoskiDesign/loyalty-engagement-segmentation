"""Content clustering: group the loyalty catalog into interpretable content clusters.

Why cluster the *content* first? User behaviour is spread over ~150 items, most of which any
single user never touches. Summarising items into a handful of content clusters turns a
sparse 150-column interaction matrix into a dense, explainable "interest profile" per user,
and gives campaign managers names they recognise ("High-value tech offers").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

TIER_ORDER = {"none": 0, "bronze": 1, "silver": 2, "gold": 3}
ONE_HOT_BLOCKS = ("item_type", "category", "frequency")


@dataclass
class KSelection:
    """Result of scanning candidate k values."""

    k_values: list[int]
    inertia: list[float]
    silhouette: list[float]
    best_k: int
    elbow_k: int

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"k": self.k_values, "inertia": self.inertia, "silhouette": self.silhouette}
        )


def item_features(catalog: pd.DataFrame, block_weights: dict[str, float]) -> pd.DataFrame:
    """One-hot categorical blocks + scaled numerics, each block multiplied by its weight.

    Weighting blocks (instead of individual columns) keeps the feature space honest: adding
    a seventh offer category should not silently increase how much "category" matters.
    """
    blocks = []
    for col in ONE_HOT_BLOCKS:
        oh = pd.get_dummies(catalog[col], prefix=col, dtype=float)
        blocks.append(oh * float(block_weights.get(col, 1.0)))
    num = pd.DataFrame(
        {
            "log_reward_value": np.log1p(catalog["reward_value"].astype(float)),
            "effort_level": catalog["effort_level"].astype(float),
            "discount_depth": catalog["discount_depth"].astype(float),
            "partner_tier": catalog["partner_tier"].map(TIER_ORDER).astype(float),
        }
    )
    scaled = StandardScaler().fit_transform(num)
    blocks.append(
        pd.DataFrame(scaled, columns=num.columns, index=catalog.index)
        * float(block_weights.get("numeric", 1.0))
    )
    return pd.concat(blocks, axis=1)


def elbow_point(k_values: list[int], inertia: list[float]) -> int:
    """Kneedle-style elbow: the k farthest from the straight line joining the end points."""
    x = np.asarray(k_values, dtype=float)
    y = np.asarray(inertia, dtype=float)
    if len(x) < 3:
        return int(x[0])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    dist = np.abs(yn - (1 - xn))  # line from (0,1) to (1,0)
    return int(x[int(np.argmax(dist))])


def select_k(X: np.ndarray, k_min: int, k_max: int, seed: int = 42, n_init: int = 20) -> KSelection:
    """Scan k in [k_min, k_max]; pick the highest silhouette, ties broken by smaller k.

    Silhouette is the decision rule because it rewards clusters that are both tight and
    separated, which is what makes a content cluster nameable. The elbow is reported next to
    it as a sanity check, not used to decide.
    """
    if k_min < 2 or k_max < k_min:
        raise ValueError(f"invalid k range [{k_min}, {k_max}]")
    k_max = min(k_max, len(X) - 1)
    ks, inertia, sil = [], [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        ks.append(k)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(X, km.labels_)))
    best = ks[int(np.argmax(np.round(sil, 6)))]
    return KSelection(ks, inertia, sil, int(best), elbow_point(ks, inertia))


def _relabel_by_first_item(labels: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    """Renumber clusters 0..k-1 by the smallest item_id they contain (stable ids)."""
    first = {lab: item_ids[labels == lab].min() for lab in np.unique(labels)}
    order = sorted(first, key=first.get)
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[lab] for lab in labels])


def name_cluster(members: pd.DataFrame, catalog: pd.DataFrame, rules: dict[str, Any]) -> str:
    """Deterministic name from the cluster centroid's dominant attributes.

    For K-Means the centroid *is* the mean of its members, so reading shares and means off
    the member rows is the same as reading the centroid in original (unscaled) units.
    """
    dom_share = float(rules.get("dominant_share_min", 0.6))
    item_type = members["item_type"].value_counts().idxmax()
    cat_counts = members["category"].value_counts(normalize=True)
    category = cat_counts.index[0] if cat_counts.iloc[0] >= dom_share else None
    freq_counts = members["frequency"].value_counts(normalize=True)
    freq = freq_counts.index[0] if freq_counts.iloc[0] >= dom_share else None

    if item_type == "game":
        effort = members["effort_level"].mean()
        style = (
            "casual"
            if effort <= rules.get("casual_effort_max", 1.5)
            else "challenge"
            if effort >= rules.get("challenge_effort_min", 2.5)
            else "skill"
        )
        prefix = {"daily": "Daily", "weekly": "Weekly", "one-off": "One-off"}.get(freq, "Mixed")
        return f"{prefix} {style} games"

    if item_type == "raffle":
        return f"{category.title()} prize raffles" if category else "Prize raffles"

    same_type = catalog.loc[catalog["item_type"] == item_type, "reward_value"]
    value = members["reward_value"].median()
    disc = members["discount_depth"].mean()
    if disc >= rules.get("deep_discount_min", 0.45):
        prefix = "Deep-discount"
    elif value >= same_type.quantile(2 / 3):
        prefix = "High-value"
    elif value <= same_type.quantile(1 / 3):
        prefix = "Everyday"
    else:
        prefix = "Mid-value"
    return f"{prefix} {category} offers" if category else f"{prefix} mixed offers"


def _unique_names(names: list[str], clusters: list[pd.DataFrame]) -> list[str]:
    out: list[str] = []
    for name, members in zip(names, clusters, strict=True):
        candidate = name
        if names.count(name) > 1:
            freq = members["frequency"].value_counts().idxmax()
            candidate = f"{name} ({freq})"
        n = 2
        base = candidate
        while candidate in out:
            candidate = f"{base} #{n}"
            n += 1
        out.append(candidate)
    return out


def cluster_content(
    catalog: pd.DataFrame, cfg: dict[str, Any], seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, KSelection]:
    """Cluster the catalog.

    Returns ``(item_clusters, cluster_summary, k_selection)`` where ``item_clusters`` has
    item_id, content_cluster, content_cluster_name.
    """
    X = item_features(catalog, cfg.get("block_weights", {})).to_numpy()
    sel = select_k(X, int(cfg["k_min"]), int(cfg["k_max"]), seed, int(cfg.get("n_init", 20)))
    km = KMeans(n_clusters=sel.best_k, n_init=int(cfg.get("n_init", 20)), random_state=seed).fit(X)
    labels = _relabel_by_first_item(km.labels_, catalog["item_id"].to_numpy())

    groups = [catalog[labels == c] for c in range(sel.best_k)]
    raw_names = [name_cluster(g, catalog, cfg.get("naming", {})) for g in groups]
    names = _unique_names(raw_names, groups)

    items = pd.DataFrame(
        {
            "item_id": catalog["item_id"].to_numpy(),
            "content_cluster": labels,
            "content_cluster_name": [names[c] for c in labels],
        }
    )
    summary = pd.DataFrame(
        [
            {
                "content_cluster": c,
                "content_cluster_name": names[c],
                "items": len(g),
                "dominant_type": g["item_type"].value_counts().idxmax(),
                "dominant_category": g["category"].value_counts().idxmax(),
                "median_reward_value": round(float(g["reward_value"].median()), 2),
                "mean_discount_depth": round(float(g["discount_depth"].mean()), 3),
                "mean_effort": round(float(g["effort_level"].mean()), 2),
            }
            for c, g in enumerate(groups)
        ]
    )
    log.info(
        "content clusters: k=%d (elbow %d), silhouette=%.3f",
        sel.best_k,
        sel.elbow_k,
        max(sel.silhouette),
    )
    return items, summary, sel
