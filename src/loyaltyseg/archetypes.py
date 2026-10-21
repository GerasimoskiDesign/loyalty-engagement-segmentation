"""User archetypes: StandardScaler -> optional PCA -> MiniBatchKMeans.

Two-stage design:

1. Users with **zero loyalty actions** are labelled by rule ("Browsers" if they open the app,
   otherwise "Inactive"). Their interest vector is all zeros, so clustering them adds no
   information; worse, a 35% point mass distorts the scaler and soaks up a centroid.
2. Everyone else is clustered.

k for stage 2 is chosen with three signals, because each one alone is easy to fool:

* **Silhouette** (on a fixed 20k sample -- the full O(n^2) version is not affordable at 800k)
  rewards tight, well-separated clusters.
* **Davies-Bouldin** (lower is better) is a second geometric opinion.
* **Stability**: the same k is fitted with 5 different seeds and the mean pairwise adjusted
  Rand index between the runs is computed. A segmentation that reshuffles when the seed
  changes cannot be used to run campaigns month after month.

On real data there is no ground truth, so stability and business interpretability are the
only guard rails. Here we additionally score against the planted archetypes (ARI), which
is a luxury of synthetic data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


@dataclass
class ArchetypeKSelection:
    table: pd.DataFrame  # k, silhouette, davies_bouldin, stability_ari, inertia, stable
    best_k: int


def share_cols(features: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in features.columns if c.startswith("share_c")),
        key=lambda c: int(c.removeprefix("share_c")),
    )


def feature_matrix(
    features: pd.DataFrame, cfg: dict[str, Any], seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Select, transform, scale and (optionally) PCA-reduce the clustering features.

    Every column is standardised, so each content-cluster share and each behavioural
    feature gets the same prior vote. log1p is applied first to heavy-tailed columns so a
    handful of power users do not define the scale for everyone.
    """
    shares = share_cols(features)
    cols = shares + list(cfg["feature_columns"])
    X = features[cols].astype(float).copy()
    alpha = float(cfg.get("share_smoothing_alpha", 0.0))
    if alpha > 0:
        # Additive (empirical-Bayes style) smoothing: a user with 2 interactions should not
        # get a "100% travel raffles" profile. Shares are pulled towards the population mix
        # with the weight of ``alpha`` depth points; heavy users are barely affected.
        depth = features["loyalty_depth"].to_numpy(dtype=float)[:, None]
        raw = X[shares].to_numpy()
        prior = (raw * depth).sum(axis=0) / max(depth.sum(), 1e-12)
        X[shares] = (raw * depth + alpha * prior) / (depth + alpha)
    for c in cfg.get("log1p_columns", []):
        X[c] = np.log1p(X[c])
    Z = StandardScaler().fit_transform(X.to_numpy())
    pca_cfg = cfg.get("pca", {})
    if pca_cfg.get("enabled", False):
        pca = PCA(
            n_components=float(pca_cfg.get("variance", 0.95)), svd_solver="full", random_state=seed
        )
        Z = pca.fit_transform(Z)
        log.info(
            "PCA kept %d components (%.1f%% variance)",
            pca.n_components_,
            100 * pca.explained_variance_ratio_.sum(),
        )
    return Z, cols


def _fit(X: np.ndarray, k: int, seed: int, cfg: dict[str, Any]) -> MiniBatchKMeans:
    """Best of ``restarts`` MiniBatchKMeans runs, judged by inertia on the full data.

    MiniBatchKMeans' own ``n_init`` compares initialisations on a small sample, which on this
    data regularly lets a poor local optimum win (e.g. splitting one diffuse group in two
    while merging two compact ones). Comparing finished runs on full-data inertia removes
    most of that seed-to-seed reshuffling at a modest cost.
    """
    best: MiniBatchKMeans | None = None
    for r in range(max(int(cfg.get("restarts", 1)), 1)):
        m = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed * 1000 + r,
            batch_size=int(cfg.get("batch_size", 4096)),
            n_init=int(cfg.get("n_init", 3)),
        ).fit(X)
        if best is None or m.inertia_ < best.inertia_:
            best = m
    assert best is not None
    return best


def choose_k(table: pd.DataFrame) -> int:
    """Decision rule over the k-scan table.

    1. Keep only *stable* k (mean pairwise ARI across seeds >= threshold).
    2. Among those, rank silhouette (high is good) and Davies-Bouldin (low is good) and take
       the best average rank; ties go to the smaller k (simpler to explain and operate).
    3. If no k is stable, fall back to the most stable k.
    """
    cand = table[table["stable"]]
    if cand.empty:
        return int(table.sort_values(["stability_ari", "k"], ascending=[False, True]).iloc[0]["k"])
    rank = (cand["silhouette"].rank(ascending=False) + cand["davies_bouldin"].rank()) / 2
    cand = cand.assign(rank=rank).sort_values(["rank", "k"])
    return int(cand.iloc[0]["k"])


def select_k(X: np.ndarray, cfg: dict[str, Any], seed: int = 42) -> ArchetypeKSelection:
    """Scan k in [k_min, k_max] (capped by sample size) and apply :func:`choose_k`."""
    k_min, k_max = int(cfg["k_min"]), int(cfg["k_max"])
    if k_min < 2 or k_max < k_min:
        raise ValueError(f"invalid k range [{k_min}, {k_max}]")
    if len(X) <= k_min:
        raise ValueError(f"need more than {k_min} users to cluster, got {len(X)}")
    k_max = min(k_max, len(X) - 1)
    seeds = [int(s) for s in cfg.get("stability_seeds", [seed])]
    rng = np.random.default_rng(seed)
    n_sample = min(int(cfg.get("silhouette_sample", 20_000)), len(X))
    sample = np.sort(rng.choice(len(X), size=n_sample, replace=False))
    min_ari = float(cfg.get("stability_min_ari", 0.85))

    rows = []
    for k in range(k_min, k_max + 1):
        fits = [_fit(X, k, s, cfg) for s in seeds]
        labels = [m.labels_ for m in fits]
        aris = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
        lab0 = labels[0][sample]
        geometric_ok = len(np.unique(lab0)) > 1
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X[sample], lab0)) if geometric_ok else -1.0,
                "davies_bouldin": (
                    float(davies_bouldin_score(X[sample], lab0)) if geometric_ok else np.inf
                ),
                "stability_ari": float(np.mean(aris)) if aris else 1.0,
                "inertia": float(fits[0].inertia_),
            }
        )
        log.info(
            "archetype k=%d silhouette=%.3f DB=%.3f stability=%.3f",
            k,
            rows[-1]["silhouette"],
            rows[-1]["davies_bouldin"],
            rows[-1]["stability_ari"],
        )
    table = pd.DataFrame(rows)
    table["stable"] = table["stability_ari"] >= min_ari
    return ArchetypeKSelection(table=table, best_k=choose_k(table))


PROFILE_COLUMNS = [
    "loyalty_actions",
    "loyalty_days_per_30d",
    "recency_days_filled",
    "app_opens_per_30d",
    "type_share_game",
    "type_share_offer",
    "type_share_raffle",
    "cat_share_tech",
    "cat_share_shopping",
    "discount_pref",
    "redeems",
    "reward_value_redeemed",
    "push_open_rate_filled",
    "loyalty_ratio",
]
MEDIAN_COLUMNS = {
    "loyalty_actions",
    "loyalty_days_per_30d",
    "recency_days_filled",
    "app_opens_per_30d",
    "redeems",
    "reward_value_redeemed",
}


def profile(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-cluster profile: medians for heavy-tailed counts, means for shares and rates."""
    df = features[PROFILE_COLUMNS].assign(cluster=labels)
    g = df.groupby("cluster")
    agg = {c: ("median" if c in MEDIAN_COLUMNS else "mean") for c in PROFILE_COLUMNS}
    prof = g.agg(agg)
    prof.insert(0, "users", g.size())
    prof.insert(1, "share", prof["users"] / len(df))
    return prof


def name_archetype(row: pd.Series, rules: dict[str, Any]) -> str:
    """Ordered business rules for an *engaged* cluster; the first rule that fires wins.

    Order matters: "lapsed" and "rarely plays" are statements about *whether* a user engages
    and override *what* they engage with.
    """
    if row["recency_days_filled"] > rules.get("dormant_min_recency_days", 90):
        return "Dormant"
    if row["loyalty_days_per_30d"] <= rules.get("occasional_max_days_per_30d", 0.6):
        return "Occasional Players"
    if row["type_share_game"] >= rules.get("game_share_min", 0.5):
        return "Gamers"
    if row["cat_share_tech"] >= rules.get("category_share_min", 0.4):
        return "Tech Chasers"
    if row["cat_share_shopping"] >= rules.get("category_share_min", 0.4):
        return "Shoppers"
    if row["discount_pref"] >= rules.get("deal_discount_min", 0.4):
        return "Deal Hunters"
    return "Mixed Engagers"


def name_clusters(prof: pd.DataFrame, rules: dict[str, Any]) -> dict[int, str]:
    """Apply :func:`name_archetype`; duplicate names get " 2", " 3" by descending size."""
    raw = {int(c): name_archetype(r, rules) for c, r in prof.iterrows()}
    names: dict[int, str] = {}
    for name in sorted(set(raw.values())):
        members = sorted(
            (c for c, n in raw.items() if n == name), key=lambda c: (-prof.loc[c, "users"], c)
        )
        for i, c in enumerate(members):
            names[c] = name if i == 0 else f"{name} {i + 1}"
    return names


def fit_archetypes(
    features: pd.DataFrame, cfg: dict[str, Any], seed: int = 42, k: int | None = None
) -> dict[str, Any]:
    """Full archetype step. Returns assignments, profile, k-selection table and best k.

    Cluster ids 0..k-1 are the engaged clusters; id k is "Browsers" and id k+1 "Inactive"
    (only present if such users exist).
    """
    engaged = features["loyalty_actions"].to_numpy() > 0
    fe = features[engaged]
    X, cols = feature_matrix(fe, cfg, seed)
    sel = None
    if k is None:
        # The k scan fits (k range x 5 seeds x restarts) models. Above `selection_sample`
        # engaged users it runs on a seeded random sample; the final model uses everyone.
        cap = int(cfg.get("selection_sample", 150_000))
        Xs = X
        if len(X) > cap:
            pick = np.random.default_rng(seed).choice(len(X), size=cap, replace=False)
            Xs = X[np.sort(pick)]
        sel = select_k(Xs, cfg, seed)
        sel.table.attrs["stability_min_ari"] = float(cfg.get("stability_min_ari", 0.85))
    best_k = k if k is not None else sel.best_k
    model = _fit(X, best_k, seed, cfg)

    labels = np.empty(len(features), dtype=np.int64)
    labels[engaged] = model.labels_
    app = features["app_opens"].to_numpy() > 0
    labels[~engaged & app] = best_k
    labels[~engaged & ~app] = best_k + 1

    prof_eng = profile(fe, model.labels_)
    names = name_clusters(prof_eng, cfg.get("naming", {}))
    names[best_k] = "Browsers"
    names[best_k + 1] = "Inactive"
    prof = profile(features, labels)
    prof.insert(0, "archetype", [names[int(c)] for c in prof.index])

    assigned = pd.DataFrame(
        {
            "user_id": features["user_id"].to_numpy(),
            "archetype_cluster": labels,
            "archetype": [names[int(c)] for c in labels],
        }
    )
    log.info("archetypes: k=%d engaged clusters -> %s", best_k, sorted(set(assigned["archetype"])))
    return {
        "assignments": assigned,
        "profile": prof,
        "k_table": sel.table if sel is not None else None,
        "best_k": best_k,
        "feature_columns": cols,
    }


def evaluate_against_truth(
    assigned: pd.DataFrame, truth: pd.DataFrame
) -> tuple[float, pd.DataFrame]:
    """ARI and confusion table (rows: found archetype, columns: planted archetype).

    Only possible on synthetic data. The ARI compares cluster ids, so it is not affected by
    whether the naming rules got the names right; the confusion table shows that separately.
    """
    m = assigned.merge(truth, on="user_id", suffixes=("", "_true"), validate="one_to_one")
    ari = float(adjusted_rand_score(m["archetype_true"], m["archetype_cluster"]))
    conf = pd.crosstab(m["archetype"], m["archetype_true"])
    return ari, conf
