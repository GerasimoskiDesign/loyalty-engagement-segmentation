"""Broad vs targeted campaign: simulation + readout (``python -m loyaltyseg experiment``).

Design. All push-reachable users are randomly split into two arms:

* **A - broad**: everyone gets the same generic "discover the loyalty hub" push.
* **B - targeted**: each user gets the message for their *interest* segment (content
  cluster); users with interest ``none`` (Browsers) get an onboarding message; ``mixed``
  users get the generic message.

Outcomes are drawn from the generator's latent model (``latent.py``): the simulator knows
each user's planted archetype, how likely that archetype is to open a push and engage, and
how relevant a given content cluster is to it. The *analysis* only sees observed outcomes
and observed segments, exactly as it would with a real campaign.

Important: the result therefore reflects the generator's parameters. It demonstrates the
analysis framework (randomisation, metrics, z-tests, CIs, power); it is not evidence that
targeting works in general.

Metrics:

* **Acquisition**: share of users with no loyalty activity in the 30 days before the send
  who perform a loyalty action within the attribution window ("new unique loyalty users").
* **Retention**: share of responders who engage again within 30 days.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from loyaltyseg import latent
from loyaltyseg.stats import (
    min_sample_size_per_arm,
    minimum_detectable_effect,
    two_proportion_ztest,
)

log = logging.getLogger(__name__)


def _param(archetypes: pd.Series, attr: str) -> np.ndarray:
    values = {a: getattr(p, attr) for a, p in latent.PARAMS.items()}
    return archetypes.map(values).to_numpy(dtype=float)


def simulate(
    master: pd.DataFrame,
    features: pd.DataFrame,
    truth: pd.DataFrame,
    catalog: pd.DataFrame,
    item_clusters: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Return one row per reachable user with arm, message and simulated outcomes."""
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    pop = (
        master[master["reachable"]]
        .merge(
            features[["user_id", "days_since_last_loyalty"]], on="user_id", validate="one_to_one"
        )
        .merge(
            truth.rename(columns={"archetype": "true_archetype"}),
            on="user_id",
            validate="one_to_one",
        )
        .sort_values("user_id", ignore_index=True)
    )

    pop["arm"] = np.where(
        rng.random(len(pop)) < float(cfg.get("arm_split", 0.5)), "targeted", "broad"
    )
    cluster_ids = item_clusters.set_index("item_id").loc[catalog["item_id"], "content_cluster"]
    rel = latent.archetype_cluster_relevance(catalog, cluster_ids.reset_index(drop=True))
    name_of = item_clusters.drop_duplicates("content_cluster").set_index("content_cluster")[
        "content_cluster_name"
    ]
    rel.index = rel.index.map(name_of)

    targeted = pop["arm"].eq("targeted")
    cluster_msg = targeted & ~pop["interest"].isin(["none", "mixed"])
    onboarding = targeted & pop["interest"].eq("none")
    pop["message"] = np.select(
        [cluster_msg, onboarding], [pop["interest"], "onboarding"], default="generic"
    )

    relevance = np.full(len(pop), latent.GENERIC_RELEVANCE)
    idx = np.flatnonzero(cluster_msg.to_numpy())
    if len(idx):
        r = rel.to_numpy()
        rows = rel.index.get_indexer(pop.loc[idx, "interest"])
        cols = rel.columns.get_indexer(pop.loc[idx, "true_archetype"])
        relevance[idx] = r[rows, cols]
    mult = np.where(onboarding, _param(pop["true_archetype"], "onboarding_lift"), 1.0)

    a = pop["true_archetype"]
    p_resp = (
        _param(a, "p_push_open")
        * _param(a, "p_engage_after_open")
        * (1 + latent.MATCH_LIFT * relevance)
        * mult
    )
    p_resp = np.clip(p_resp, 0, 1)
    pop["responded"] = rng.random(len(pop)) < p_resp
    p_rep = np.clip(_param(a, "p_repeat_30d") * (1 + latent.RETENTION_MATCH_LIFT * relevance), 0, 1)
    pop["repeat_30d"] = pop["responded"] & (rng.random(len(pop)) < p_rep)
    recent = pop["days_since_last_loyalty"].fillna(np.inf) < 30
    pop["acq_eligible"] = ~recent
    pop["acquired"] = pop["acq_eligible"] & pop["responded"]
    return pop.drop(columns=["true_archetype"])


def readout(sim: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """z-tests + CIs for both metrics, a per-interest breakdown and power planning."""
    alpha = float(cfg.get("alpha", 0.05))
    power = float(cfg.get("power", 0.80))
    a = sim[sim["arm"] == "broad"]
    b = sim[sim["arm"] == "targeted"]

    acq = two_proportion_ztest(
        int(a["acquired"].sum()),
        int(a["acq_eligible"].sum()),
        int(b["acquired"].sum()),
        int(b["acq_eligible"].sum()),
        alpha,
    )
    ret = two_proportion_ztest(
        int(a["repeat_30d"].sum()),
        int(a["responded"].sum()),
        int(b["repeat_30d"].sum()),
        int(b["responded"].sum()),
        alpha,
    )

    by_interest = sim.groupby(["interest", "arm"])["responded"].agg(["sum", "count"]).unstack("arm")
    by_interest.columns = [f"{m}_{arm}" for m, arm in by_interest.columns]
    by_interest = by_interest.fillna(0).astype(int).reset_index()
    by_interest["rate_broad"] = by_interest["sum_broad"] / by_interest["count_broad"].clip(1)
    by_interest["rate_targeted"] = by_interest["sum_targeted"] / by_interest["count_targeted"].clip(
        1
    )

    base = acq.p_a
    planning = {
        "baseline_acquisition_rate": base,
        "relative_lift_to_detect": 0.10,
        "required_n_per_arm": (
            min_sample_size_per_arm(base, base * 1.10, alpha, power)
            if 0 < base < 1 / 1.10
            else None
        ),
        "current_n_per_arm": int(min(acq.n_a, acq.n_b)),
        "mde_absolute_at_current_n": (
            minimum_detectable_effect(base, int(min(acq.n_a, acq.n_b)), alpha, power)
            if 0 < base < 1
            else None
        ),
        "alpha": alpha,
        "power": power,
    }
    out = {
        "arms": {"broad": int(len(a)), "targeted": int(len(b))},
        "acquisition": acq.to_dict(),
        "retention": ret.to_dict(),
        "planning": planning,
        "by_interest": by_interest,
        "disclaimer": (
            "Simulated outcome driven by the generator's latent parameters. It "
            "demonstrates the analysis framework and is not evidence on its own."
        ),
    }
    log.info(
        "acquisition broad %.4f vs targeted %.4f (p=%.3g); retention %.3f vs %.3f (p=%.3g)",
        acq.p_a,
        acq.p_b,
        acq.p_value,
        ret.p_a,
        ret.p_b,
        ret.p_value,
    )
    return out
