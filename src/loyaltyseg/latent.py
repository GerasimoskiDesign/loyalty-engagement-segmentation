"""The generator's hidden (latent) behavioural model.

Everything in this module is *ground truth that real data would never give you*. It is used
in exactly two places:

* ``generate.py`` draws the synthetic users and events from these parameters.
* ``experiment.py`` draws simulated campaign responses from the same parameters, because a
  simulation needs a response model and the honest choice is to reuse the one that produced
  the data.

No modelling step (content clustering, user features, archetypes, segments) may import this
module. Keeping the latent model isolated is what makes the ARI evaluation meaningful: the
pipeline has to rediscover the archetypes from observable behaviour only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ARCHETYPES: tuple[str, ...] = (
    "Deal Hunters",
    "Gamers",
    "Tech Chasers",
    "Shoppers",
    "Occasional Players",
    "Browsers",
    "Dormant",
)


@dataclass(frozen=True)
class ArchetypeParams:
    """Behavioural parameters of one planted archetype.

    Rates are expressed per 30 days of the user's activity window so that new users (short
    window) naturally have fewer events than long-tenured users.
    """

    share: float  # population share
    loyalty_sessions_30d: (
        float  # mean loyalty-hub sessions per 30 days (each starts with an app open)
    )
    app_opens_30d: float  # mean *additional* app opens per 30 days that never reach the hub
    p_unlock: float  # P(unlock | offer viewed)
    p_redeem: float  # P(redeem | offer unlocked)
    p_push_open: float  # P(open | push received)
    p_engage_after_open: float  # P(loyalty action | push opened) -- used by the simulation
    p_repeat_30d: float  # P(another loyalty action within 30 days | responded)
    onboarding_lift: float  # response multiplier for an onboarding-style message


PARAMS: dict[str, ArchetypeParams] = {
    "Deal Hunters": ArchetypeParams(0.09, 1.5, 2.0, 0.75, 0.75, 0.24, 0.45, 0.40, 1.0),
    "Gamers": ArchetypeParams(0.10, 3.5, 3.0, 0.30, 0.30, 0.34, 0.55, 0.60, 1.0),
    "Tech Chasers": ArchetypeParams(0.08, 1.5, 2.0, 0.55, 0.50, 0.20, 0.40, 0.35, 1.0),
    "Shoppers": ArchetypeParams(0.08, 1.5, 2.0, 0.60, 0.60, 0.20, 0.42, 0.35, 1.0),
    "Occasional Players": ArchetypeParams(0.15, 0.4, 1.5, 0.30, 0.30, 0.12, 0.30, 0.15, 1.0),
    "Browsers": ArchetypeParams(0.35, 0.0, 2.0, 0.0, 0.0, 0.06, 0.08, 0.10, 1.8),
    "Dormant": ArchetypeParams(0.15, 1.5, 0.25, 0.40, 0.40, 0.03, 0.10, 0.10, 1.3),
}

# Push campaigns received per 30 days by a push-enabled, in-country user.
PUSH_PER_30D: float = 1.0
# Population-level probabilities for user attributes.
P_IN_COUNTRY: float = 0.93
P_PUSH_ENABLED: dict[str, float] = {
    "Deal Hunters": 0.58,
    "Gamers": 0.60,
    "Tech Chasers": 0.52,
    "Shoppers": 0.52,
    "Occasional Players": 0.48,
    "Browsers": 0.48,
    "Dormant": 0.40,
}
NEW_USER_SHARE: float = 0.04  # signed up in the last 30 days
NOISE_SHARE: float = 0.10  # share of loyalty sessions where the item is picked uniformly

# How strongly a message that matches the user's real interests lifts the response.
# relevance is the archetype's affinity share on the message's content cluster (0..1).
MATCH_LIFT: float = 2.0
# A generic "check out the loyalty hub" message is treated as having this relevance.
GENERIC_RELEVANCE: float = 0.10
# Retention lift for responders whose first touch was a relevant message.
RETENTION_MATCH_LIFT: float = 0.5

# Share of engaged users (not Browsers / Dormant) whose loyalty activity stopped 30-90 days
# before the as-of date. This plants a realistic "at-risk" lifecycle group.
AT_RISK_SHARE: float = 0.12


def _offer_weights(category: pd.Series, by_cat: dict[str, float], default: float) -> np.ndarray:
    return category.map(by_cat).fillna(default).to_numpy(dtype=float)


def item_affinity(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return an items x archetypes matrix of choice probabilities (columns sum to 1).

    The affinities are written in terms of *item attributes* (type, category, sub-type,
    discount depth), never in terms of content clusters. Content clusters are something the
    pipeline learns later; if the generator spoke in clusters, the evaluation would be
    circular.
    """
    t = catalog["item_type"]
    cat = catalog["category"]
    sub = catalog["item_subtype"]
    disc = catalog["discount_depth"].to_numpy(dtype=float)
    is_game = (t == "game").to_numpy()
    is_raffle = (t == "raffle").to_numpy()
    is_offer = (t == "offer").to_numpy()

    w = pd.DataFrame(0.0, index=catalog.index, columns=list(ARCHETYPES))

    # Deal Hunters: any category, but steeply prefer deep discounts.
    deal_cat = {
        "food": 1.0,
        "entertainment": 1.0,
        "fitness": 0.8,
        "shopping": 0.6,
        "travel": 0.4,
        "tech": 0.3,
    }
    w["Deal Hunters"] = np.where(
        is_offer,
        _offer_weights(cat, deal_cat, 0.3) * np.exp(6.0 * disc),
        np.where(is_game, 0.05, 0.10),
    )
    # Gamers: games, with more effortful formats preferred; a little raffle play.
    game_sub = {"streak mission": 1.5, "quiz": 1.3, "scratch card": 0.9, "spin wheel": 0.8}
    w["Gamers"] = np.where(
        is_game,
        sub.map(game_sub).fillna(1.0).to_numpy(dtype=float),
        np.where(is_raffle, 0.25, 0.03),
    )
    # Tech Chasers: tech offers and tech-prize raffles.
    is_tech = (cat == "tech").to_numpy()
    w["Tech Chasers"] = np.where(
        is_offer & is_tech,
        1.0,
        np.where(is_raffle & is_tech, 0.5, np.where(is_game, 0.08, 0.04)),
    )
    # Shoppers: shopping offers.
    is_shop = (cat == "shopping").to_numpy()
    w["Shoppers"] = np.where(is_offer & is_shop, 1.0, np.where(is_game, 0.08, 0.05))
    # Occasional Players: raffles and low-effort instant games, rarely anything else.
    light_game = sub.isin(["spin wheel", "scratch card"]).to_numpy()
    w["Occasional Players"] = np.where(
        is_raffle, 1.0, np.where(light_game, 0.5, np.where(is_game, 0.1, 0.05))
    )
    # Dormant users used to engage with a broad mix before they lapsed.
    w["Dormant"] = 0.3
    # Browsers never touch the loyalty hub: a zero column, kept for shape consistency.
    w["Browsers"] = 0.0

    sums = w.sum(axis=0).replace(0.0, 1.0)
    return w / sums


def archetype_cluster_relevance(catalog: pd.DataFrame, item_cluster: pd.Series) -> pd.DataFrame:
    """Affinity share of each archetype on each content cluster (rows: clusters).

    Used only by the experiment simulation to decide how relevant a cluster-targeted message
    is to a user whose *true* archetype is known to the simulator.
    """
    aff = item_affinity(catalog)
    aff = aff.assign(_cluster=item_cluster.to_numpy())
    return aff.groupby("_cluster").sum()
