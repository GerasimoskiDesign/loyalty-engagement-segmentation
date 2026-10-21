"""Seeded synthetic data generator (``python -m loyaltyseg generate``).

Produces, under ``data/raw``:

* ``users.parquet``         one row per user (observable attributes only)
* ``catalog.parquet``       ~150 loyalty items (games, partner offers, raffles)
* ``events/part-*.parquet`` behavioural events, partitioned by user block
* ``ground_truth.parquet``  the planted archetype per user -- evaluation only, never a feature
* ``contact_history.csv``   which users were pushed recently (used as an exclusion list)
* ``meta.json``             seed, size, as-of date and row counts

Determinism: one ``SeedSequence`` is created from ``--seed`` and spawned into child streams
(index 0 = users/catalog, index 1 = contact history, index 2+i = event chunk i). Child
streams are keyed by index, so chunk i always gets the same random numbers for a given seed
and the output is bit-identical between runs. See ``data/GENERATOR.md`` for the assumptions.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from loyaltyseg import latent
from loyaltyseg.io import EVENTS_SUBDIR, write_csv, write_json, write_parquet

log = logging.getLogger(__name__)

EVENT_TYPES: tuple[str, ...] = (
    "app_open",
    "loyalty_tab_view",
    "game_play",
    "offer_unlock",
    "offer_redeem",
    "raffle_entry",
    "push_received",
    "push_opened",
)
LOYALTY_EVENTS: frozenset[str] = frozenset(
    {"loyalty_tab_view", "game_play", "offer_unlock", "offer_redeem", "raffle_entry"}
)
EV = {name: i for i, name in enumerate(EVENT_TYPES)}
NO_ITEM = -1  # item_id sentinel for events that are not about a catalog item
CHUNK_USERS = 50_000
HISTORY_DAYS = 365
PHONE_PREFIX = "+999"  # fake country code: synthetic numbers can never be real ones

OFFER_CATEGORIES = ("tech", "shopping", "food", "travel", "entertainment", "fitness")
OFFER_VALUE_RANGE = {
    "tech": (20, 150),
    "shopping": (10, 60),
    "food": (3, 20),
    "travel": (30, 200),
    "entertainment": (5, 30),
    "fitness": (5, 40),
}
GAME_KINDS = {
    # kind: (frequency, effort_level, reward value range)
    "spin wheel": ("daily", 1, (0.5, 3.0)),
    "scratch card": ("daily", 1, (0.5, 4.0)),
    "quiz": ("weekly", 2, (1.0, 6.0)),
    "streak mission": ("daily", 3, (2.0, 10.0)),
}
RAFFLE_PRIZES = {"tech": 10, "travel": 7, "entertainment": 7}


# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------
def generate_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Build the loyalty catalog: 24 games, 102 partner offers, 24 raffles."""
    rows: list[dict] = []

    for kind, (freq, effort, (lo, hi)) in GAME_KINDS.items():
        for i in range(6):
            rows.append(
                dict(
                    item_name=f"{kind.title()} {i + 1}",
                    item_type="game",
                    item_subtype=kind,
                    category="games",
                    reward_value=round(float(rng.uniform(lo, hi)), 2),
                    effort_level=effort,
                    frequency=freq,
                    discount_depth=0.0,
                    partner_tier="none",
                )
            )

    for cat in OFFER_CATEGORIES:
        lo, hi = OFFER_VALUE_RANGE[cat]
        values = np.exp(rng.uniform(np.log(lo), np.log(hi), size=17))
        tiers = np.where(
            values >= np.quantile(values, 0.7),
            "gold",
            np.where(values >= np.quantile(values, 0.35), "silver", "bronze"),
        )
        for i in range(17):
            weekly_p = 0.6 if cat == "food" else 0.3
            rows.append(
                dict(
                    item_name=f"{cat.title()} partner offer {i + 1}",
                    item_type="offer",
                    item_subtype="voucher",
                    category=cat,
                    reward_value=round(float(values[i]), 2),
                    effort_level=int(rng.integers(1, 3)),
                    frequency="weekly" if rng.random() < weekly_p else "one-off",
                    discount_depth=round(float(0.05 + 0.70 * rng.beta(2.0, 3.0)), 3),
                    partner_tier=str(tiers[i]),
                )
            )

    for cat, n in RAFFLE_PRIZES.items():
        for i in range(n):
            rows.append(
                dict(
                    item_name=f"{cat.title()} prize raffle {i + 1}",
                    item_type="raffle",
                    item_subtype="raffle",
                    category=cat,
                    reward_value=round(float(np.exp(rng.uniform(np.log(100), np.log(1000)))), 2),
                    effort_level=1,
                    frequency="weekly" if rng.random() < 0.5 else "one-off",
                    discount_depth=0.0,
                    partner_tier="gold" if cat == "tech" else "silver",
                )
            )

    cat_df = pd.DataFrame(rows)
    cat_df.insert(0, "item_id", np.arange(1, len(cat_df) + 1, dtype=np.int16))
    return cat_df


# --------------------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------------------
def generate_users(
    n_users: int, as_of: date, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (users, archetype index per user, tenure in days per user)."""
    shares = np.array([latent.PARAMS[a].share for a in latent.ARCHETYPES])
    arche = rng.choice(len(latent.ARCHETYPES), size=n_users, p=shares / shares.sum())

    is_new = rng.random(n_users) < latent.NEW_USER_SHARE
    tenure = np.where(is_new, rng.integers(0, 30, n_users), rng.integers(30, 1461, n_users))
    dormant = arche == latent.ARCHETYPES.index("Dormant")
    # Dormant users must have had time to engage and then lapse for 90+ days.
    tenure = np.where(dormant, rng.integers(180, 1461, n_users), tenure)

    p_push = np.array([latent.P_PUSH_ENABLED[a] for a in latent.ARCHETYPES])[arche]
    user_id = np.arange(1, n_users + 1, dtype=np.int64)
    as_of_ts = pd.Timestamp(as_of)
    users = pd.DataFrame(
        {
            "user_id": user_id,
            "msisdn": [f"{PHONE_PREFIX}{700_000_000 + int(u):09d}" for u in user_id],
            "signup_date": (as_of_ts - pd.to_timedelta(tenure, unit="D")).date,
            "os": np.where(rng.random(n_users) < 0.62, "android", "ios"),
            "push_enabled": rng.random(n_users) < p_push,
            "in_country": rng.random(n_users) < latent.P_IN_COUNTRY,
            "preferred_channel": rng.choice(
                ["push", "email", "sms", "in_app"], size=n_users, p=[0.45, 0.25, 0.10, 0.20]
            ),
        }
    )
    return users, arche, tenure


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------
def _uniform_ts(
    rng: np.random.Generator, end_ts: int, lo_ago: np.ndarray, hi_ago: np.ndarray
) -> np.ndarray:
    """Timestamps (epoch seconds) uniformly between ``hi_ago`` and ``lo_ago`` days ago."""
    ago = lo_ago + rng.random(len(lo_ago)) * (hi_ago - lo_ago)
    return end_ts - (ago * 86_400).astype(np.int64)


def _chunk_events(
    user_ids: np.ndarray,
    arche: np.ndarray,
    tenure: np.ndarray,
    push_ok: np.ndarray,
    catalog: pd.DataFrame,
    affinity: np.ndarray,
    end_ts: int,
    rng: np.random.Generator,
) -> pa.Table:
    """Simulate one block of users. Fully vectorised: ~O(events) numpy work."""
    n = len(user_ids)
    P = latent.PARAMS
    names = latent.ARCHETYPES
    get = lambda attr: np.array([getattr(P[a], attr) for a in names])[arche]  # noqa: E731

    start_ago = np.minimum(tenure + 1.0, HISTORY_DAYS).astype(float)
    end_ago = np.zeros(n)
    browser = arche == names.index("Browsers")
    dormant = arche == names.index("Dormant")
    engaged = ~(browser | dormant)

    at_risk = engaged & (rng.random(n) < latent.AT_RISK_SHARE)
    risk_end = rng.uniform(31, 89, n)
    at_risk &= start_ago > risk_end + 5
    end_ago = np.where(at_risk, risk_end, end_ago)
    dorm_hi = np.maximum(np.minimum(330.0, start_ago - 10.0), 96.0)
    end_ago = np.where(dormant, 95.0 + rng.random(n) * (dorm_hi - 95.0), end_ago)

    # Gamma(2, 0.5) has mean 1: per-user heterogeneity in how active they are.
    g_loy = rng.gamma(2.0, 0.5, n)
    g_app = rng.gamma(2.0, 0.5, n)
    window = np.maximum(start_ago - end_ago, 0.0)
    n_sess = np.where(
        browser, 0, 1 + rng.poisson(get("loyalty_sessions_30d") * g_loy * window / 30.0)
    )
    n_app = rng.poisson(get("app_opens_30d") * g_app * start_ago / 30.0)
    n_app = np.where(browser, n_app + 1, n_app)  # Browsers open the app by definition
    n_push = np.where(push_ok, rng.poisson(latent.PUSH_PER_30D * start_ago / 30.0), 0)

    cols_u, cols_t, cols_e, cols_i = [], [], [], []

    def add(u: np.ndarray, t: np.ndarray, e: int | np.ndarray, i: np.ndarray | int) -> None:
        cols_u.append(u)
        # Follow-up events (play after view, redeem after unlock) are offset forward in
        # time; clip so nothing lands after the as-of day.
        cols_t.append(np.minimum(t, end_ts))
        cols_e.append(np.broadcast_to(np.asarray(e, dtype=np.int8), u.shape).copy())
        cols_i.append(np.broadcast_to(np.asarray(i, dtype=np.int16), u.shape).copy())

    # Loyalty sessions: app_open -> loyalty_tab_view -> item-specific action(s).
    s_idx = np.repeat(np.arange(n), n_sess)
    s_ts = _uniform_ts(rng, end_ts, end_ago[s_idx], start_ago[s_idx])
    s_item = np.empty(len(s_idx), dtype=np.int64)
    n_items = len(catalog)
    for a in np.unique(arche[s_idx]):
        m = arche[s_idx] == a
        s_item[m] = rng.choice(n_items, size=int(m.sum()), p=affinity[:, a])
    noise = rng.random(len(s_idx)) < latent.NOISE_SHARE
    s_item[noise] = rng.integers(0, n_items, int(noise.sum()))
    item_ids = catalog["item_id"].to_numpy()[s_item]
    item_type = catalog["item_type"].to_numpy()[s_item]
    s_user = user_ids[s_idx]

    add(s_user, s_ts, EV["app_open"], NO_ITEM)
    view_ts = s_ts + rng.integers(5, 60, len(s_idx))
    add(s_user, view_ts, EV["loyalty_tab_view"], item_ids)

    g = item_type == "game"
    plays = 1 + rng.poisson(0.3, int(g.sum()))
    gp_rep = np.repeat(np.flatnonzero(g), plays)
    add(
        s_user[gp_rep],
        view_ts[gp_rep] + rng.integers(30, 900, len(gp_rep)),
        EV["game_play"],
        item_ids[gp_rep],
    )

    r = item_type == "raffle"
    add(
        s_user[r], view_ts[r] + rng.integers(30, 300, int(r.sum())), EV["raffle_entry"], item_ids[r]
    )

    o = np.flatnonzero(item_type == "offer")
    unlock = o[rng.random(len(o)) < get("p_unlock")[s_idx[o]]]
    unlock_ts = view_ts[unlock] + rng.integers(30, 900, len(unlock))
    add(s_user[unlock], unlock_ts, EV["offer_unlock"], item_ids[unlock])
    red_mask = rng.random(len(unlock)) < get("p_redeem")[s_idx[unlock]]
    red = unlock[red_mask]
    red_ts = unlock_ts[red_mask] + rng.integers(600, 2 * 86_400, len(red))
    add(s_user[red], red_ts, EV["offer_redeem"], item_ids[red])

    # Standalone app opens that never reach the loyalty hub.
    a_idx = np.repeat(np.arange(n), n_app)
    add(
        user_ids[a_idx],
        _uniform_ts(rng, end_ts, np.zeros(len(a_idx)), start_ago[a_idx]),
        EV["app_open"],
        NO_ITEM,
    )

    # Push campaigns.
    p_idx = np.repeat(np.arange(n), n_push)
    p_ts = _uniform_ts(rng, end_ts, np.zeros(len(p_idx)), start_ago[p_idx])
    add(user_ids[p_idx], p_ts, EV["push_received"], NO_ITEM)
    opened = rng.random(len(p_idx)) < get("p_push_open")[p_idx]
    add(
        user_ids[p_idx][opened],
        p_ts[opened] + rng.integers(30, 3_600, int(opened.sum())),
        EV["push_opened"],
        NO_ITEM,
    )

    u = np.concatenate(cols_u).astype(np.int64)
    t = np.concatenate(cols_t).astype(np.int64)
    e = np.concatenate(cols_e)
    i = np.concatenate(cols_i)
    order = np.lexsort((i, e, t, u))
    return pa.table(
        {
            "user_id": pa.array(u[order]),
            "ts": pa.array(t[order], type=pa.int64()).cast(pa.timestamp("s")),
            "event_type": pa.DictionaryArray.from_arrays(
                pa.array(e[order], type=pa.int8()), pa.array(list(EVENT_TYPES))
            ),
            "item_id": pa.array(i[order], type=pa.int16()),
        }
    )


def _contact_history(users: pd.DataFrame, as_of: date, rng: np.random.Generator) -> pd.DataFrame:
    """Recent campaign sends: 10% of push-enabled users were pushed in the last 14 days."""
    pool = users.loc[users["push_enabled"], "user_id"].to_numpy()
    chosen = np.sort(rng.choice(pool, size=int(0.10 * len(pool)), replace=False))
    days_ago = rng.integers(0, 14, len(chosen))
    sent = [as_of - timedelta(days=int(d)) for d in days_ago]
    return pd.DataFrame(
        {
            "user_id": chosen,
            "campaign_id": [f"CMP-{s:%Y%m%d}" for s in sent],
            "sent_date": [s.isoformat() for s in sent],
        }
    )


def generate(
    raw_dir: Path, n_users: int = 200_000, seed: int = 42, as_of: date = date(2026, 8, 31)
) -> dict:
    """Generate the full synthetic dataset into ``raw_dir`` and return its metadata."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = raw_dir / EVENTS_SUBDIR
    ev_dir.mkdir(exist_ok=True)
    for old in ev_dir.glob("part-*.parquet"):
        old.unlink()

    ss = np.random.SeedSequence(seed)
    n_chunks = (n_users + CHUNK_USERS - 1) // CHUNK_USERS
    children = ss.spawn(2 + n_chunks)
    rng0 = np.random.default_rng(children[0])

    catalog = generate_catalog(rng0)
    users, arche, tenure = generate_users(n_users, as_of, rng0)
    affinity = latent.item_affinity(catalog).to_numpy()
    push_ok = (users["push_enabled"] & users["in_country"]).to_numpy()
    # Last second of the as-of day (naive UTC epoch seconds).
    end_ts = int(pd.Timestamp(as_of).timestamp()) + 86_399

    n_events = 0
    uid = users["user_id"].to_numpy()
    for c in range(n_chunks):
        sl = slice(c * CHUNK_USERS, min((c + 1) * CHUNK_USERS, n_users))
        table = _chunk_events(
            uid[sl],
            arche[sl],
            tenure[sl],
            push_ok[sl],
            catalog,
            affinity,
            end_ts,
            np.random.default_rng(children[2 + c]),
        )
        pq.write_table(table, ev_dir / f"part-{c:04d}.parquet", compression="zstd")
        n_events += table.num_rows
        log.info("events chunk %d/%d: %d rows", c + 1, n_chunks, table.num_rows)

    write_parquet(users, raw_dir / "users.parquet")
    write_parquet(catalog, raw_dir / "catalog.parquet")
    truth = pd.DataFrame(
        {"user_id": uid, "archetype": np.array(latent.ARCHETYPES, dtype=object)[arche]}
    )
    write_parquet(truth, raw_dir / "ground_truth.parquet")
    write_csv(
        _contact_history(users, as_of, np.random.default_rng(children[1])),
        raw_dir / "contact_history.csv",
    )

    meta = {
        "seed": seed,
        "users": n_users,
        "as_of": as_of.isoformat(),
        "catalog_items": len(catalog),
        "events": n_events,
        "event_parts": n_chunks,
    }
    write_json(meta, raw_dir / "meta.json")
    log.info("generated %d users, %d items, %d events", n_users, len(catalog), n_events)
    return meta
