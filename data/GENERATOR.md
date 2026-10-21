# Synthetic data generator: assumptions

`python -m loyaltyseg generate --users 200000 --seed 42 --as-of 2026-08-31`

Everything in this folder except this file is generated and git-ignored. The generator is
seeded and deterministic: the same `--users`, `--seed` and `--as-of` always produce
identical tables (checked by `tests/test_generator.py`). All parameters below live in
`src/loyaltyseg/latent.py` and `src/loyaltyseg/generate.py`.

## Tables (`data/raw/`)

| File | Grain | Columns |
|---|---|---|
| `users.parquet` | user | user_id, msisdn (fake `+999` prefix), signup_date, os, push_enabled, in_country, preferred_channel |
| `catalog.parquet` | item (150) | item_id, item_name, item_type, item_subtype, category, reward_value, effort_level, frequency, discount_depth, partner_tier |
| `events/part-NNNN.parquet` | event | user_id, ts (second precision), event_type, item_id (`-1` = not about an item) |
| `ground_truth.parquet` | user | user_id, archetype (planted; **evaluation only**) |
| `contact_history.csv` | send | user_id, campaign_id, sent_date: recent pushes, used as the audience exclusion list |
| `meta.json` | run | seed, users, as_of, row counts |

Events are partitioned in blocks of 50,000 users, so every user's events sit in exactly one
part file. Downstream aggregation works part by part, which keeps memory flat at 800k users.

## Catalog (150 items)

* **24 games**: spin wheel, scratch card (daily, low effort), quiz (weekly, medium effort),
  streak mission (daily, high effort), 6 of each. Small reward values.
* **102 partner offers**: 17 each in tech, shopping, food, travel, entertainment and
  fitness. Voucher value is log-uniform within a category range (tech and travel high, food
  low); discount depth ~ 0.05 + 0.70·Beta(2,3); partner tier from value tertiles.
* **24 raffles** with tech, travel or entertainment prizes, high prize values.

## Planted archetypes

| Archetype | Share | Loyalty sessions / 30d | Extra app opens / 30d | What they choose |
|---|---|---|---|---|
| Deal Hunters | 9% | 1.5 | 2.0 | offers in any category, weighted by `exp(6·discount_depth)` |
| Gamers | 10% | 3.5 | 3.0 | games (streak missions and quizzes preferred), some raffles |
| Tech Chasers | 8% | 1.5 | 2.0 | tech offers, tech raffles |
| Shoppers | 8% | 1.5 | 2.0 | shopping offers |
| Occasional Players | 15% | 0.4 | 1.5 | raffles and instant games |
| Browsers | 35% | 0 | 2.0 | nothing: they open the app but never touch the loyalty hub |
| Dormant | 15% | 1.5 (only in the past) | 0.25 | a broad mix, but their activity stopped 95 to 330 days before the as-of date |

Other mechanics:

* **Heterogeneity**: each user's rates are multiplied by a Gamma(2, 0.5) factor (mean 1).
* **Observation window**: up to 12 months, shorter for users who signed up later.
  4% of users signed up in the last 30 days. Dormant users have at least 180 days of tenure.
* **Noise**: 10% of loyalty sessions pick an item uniformly at random, so no archetype is
  perfectly pure.
* **At-risk users**: 12% of engaged users (not Browsers or Dormant) stopped loyalty activity
  31 to 89 days before the as-of date. They keep opening the app.
* **Sessions**: each loyalty session is `app_open` → `loyalty_tab_view` → an item-specific
  action. Games give 1 + Poisson(0.3) `game_play` events and raffles give one
  `raffle_entry`. Offers give `offer_unlock` with P(unlock) and then `offer_redeem` with
  P(redeem | unlock), up to 2 days later. Both probabilities depend on the archetype.
* **Push**: push-enabled, in-country users receive about one campaign push per 30 days.
  P(open) depends on the archetype (Gamers 0.34, Browsers 0.06, Dormant 0.03).
* **Reachability**: push enabled (40-60% by archetype) AND in country (93%) AND app opened
  in the last 30 days. This comes out at roughly one third of users.
* **Contact history**: 10% of push-enabled users received a campaign in the 14 days before
  the as-of date.

## Response model used only by the experiment simulation

`latent.py` also holds P(engage | push opened), P(repeat within 30 days | responded), a
relevance multiplier for messages that match the user's planted interests
(`1 + 2.0 · relevance`, where relevance is the archetype's affinity share on the message's
content cluster) and an onboarding lift for Browsers (1.8×) and Dormant users (1.3×).
These numbers are assumptions. **The experiment readout reflects them by construction.**

## What the generator does not model

Seasonality, day-of-week effects, campaign fatigue, partner stock-outs, device changes,
churned accounts, and archetype drift over time. A real feed would also contain duplicate
and late events. The pipeline handles duplicates in audiences but does not clean events.
