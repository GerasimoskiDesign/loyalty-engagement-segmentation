"""Pipeline steps. Each step reads the previous step's files and writes its own, so any
step can be re-run on its own from the CLI."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from loyaltyseg import archetypes as arch
from loyaltyseg import audiences as aud
from loyaltyseg import charts, content, experiment, opportunity, segments
from loyaltyseg.config import Settings
from loyaltyseg.features import build_user_features
from loyaltyseg.generate import generate
from loyaltyseg.io import (
    iter_event_parts,
    read_json,
    read_parquet,
    write_csv,
    write_json,
    write_parquet,
)

log = logging.getLogger(__name__)
README_FIGURES = ("lifecycle_opportunity.png", "content_k_selection.png", "experiment.png")


def _out(s: Settings, *parts: str) -> Path:
    p = s.output_dir.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _require(path: Path, step: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m loyaltyseg {step}` first")
    return path


def step_generate(s: Settings, n_users: int) -> dict[str, Any]:
    return generate(s.raw_dir, n_users=n_users, seed=s.seed, as_of=s.as_of)


def step_content(s: Settings) -> pd.DataFrame:
    catalog = read_parquet(_require(s.raw_dir / "catalog.parquet", "generate"))
    items, summary, sel = content.cluster_content(catalog, s.pipeline["content"], s.seed)
    write_parquet(items, _out(s, "content", "item_clusters.parquet"))
    write_csv(summary, _out(s, "content", "cluster_summary.csv"))
    write_csv(sel.table(), _out(s, "content", "k_selection.csv"))
    write_json({"best_k": sel.best_k, "elbow_k": sel.elbow_k}, _out(s, "content", "k.json"))
    charts.k_selection_chart(
        sel.table(), sel.best_k, sel.elbow_k, _out(s, "figures", "content_k_selection.png")
    )
    return summary


def step_features(s: Settings) -> pd.DataFrame:
    users = read_parquet(_require(s.raw_dir / "users.parquet", "generate"))
    catalog = read_parquet(s.raw_dir / "catalog.parquet")
    items = read_parquet(_require(s.output_dir / "content" / "item_clusters.parquet", "content"))
    feats = build_user_features(
        users, iter_event_parts(s.raw_dir), catalog, items, s.as_of, s.pipeline["features"]
    )
    write_parquet(feats, _out(s, "features", "user_features.parquet"))
    return feats


def step_archetypes(s: Settings, k: int | None = None) -> dict[str, Any]:
    feats = read_parquet(_require(s.output_dir / "features" / "user_features.parquet", "features"))
    res = arch.fit_archetypes(feats, s.pipeline["archetypes"], s.seed, k=k)
    write_parquet(res["assignments"], _out(s, "archetypes", "assignments.parquet"))
    write_csv(res["profile"].reset_index(), _out(s, "archetypes", "profile.csv"))
    if res["k_table"] is not None:
        write_csv(res["k_table"], _out(s, "archetypes", "k_selection.csv"))
        charts.archetype_k_chart(
            res["k_table"], res["best_k"], _out(s, "figures", "archetype_k_selection.png")
        )
    charts.archetype_profile_chart(res["profile"], _out(s, "figures", "archetype_profile.png"))

    evaluation: dict[str, Any] = {"best_k_engaged": res["best_k"]}
    truth_path = s.raw_dir / "ground_truth.parquet"
    if truth_path.exists():
        ari, conf = arch.evaluate_against_truth(res["assignments"], read_parquet(truth_path))
        evaluation["ari_vs_planted"] = ari
        write_csv(conf.reset_index(), _out(s, "archetypes", "confusion_vs_planted.csv"))
        log.info("ARI vs planted archetypes: %.3f", ari)
    write_json(evaluation, _out(s, "archetypes", "evaluation.json"))
    return res


def _cluster_names(s: Settings) -> dict[int, str]:
    items = read_parquet(s.output_dir / "content" / "item_clusters.parquet")
    d = items.drop_duplicates("content_cluster")
    return dict(zip(d["content_cluster"].astype(int), d["content_cluster_name"], strict=True))


def step_segments(s: Settings) -> pd.DataFrame:
    feats = read_parquet(_require(s.output_dir / "features" / "user_features.parquet", "features"))
    users = read_parquet(s.raw_dir / "users.parquet")
    assigned = read_parquet(
        _require(s.output_dir / "archetypes" / "assignments.parquet", "archetypes")
    )
    master = segments.build_master(feats, users, assigned, _cluster_names(s), s.segments)
    write_parquet(master, _out(s, "segments", "master.parquet"))

    sizes = opportunity.layer_sizes(master)
    write_csv(sizes, _out(s, "segments", "layer_sizes.csv"))
    for name, t in opportunity.combination_sizes(master).items():
        write_csv(t, _out(s, "segments", f"combo_{name}.csv"))
    write_json(opportunity.browser_highlight(master), _out(s, "segments", "browsers.json"))
    charts.lifecycle_opportunity_chart(sizes, _out(s, "figures", "lifecycle_opportunity.png"))
    return master


def step_audiences(s: Settings, exclusions: Path | None = None) -> dict[str, Any]:
    master = read_parquet(_require(s.output_dir / "segments" / "master.parquet", "segments"))
    users = read_parquet(s.raw_dir / "users.parquet")
    excl = exclusions if exclusions is not None else s.raw_dir / "contact_history.csv"
    manifest = aud.export_audiences(
        master,
        users,
        s.audiences,
        s.output_dir / "audiences",
        s.as_of,
        s.seed,
        s.config_hash(),
        excl,
    )
    charts.audience_chart(manifest, _out(s, "figures", "audiences.png"))
    return manifest


def step_experiment(s: Settings) -> dict[str, Any]:
    master = read_parquet(_require(s.output_dir / "segments" / "master.parquet", "segments"))
    feats = read_parquet(s.output_dir / "features" / "user_features.parquet")
    truth = read_parquet(s.raw_dir / "ground_truth.parquet")
    catalog = read_parquet(s.raw_dir / "catalog.parquet")
    items = read_parquet(s.output_dir / "content" / "item_clusters.parquet")
    cfg = s.pipeline.get("experiment", {})
    sim = experiment.simulate(master, feats, truth, catalog, items, cfg)
    ro = experiment.readout(sim, cfg)
    write_parquet(sim, _out(s, "experiment", "simulated_outcomes.parquet"))
    write_csv(ro.pop("by_interest"), _out(s, "experiment", "by_interest.csv"))
    write_json(ro, _out(s, "experiment", "readout.json"))
    charts.experiment_chart(ro, _out(s, "figures", "experiment.png"))
    return ro


def step_report(s: Settings, docs_img: Path | None = None) -> Path:
    from loyaltyseg.report import build_report

    pdf = build_report(s)
    if docs_img is not None:
        docs_img.mkdir(parents=True, exist_ok=True)
        for name in README_FIGURES:
            shutil.copyfile(s.output_dir / "figures" / name, docs_img / name)
        log.info("copied README figures to %s", docs_img)
    return pdf


def run_all(
    s: Settings, n_users: int, docs_img: Path | None = None, k: int | None = None
) -> dict[str, Any]:
    meta = step_generate(s, n_users)
    step_content(s)
    step_features(s)
    step_archetypes(s, k=k)
    step_segments(s)
    manifest = step_audiences(s)
    ro = step_experiment(s)
    pdf = step_report(s, docs_img)
    return {
        "meta": meta,
        "audiences": manifest,
        "experiment": ro,
        "report": str(pdf),
        "evaluation": read_json(s.output_dir / "archetypes" / "evaluation.json"),
    }
