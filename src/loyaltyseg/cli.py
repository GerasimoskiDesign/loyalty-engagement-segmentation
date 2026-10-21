"""Command line interface: ``python -m loyaltyseg <command> [options]``."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from loyaltyseg import pipeline
from loyaltyseg.config import DEFAULT_AS_OF, DEFAULT_SEED, load_settings, project_root

log = logging.getLogger("loyaltyseg")
COMMANDS = (
    "generate",
    "content",
    "features",
    "archetypes",
    "segments",
    "audiences",
    "experiment",
    "report",
    "all",
)


def _date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loyaltyseg", description=__doc__)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--users", type=int, default=200_000, help="users to generate")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--as-of",
        type=_date,
        default=DEFAULT_AS_OF,
        help="reference date YYYY-MM-DD (end of the 12-month window)",
    )
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="force the number of engaged archetype clusters (skip k scan)",
    )
    p.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        help="contact-history CSV (user_id, sent_date); default data/raw/...",
    )
    p.add_argument(
        "--docs-img", action="store_true", help="also copy the README charts into docs/img"
    )
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    level = args.log_level or os.environ.get("LOYALTYSEG_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    s = load_settings(args.config_dir, args.data_dir, args.output_dir, args.seed, args.as_of)
    docs_img = project_root() / "docs" / "img" if args.docs_img else None
    log.info(
        "command=%s seed=%d as_of=%s data=%s output=%s",
        args.command,
        s.seed,
        s.as_of,
        s.data_dir,
        s.output_dir,
    )

    cmd = args.command
    if cmd == "generate":
        pipeline.step_generate(s, args.users)
    elif cmd == "content":
        pipeline.step_content(s)
    elif cmd == "features":
        pipeline.step_features(s)
    elif cmd == "archetypes":
        pipeline.step_archetypes(s, k=args.k)
    elif cmd == "segments":
        pipeline.step_segments(s)
    elif cmd == "audiences":
        pipeline.step_audiences(s, args.exclusions)
    elif cmd == "experiment":
        pipeline.step_experiment(s)
    elif cmd == "report":
        pipeline.step_report(s, docs_img)
    else:
        pipeline.run_all(s, args.users, docs_img, k=args.k)
    log.info("done: %s", cmd)
    return 0
