"""Matplotlib charts shared by the PDF report and the README.

Neutral styling: one accent hue for the thing being highlighted, grey for context, thin
marks, recessive grid, no dual axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
ORANGE = "#eb6834"
GREY = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _thousands(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return path


def k_selection_chart(
    table: pd.DataFrame,
    best_k: int,
    elbow_k: int,
    path: Path,
    title: str = "Content clusters: choosing k",
) -> Path:
    """Silhouette and elbow side by side (two panels, never two y-axes)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for ax, col, label in (
        (axes[0], "silhouette", "Silhouette (higher is better)"),
        (axes[1], "inertia", "Inertia / elbow (lower is tighter)"),
    ):
        ax.plot(table["k"], table[col], color=GREY, linewidth=2, marker="o", markersize=5)
        mark = best_k if col == "silhouette" else elbow_k
        row = table[table["k"] == mark]
        ax.plot(
            row["k"],
            row[col],
            "o",
            color=BLUE if col == "silhouette" else ORANGE,
            markersize=9,
            zorder=3,
        )
        ax.annotate(
            f"k={mark}",
            (mark, float(row[col].iloc[0])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=INK,
        )
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_xlabel("k", fontsize=8)
        _style(ax)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def archetype_k_chart(table: pd.DataFrame, best_k: int, path: Path) -> Path:
    """Silhouette, Davies-Bouldin and stability as small multiples."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    specs = (
        ("silhouette", "Silhouette (20k sample)"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("stability_ari", "Stability (mean ARI, 5 seeds)"),
    )
    for ax, (col, label) in zip(axes, specs, strict=True):
        ax.plot(table["k"], table[col], color=GREY, linewidth=2, marker="o", markersize=5)
        row = table[table["k"] == best_k]
        ax.plot(row["k"], row[col], "o", color=BLUE, markersize=9, zorder=3)
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_xlabel("k (engaged users)", fontsize=8)
        _style(ax)
    thr = table.attrs.get("stability_min_ari")
    if thr is not None:
        axes[2].axhline(thr, color=ORANGE, linewidth=1, linestyle="--")
    fig.tight_layout()
    return _save(fig, path)


def lifecycle_opportunity_chart(sizes: pd.DataFrame, path: Path) -> Path:
    """Users vs push-reachable users per lifecycle stage, Browsers highlighted."""
    t = sizes[sizes["layer"] == "lifecycle"].sort_values("users")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(t))
    is_b = (t["label"] == "browser").to_numpy()
    ax.barh(
        y, t["users"], color=np.where(is_b, "#f7c2ad", "#d9d8d3"), height=0.6, label="All users"
    )
    ax.barh(
        y,
        t["reachable_users"],
        color=np.where(is_b, ORANGE, GREY),
        height=0.6,
        label="Push-reachable",
    )
    for yi, (u, r) in enumerate(zip(t["users"], t["reachable_users"], strict=True)):
        ax.text(u, yi, f"  {u:,.0f} ({r:,.0f} reachable)", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, t["label"])
    ax.set_xlim(0, t["users"].max() * 1.45)
    ax.set_xlabel("users (synthetic)", fontsize=8)
    _thousands(ax)
    ax.set_title("Lifecycle stage: size and push reachability", fontsize=10, loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def experiment_chart(readout: dict[str, Any], path: Path) -> Path:
    """Difference (targeted - broad) with 95% CI for both metrics, in percentage points."""
    rows = [
        ("Acquisition\n(new loyalty users)", readout["acquisition"]),
        ("Retention\n(30-day repeat)", readout["retention"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.6))
    for ax, (label, r) in zip(axes, rows, strict=True):
        d, lo, hi = 100 * r["diff"], 100 * r["ci_low"], 100 * r["ci_high"]
        ax.errorbar(
            [d],
            [0],
            xerr=[[d - lo], [hi - d]],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            elinewidth=2,
            capsize=5,
            markersize=8,
        )
        ax.axvline(0, color=GREY, linewidth=1)
        ax.set_yticks([])
        span = max(abs(lo), abs(hi)) * 1.4 or 1
        ax.set_xlim(-span * 0.3 if lo > 0 else -span, span)
        ax.set_title(label.replace("\n", " "), fontsize=9, loc="left")
        ax.set_xlabel("targeted minus broad, percentage points", fontsize=8)
        ax.text(
            d,
            0.25,
            f"{d:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]\n"
            f"{100 * r['p_a']:.2f}% -> {100 * r['p_b']:.2f}%,  p={r['p_value']:.2g}",
            ha="center",
            fontsize=8,
            color=INK,
        )
        ax.set_ylim(-0.5, 0.8)
        _style(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Simulated campaign: targeted vs broad (95% CI)", fontsize=11, x=0.01, ha="left", color=INK
    )
    fig.tight_layout()
    return _save(fig, path)


def archetype_profile_chart(profile: pd.DataFrame, path: Path) -> Path:
    """Heatmap of column-normalised profile metrics (single-hue sequential)."""
    cols = [
        "loyalty_days_per_30d",
        "recency_days_filled",
        "type_share_game",
        "type_share_raffle",
        "cat_share_tech",
        "cat_share_shopping",
        "discount_pref",
        "reward_value_redeemed",
        "push_open_rate_filled",
    ]
    p = profile.set_index("archetype")[cols].astype(float)
    norm = (p - p.min()) / (p.max() - p.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(p) + 1.4))
    ax.imshow(norm.to_numpy(), cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(
        range(len(cols)),
        [c.replace("_filled", "").replace("_", " ") for c in cols],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(range(len(p)), p.index, fontsize=8)
    for i in range(len(p)):
        for j in range(len(cols)):
            v = p.iat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}" if v < 10 else f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if norm.iat[i, j] > 0.55 else INK,
            )
    ax.set_title(
        "Archetype profiles (cell colour = relative level within column)", fontsize=9, loc="left"
    )
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def audience_chart(manifest: dict[str, Any], path: Path) -> Path:
    """Matched vs final rows per audience."""
    names = [a["audience"] for a in manifest["audiences"]]
    matched = [a["counts"]["matched"] for a in manifest["audiences"]]
    final = [a["rows_written"] for a in manifest["audiences"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, matched, color="#d9d8d3", height=0.6, label="Matched segment")
    ax.barh(y, final, color=BLUE, height=0.6, label="Exported (after all filters)")
    for yi, (m, f) in enumerate(zip(matched, final, strict=True)):
        ax.text(m, yi, f"  {f:,} of {m:,}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(matched + [1]) * 1.35)
    _thousands(ax)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Campaign audiences: matched vs exported", fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, path)
