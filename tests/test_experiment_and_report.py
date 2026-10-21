from __future__ import annotations

import pandas as pd

from loyaltyseg.config import Settings
from loyaltyseg.io import read_json, read_parquet
from loyaltyseg.stats import two_proportion_ztest


def test_simulation_outcomes_are_consistent(run: Settings, master: pd.DataFrame) -> None:
    sim = read_parquet(run.output_dir / "experiment" / "simulated_outcomes.parquet")
    assert set(sim["user_id"]) == set(master.loc[master["reachable"], "user_id"])
    assert set(sim["arm"]) == {"broad", "targeted"}
    share_targeted = (sim["arm"] == "targeted").mean()
    assert 0.45 < share_targeted < 0.55
    assert not (sim["acquired"] & ~sim["responded"]).any()
    assert not (sim["repeat_30d"] & ~sim["responded"]).any()
    assert not (sim["acquired"] & ~sim["acq_eligible"]).any()
    assert (sim.loc[sim["arm"] == "broad", "message"] == "generic").all()
    onboarding = sim["message"] == "onboarding"
    assert (sim.loc[onboarding, "interest"] == "none").all()


def test_readout_matches_recomputation(run: Settings) -> None:
    sim = read_parquet(run.output_dir / "experiment" / "simulated_outcomes.parquet")
    ro = read_json(run.output_dir / "experiment" / "readout.json")
    a, b = sim[sim["arm"] == "broad"], sim[sim["arm"] == "targeted"]
    r = two_proportion_ztest(
        int(a["acquired"].sum()),
        int(a["acq_eligible"].sum()),
        int(b["acquired"].sum()),
        int(b["acq_eligible"].sum()),
    )
    assert ro["acquisition"]["p_value"] == r.p_value
    assert "not evidence" in ro["disclaimer"]
    assert ro["planning"]["required_n_per_arm"] > 0


def test_report_and_figures_exist(run: Settings) -> None:
    pdf = run.output_dir / "report" / "loyalty_segmentation_report.pdf"
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF")
    assert (
        4 <= data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") <= 6
        or 4 <= data.count(b"/Type /Page") - data.count(b"/Type /Pages") <= 6
    )
    for name in ("content_k_selection.png", "lifecycle_opportunity.png", "experiment.png"):
        assert (run.output_dir / "figures" / name).stat().st_size > 10_000
