from __future__ import annotations

import pytest
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from loyaltyseg.stats import (
    min_sample_size_per_arm,
    minimum_detectable_effect,
    two_proportion_ztest,
)

CASES = [(78, 2009, 116, 2110), (50, 1000, 50, 1000), (300, 5000, 260, 5000), (1, 40, 9, 45)]


@pytest.mark.parametrize(("xa", "na", "xb", "nb"), CASES)
def test_ztest_matches_statsmodels(xa: int, na: int, xb: int, nb: int) -> None:
    r = two_proportion_ztest(xa, na, xb, nb)
    z, p = proportions_ztest([xb, xa], [nb, na])
    assert r.z == pytest.approx(z, rel=1e-9, abs=1e-12)
    assert r.p_value == pytest.approx(p, rel=1e-9)
    lo, hi = confint_proportions_2indep(xb, nb, xa, na, method="wald", compare="diff")
    assert r.ci_low == pytest.approx(lo, rel=1e-9)
    assert r.ci_high == pytest.approx(hi, rel=1e-9)


def test_ztest_input_validation() -> None:
    with pytest.raises(ValueError):
        two_proportion_ztest(1, 0, 1, 10)
    with pytest.raises(ValueError):
        two_proportion_ztest(11, 10, 1, 10)


def test_power_helper_known_value() -> None:
    # Textbook value (Fleiss, no continuity correction): 10% vs 15%, alpha .05, power .8
    assert min_sample_size_per_arm(0.10, 0.15) == 686
    # Symmetric in the two proportions and larger for smaller effects.
    assert min_sample_size_per_arm(0.15, 0.10) == 686
    assert min_sample_size_per_arm(0.10, 0.12) > 686


def test_mde_is_inverse_of_sample_size() -> None:
    mde = minimum_detectable_effect(0.10, 686)
    assert mde == pytest.approx(0.05, abs=5e-4)
    with pytest.raises(ValueError):
        min_sample_size_per_arm(0.1, 0.1)
