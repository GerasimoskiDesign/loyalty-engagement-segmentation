"""Small, dependency-light statistics helpers for campaign tests.

Implemented with scipy only (statsmodels is used in the tests as the reference) so that the
formulas are visible and auditable in one screen.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class ProportionTest:
    """Result of a two-proportion comparison (treatment = arm B, control = arm A)."""

    x_a: int
    n_a: int
    x_b: int
    n_b: int
    p_a: float
    p_b: float
    diff: float  # p_b - p_a
    relative_lift: float  # diff / p_a
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def two_proportion_ztest(
    x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05
) -> ProportionTest:
    """Two-sided two-proportion z-test.

    * The test statistic uses the **pooled** proportion: under H0 both arms share one rate,
      so the pooled estimate is the right variance under the null (same as
      ``statsmodels.stats.proportion.proportions_ztest``).
    * The confidence interval for the difference uses the **unpooled** (Wald) standard
      error, because the interval describes the difference *without* assuming H0.
    """
    if min(n_a, n_b) <= 0:
        raise ValueError("both arms need at least one observation")
    if not (0 <= x_a <= n_a and 0 <= x_b <= n_b):
        raise ValueError("successes must be between 0 and n")
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    diff = p_b - p_a
    z = diff / se0 if se0 > 0 else 0.0
    p_value = float(2 * norm.sf(abs(z)))
    se1 = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = float(norm.ppf(1 - alpha / 2))
    return ProportionTest(
        x_a=int(x_a),
        n_a=int(n_a),
        x_b=int(x_b),
        n_b=int(n_b),
        p_a=p_a,
        p_b=p_b,
        diff=diff,
        relative_lift=diff / p_a if p_a > 0 else math.inf,
        z=float(z),
        p_value=p_value,
        ci_low=diff - zc * se1,
        ci_high=diff + zc * se1,
        alpha=alpha,
    )


def min_sample_size_per_arm(
    p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Users needed per arm to detect ``p_a`` vs ``p_b`` with a two-sided z-test.

    Classic normal-approximation formula (Fleiss, no continuity correction):

        n = (z_{1-a/2} * sqrt(2 * pbar * qbar) + z_{power} * sqrt(p_a q_a + p_b q_b))^2
            / (p_a - p_b)^2

    e.g. 10% vs 15% at alpha 0.05 / power 0.80 needs 686 per arm.
    """
    if not (0 < p_a < 1 and 0 < p_b < 1):
        raise ValueError("proportions must be in (0, 1)")
    if p_a == p_b:
        raise ValueError("p_a and p_b must differ")
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(power)
    pbar = (p_a + p_b) / 2
    num = (
        za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))
    ) ** 2
    return int(math.ceil(num / (p_a - p_b) ** 2))


def minimum_detectable_effect(
    p_a: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest absolute uplift over ``p_a`` detectable with ``n_per_arm`` users per arm.

    Solved by bisection on :func:`min_sample_size_per_arm`, so both helpers always agree.
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    lo, hi = 1e-6, 1 - p_a - 1e-6
    if min_sample_size_per_arm(p_a, p_a + hi, alpha, power) > n_per_arm:
        return math.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        if min_sample_size_per_arm(p_a, p_a + mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi
