"""Monte-Carlo power of H2 of the second held-out test (docs/preregistration_heldout_relations_2.md).

H2 holds when each of the three core NASA-TLX pairs has a positive pooled within-person correlation,
one-sided HKSJ test, Holm-corrected over the three pairs, all three significant.

Per simulated study the true effects of the three pairs are drawn on Fisher's z from N(atanh(r), tau^2),
correlated 0.8 across the pairs (a study whose TLX subscales cohere does so for every pair), plus sampling
error with variance 1 / (df - 1), df = n * (m - 1) - 1 for a repeated-measures correlation of n
participants measured m times. Studies have 15 to 59 participants (uniform) and m = 3. The effects are
pooled with the repository's REML (``multi_study_analysis.pool_effects``) and tested with HKSJ as
``construct_relations`` does.

    python -m analyses.heldout_2.power_h2
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from opendv.analysis import multi_study_analysis as msa
from opendv.analysis.heldout import holm

SEED = 20261114
SIMS = 2000


def p_one_sided(z: np.ndarray, v: np.ndarray) -> float:
    pooled = msa.pool_effects(z, v, "REML")
    return float(stats.t.sf(pooled["random_effects_mean"] / pooled["hksj_se"], len(z) - 1))


def h2_power(k: int, r: float, tau: float, rng: np.random.Generator, n: int = 30, m: int = 3,
             sims: int = SIMS, rho: float = 0.8) -> tuple[float, float]:
    """(P(H2 supported), P(one pair alone p < .05))."""
    mu = np.arctanh(r)
    cov = tau ** 2 * (rho * np.ones((3, 3)) + (1 - rho) * np.eye(3))
    supported = single = 0
    for _ in range(sims):
        ns = rng.integers(max(11, n // 2), n * 2, size=k)
        v = 1.0 / (ns * (m - 1) - 2)
        z = rng.multivariate_normal([mu] * 3, cov, size=k) + rng.normal(0, np.sqrt(v)[:, None], size=(k, 3))
        p = {str(i): p_one_sided(z[:, i], v) for i in range(3)}
        supported += all(adjusted < 0.05 for adjusted in holm(p).values())
        single += p["0"] < 0.05
    return supported / sims, single / sims


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("k   r     tau   P(H2 supported)  P(one pair p < .05)")
    for k in (6, 8, 10):
        for r in (0.30, 0.45, 0.55):
            for tau in (0.25, 0.40):
                both, one = h2_power(k, r, tau, rng)
                print(f"{k:<3} {r:.2f}  {tau:.2f}  {both:.2f}             {one:.2f}")


if __name__ == "__main__":
    main()
