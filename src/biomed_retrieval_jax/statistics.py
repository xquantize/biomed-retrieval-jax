"""Paired statistics over per-query differences between two runs on the same queries."""

import numpy as np
from jaxtyping import Float


def permutation_p_value(
    differences: Float[np.ndarray, " queries"],
    rng: np.random.Generator,
    permutations: int = 10_000,
) -> float:
    """Two-sided paired permutation (sign-flip) test of the mean difference.

    Under the null hypothesis that neither run is better, each query's difference is as likely
    to have either sign, so the observed mean is compared with those of random sign flips.

    Args:
        differences: Each query's metric for one run minus the other's.
        rng: Source of the random sign flips.
        permutations: Number of random sign flips.

    Returns:
        The fraction of sign flips, counting the observed differences as one, whose mean is at
        least as far from zero as the observed mean: at least `1 / (permutations + 1)`.
    """
    observed = abs(differences.mean())
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(differences)))
    extreme = (np.abs((signs * differences).mean(axis=1)) >= observed - 1e-15).sum()
    return float((extreme + 1) / (permutations + 1))


def bootstrap_interval(
    differences: Float[np.ndarray, " queries"],
    rng: np.random.Generator,
    resamples: int = 10_000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean difference, resampling queries.

    Args:
        differences: Each query's metric for one run minus the other's.
        rng: Source of the resamples.
        resamples: Number of resamples of the queries, with replacement.
        level: Coverage of the interval.

    Returns:
        The interval's lower and upper ends.
    """
    samples = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[samples].mean(axis=1)
    low, high = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)
