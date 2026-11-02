"""The permutation test and the bootstrap interval, against exact and analytic values."""

import itertools

import numpy as np
import pytest

from biomed_retrieval_jax.statistics import bootstrap_interval, permutation_p_value

# Standard error of a proportion estimated from 10,000 permutations: at most 0.5 / 100.
MONTE_CARLO_ERROR = 0.005


def exact_p_value(differences):
    """The permutation p-value from every sign pattern, for a few queries."""
    observed = abs(differences.mean())
    patterns = np.array(list(itertools.product([-1.0, 1.0], repeat=len(differences))))
    return float(np.mean(np.abs((patterns * differences).mean(axis=1)) >= observed - 1e-15))


@pytest.mark.parametrize("seed", range(3))
def test_permutation_p_value_matches_exact_enumeration(seed):
    differences = np.random.default_rng(seed).normal(0.3, 1.0, size=10)
    estimate = permutation_p_value(differences, np.random.default_rng(seed))
    exact = exact_p_value(differences)
    print(f"seed {seed}: estimate {estimate:.4f}, exact {exact:.4f}")
    assert abs(estimate - exact) < 4 * MONTE_CARLO_ERROR


def test_permutation_p_value_limits():
    rng = np.random.default_rng(0)
    assert permutation_p_value(np.zeros(50), rng) == 1.0
    # Only the two all-equal sign patterns (probability 2 / 2^50) are as extreme.
    assert permutation_p_value(np.full(50, 0.1), rng) == pytest.approx(1 / 10_001)


def test_bootstrap_interval_matches_normal_approximation():
    differences = np.random.default_rng(0).normal(0.05, 1.0, size=2000)
    low, high = bootstrap_interval(differences, np.random.default_rng(1))
    half_width = 1.96 * differences.std(ddof=1) / np.sqrt(len(differences))
    gap = abs((high - low) / 2 - half_width) / half_width
    print(f"half-width {(high - low) / 2:.5f}, normal approximation {half_width:.5f} ({gap:.1%})")
    assert low < differences.mean() < high
    assert gap < 0.1
    assert bootstrap_interval(np.zeros(20), np.random.default_rng(0)) == (0.0, 0.0)
