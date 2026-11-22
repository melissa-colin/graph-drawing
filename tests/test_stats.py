import numpy as np
import pytest

from stats import bootstrap_ci, holm, paired_test


def test_bootstrap_ci_covers_the_mean_and_shrinks():
    rng = np.random.default_rng(0)
    small, large = rng.normal(3, 1, 50), rng.normal(3, 1, 2000)
    m, lo, hi = bootstrap_ci(small)
    assert lo < m < hi and m == pytest.approx(small.mean())
    _, lo2, hi2 = bootstrap_ci(large)
    assert hi2 - lo2 < hi - lo
    # about 1.96 sigma / sqrt(n) on each side
    assert (hi2 - lo2) / 2 == pytest.approx(1.96 / np.sqrt(2000), rel=0.15)


def test_bootstrap_ci_of_a_constant():
    assert bootstrap_ci([2.0] * 10) == (2.0, 2.0, 2.0)


def test_paired_test():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 200)
    diff, p = paired_test(base + 0.3, base)
    assert diff == pytest.approx(0.3) and p < 1e-10
    _, p_null = paired_test(base + rng.normal(0, 0.01, 200), base)
    assert p_null > 0.01
    assert paired_test(base, base) == (0.0, 1.0)


def test_holm_against_a_hand_computation():
    # sorted p: 0.01 (x4 = 0.04), 0.02 (x3 = 0.06), 0.03 (x2 = 0.06), 0.5 (x1 = 0.5)
    assert holm([0.03, 0.5, 0.01, 0.02]) == pytest.approx([0.06, 0.5, 0.04, 0.06])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])
