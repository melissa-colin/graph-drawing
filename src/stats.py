"""Statistics of the sweeps: confidence intervals over graphs, paired tests, multiple-comparison correction."""
import numpy as np
from scipy import stats


def bootstrap_ci(values, level=0.95, resamples=2000, seed=0):
    """Mean and percentile bootstrap confidence interval of the mean: (mean, low, high)."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_test(a, b):
    """Two-sided Wilcoxon signed-rank test of a vs b, paired by graph. Returns (mean of a - b, p-value);
    p = 1 when every pair is equal (nothing to test)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.all(d == 0):
        return 0.0, 1.0
    return float(d.mean()), float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as given): control of the family-wise error rate when
    many settings are tested at once."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()
