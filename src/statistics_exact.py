"""Exact small-sample statistics used for the restoration comparison.

Only the Python standard library is required. Clopper-Pearson bounds use
binomial-CDF inversion, matching the original E16 analysis (80 bisections).
"""

import math


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1-alpha) binomial interval, including x=0 and x=n."""
    if not isinstance(x, int) or not isinstance(n, int):
        raise ValueError("x and n must be integers")
    if not 0 <= x <= n or n <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid Clopper-Pearson arguments")

    def invert(k: int, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo + hi) / 2
            cdf = sum(math.comb(n, i) * p**i * (1 - p)**(n - i)
                      for i in range(k + 1))
            if cdf > target:
                lo = p
            else:
                hi = p
        return (lo + hi) / 2

    return (0.0 if x == 0 else invert(x - 1, 1 - alpha / 2),
            1.0 if x == n else invert(x, alpha / 2))


def paired_risk_difference(b: int, c: int, n: int) -> tuple[float, float, float]:
    """RD and conservative individual 95% interval from two 97.5% CP CIs.

    b is target-only execution, c is control-only execution. The interval is
    [L_b - U_c, U_b - L_c], not an inversion of the McNemar test.
    """
    if any(not isinstance(v, int) for v in (b, c, n)):
        raise ValueError("paired counts must be integers")
    if n <= 0 or min(b, c) < 0 or b + c > n:
        raise ValueError("invalid paired counts")
    lb, ub = clopper_pearson(b, n, alpha=0.025)
    lc, uc = clopper_pearson(c, n, alpha=0.025)
    return (b - c) / n, lb - uc, ub - lc


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value conditional on discordant pairs."""
    if any(not isinstance(v, int) or v < 0 for v in (b, c)):
        raise ValueError("discordant counts must be nonnegative integers")
    if b + c == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(b + c, i)
                           for i in range(min(b, c) + 1)) / 2**(b + c))


def holm(pvalues: list[float]) -> list[float]:
    """Holm adjusted p-values, returned in the original comparison order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("invalid p-value")
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted
