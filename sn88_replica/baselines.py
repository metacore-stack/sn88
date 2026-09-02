"""Baseline portfolios - the bar every model has to clear, measured through the replica.

Guide §19: a model enters production only if it beats equal weight and inverse
volatility *through the score replica*, not through Sharpe. DeMiguel, Garlappi & Uppal
compared fourteen models and found none consistently beat 1/N out of sample, so this is
a much harder bar than it sounds.

Pure standard library, no numpy. The linear algebra here runs over the 15-40 names you
actually hold, not the 555-name universe, so an O(n^3) solve is a few milliseconds.

Every constructor returns weights that sum to ``gross`` (default 0.99, leaving the 1%
cash band from guide §3.5) and is long-only by default, because shorts count toward the
cash penalty *and* consume gross budget.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

__all__ = [
    "mean",
    "stdev",
    "covariance_matrix",
    "shrink_covariance",
    "ewma_covariance",
    "equal_weight",
    "inverse_volatility",
    "equal_risk_contribution",
    "minimum_variance",
    "hierarchical_risk_parity",
    "BASELINES",
]


# --- statistics -------------------------------------------------------------


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def covariance_matrix(returns: Mapping[str, Sequence[float]]) -> tuple[list[str], list[list[float]]]:
    """Sample covariance. Never invert this directly for production weights."""
    names = sorted(returns)
    n = len(names)
    series = [list(returns[k]) for k in names]
    length = min(len(s) for s in series) if series else 0
    series = [s[-length:] for s in series]
    means = [mean(s) for s in series]
    cov = [[0.0] * n for _ in range(n)]
    if length < 2:
        return names, cov
    for i in range(n):
        for j in range(i, n):
            c = sum((series[i][t] - means[i]) * (series[j][t] - means[j])
                    for t in range(length)) / (length - 1)
            cov[i][j] = cov[j][i] = c
    return names, cov


def shrink_covariance(cov: Sequence[Sequence[float]], intensity: float = 0.2) -> list[list[float]]:
    """Shrink toward a constant-correlation / diagonal target (Ledoit-Wolf in spirit).

    ``intensity`` is fixed rather than estimated - a deliberate simplification. What
    matters for the baseline is that the matrix is conditioned before inversion, not
    that the shrinkage constant is optimal.
    """
    n = len(cov)
    if n == 0:
        return []
    avg_var = sum(cov[i][i] for i in range(n)) / n
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            target = avg_var if i == j else 0.0
            out[i][j] = (1 - intensity) * cov[i][j] + intensity * target
    return out


def ewma_covariance(
    returns: Mapping[str, Sequence[float]], halflife: int = 20
) -> tuple[list[str], list[list[float]]]:
    """Exponentially weighted covariance - the second estimator guide §11.1 requires."""
    names = sorted(returns)
    n = len(names)
    series = [list(returns[k]) for k in names]
    length = min(len(s) for s in series) if series else 0
    series = [s[-length:] for s in series]
    if length < 2:
        return names, [[0.0] * n for _ in range(n)]

    lam = 0.5 ** (1.0 / halflife)
    weights = [lam ** (length - 1 - t) for t in range(length)]
    total = sum(weights)
    weights = [w / total for w in weights]
    means = [sum(weights[t] * series[i][t] for t in range(length)) for i in range(n)]

    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            c = sum(weights[t] * (series[i][t] - means[i]) * (series[j][t] - means[j])
                    for t in range(length))
            cov[i][j] = cov[j][i] = c
    return names, cov


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Gauss-Jordan with partial pivoting. Fine for the 15-40 names actually held."""
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("singular covariance - shrink it before inverting")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        for k in range(col, n + 1):
            aug[col][k] /= p
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col]
            if f:
                for k in range(col, n + 1):
                    aug[r][k] -= f * aug[col][k]
    return [aug[i][n] for i in range(n)]


def _normalise(weights: Mapping[str, float], gross: float, long_only: bool) -> dict[str, float]:
    w = {k: (max(v, 0.0) if long_only else v) for k, v in weights.items()}
    total = sum(abs(v) for v in w.values())
    if total <= 0:
        n = len(w) or 1
        return {k: gross / n for k in w}
    return {k: v / total * gross for k, v in w.items()}


# --- the baselines ----------------------------------------------------------


def equal_weight(names: Sequence[str], *, gross: float = 0.99) -> dict[str, float]:
    """1/N. The hardest baseline to beat, and the cheapest to run."""
    if not names:
        return {}
    return {n: gross / len(names) for n in names}


def inverse_volatility(
    returns: Mapping[str, Sequence[float]], *, gross: float = 0.99
) -> dict[str, float]:
    """Weight by 1/sigma. Naive risk parity, and a genuinely strong baseline here."""
    vols = {k: stdev(v) for k, v in returns.items()}
    inv = {k: (1.0 / s if s > 1e-12 else 0.0) for k, s in vols.items()}
    if not any(inv.values()):
        return equal_weight(sorted(returns), gross=gross)
    return _normalise(inv, gross, long_only=True)


def equal_risk_contribution(
    returns: Mapping[str, Sequence[float]],
    *,
    gross: float = 0.99,
    shrinkage: float = 0.0,
    iterations: int = 2000,
    tol: float = 1e-12,
) -> dict[str, float]:
    """Equal risk contribution (Maillard, Roncalli & Teiletche) by fixed-point iteration.

    ``shrinkage`` defaults to **zero**: unlike :func:`minimum_variance`, ERC never
    inverts the covariance, so it needs no conditioning - and shrinking here would mean
    the risk contributions are equalised on a matrix that is not the one you measure
    them with, which silently leaves them unequal.
    """
    names, cov = covariance_matrix(returns)
    if shrinkage:
        cov = shrink_covariance(cov, shrinkage)
    n = len(names)
    if n == 0:
        return {}
    if n == 1:
        return {names[0]: gross}

    w = [1.0 / n] * n
    for _ in range(iterations):
        marginal = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        port_var = sum(w[i] * marginal[i] for i in range(n))
        if port_var <= 0:
            break
        target = port_var / n
        new = [w[i] * (target / (w[i] * marginal[i])) ** 0.5
               if w[i] * marginal[i] > 1e-18 else w[i] for i in range(n)]
        total = sum(new)
        new = [x / total for x in new]
        if max(abs(new[i] - w[i]) for i in range(n)) < tol:
            w = new
            break
        w = new
    return _normalise(dict(zip(names, w)), gross, long_only=True)


def minimum_variance(
    returns: Mapping[str, Sequence[float]],
    *,
    gross: float = 0.99,
    shrinkage: float = 0.2,
    long_only: bool = True,
) -> dict[str, float]:
    """Global minimum variance on a shrunk covariance.

    Long-only is enforced by clipping and renormalising rather than by a QP - crude, but
    the point of a baseline is to be beaten, not to be elegant.
    """
    names, cov = covariance_matrix(returns)
    if not names:
        return {}
    cov = shrink_covariance(cov, shrinkage)
    try:
        raw = _solve(cov, [1.0] * len(names))
    except ValueError:
        return equal_weight(names, gross=gross)
    return _normalise(dict(zip(names, raw)), gross, long_only=long_only)


def _correlation(cov: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(cov)
    sd = [math.sqrt(cov[i][i]) if cov[i][i] > 0 else 0.0 for i in range(n)]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = cov[i][j] / (sd[i] * sd[j]) if sd[i] > 0 and sd[j] > 0 else 0.0
    return out


def hierarchical_risk_parity(
    returns: Mapping[str, Sequence[float]], *, gross: float = 0.99
) -> dict[str, float]:
    """HRP (Lopez de Prado): cluster, quasi-diagonalise, then bisect by inverse variance.

    Single-linkage clustering on the correlation distance, which is the classic
    formulation and avoids inverting the covariance at all.
    """
    names, cov = covariance_matrix(returns)
    n = len(names)
    if n == 0:
        return {}
    if n <= 2:
        return inverse_volatility(returns, gross=gross)

    corr = _correlation(cov)
    dist = [[math.sqrt(max(0.0, 0.5 * (1 - corr[i][j]))) for j in range(n)] for i in range(n)]

    # single-linkage agglomeration, tracking leaf order
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, pair = float("inf"), (0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(dist[i][j] for i in clusters[a] for j in clusters[b])
                if d < best:
                    best, pair = d, (a, b)
        a, b = pair
        merged = clusters[a] + clusters[b]
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [merged]
    order = clusters[0]

    def cluster_var(items: Sequence[int]) -> float:
        inv = [1.0 / cov[i][i] if cov[i][i] > 1e-18 else 0.0 for i in items]
        total = sum(inv) or 1.0
        w = [x / total for x in inv]
        return sum(w[a] * cov[items[a]][items[b]] * w[b]
                   for a in range(len(items)) for b in range(len(items)))

    weights = {i: 1.0 for i in order}
    stack = [order]
    while stack:
        group = stack.pop()
        if len(group) <= 1:
            continue
        half = len(group) // 2
        left, right = group[:half], group[half:]
        vl, vr = cluster_var(left), cluster_var(right)
        alpha = 1 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= 1 - alpha
        stack.extend([left, right])

    return _normalise({names[i]: weights[i] for i in order}, gross, long_only=True)


#: Every baseline, for the "must beat these" gate in guide §19.
BASELINES = {
    "equal_weight": lambda r, g=0.99: equal_weight(sorted(r), gross=g),
    "inverse_volatility": lambda r, g=0.99: inverse_volatility(r, gross=g),
    "equal_risk_contribution": lambda r, g=0.99: equal_risk_contribution(r, gross=g),
    "minimum_variance": lambda r, g=0.99: minimum_variance(r, gross=g),
    "hierarchical_risk_parity": lambda r, g=0.99: hierarchical_risk_parity(r, gross=g),
}
