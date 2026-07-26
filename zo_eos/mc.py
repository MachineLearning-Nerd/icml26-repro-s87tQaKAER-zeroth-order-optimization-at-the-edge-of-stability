"""Monte Carlo simulation of ZO dynamics on a quadratic -- an independent route
to the mean-square critical step size (route 3 in the verification stack).

Neither the operator nor the paper's formula appear here: we just run the actual
stochastic iterates and watch whether E||x_t||^2 stays bounded.
"""

from __future__ import annotations

import warnings

import numpy as np

# Overflow is expected (and harmless) in the unstable regime eta >> eta_ms*.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def simulate_zogd(H: np.ndarray, eta: float, T: int, n_seeds: int, rng: np.random.Generator,
                  x0_scale: float = 1.0) -> np.ndarray:
    """Return array of shape (n_seeds, T+1) of ||x_t||^2 under ZO-GD on 1/2 x^T H x.

    Vectorized across seeds: x is (n_seeds, d), u is (n_seeds, d) per step.
    """
    d = H.shape[0]
    X = rng.standard_normal((n_seeds, d)) * x0_scale  # (n_seeds, d)
    sq = np.empty((n_seeds, T + 1))
    sq[:, 0] = (X * X).sum(axis=1)
    for t in range(T):
        U = rng.standard_normal((n_seeds, d))
        Hx = X @ H.T                        # (n_seeds, d) = (u^T H x) needs u dot Hx
        dot = (U * Hx).sum(axis=1)          # (n_seeds,) = u^T H x
        g = dot[:, None] * U                # (n_seeds, d) gradient estimate
        X = X - eta * g
        sq[:, t + 1] = (X * X).sum(axis=1)
    return sq


def ms_stable_mc(H: np.ndarray, eta: float, T: int = 400, n_seeds: int = 60,
                 rng: np.random.Generator | None = None,
                 grow_factor: float = 30.0) -> bool:
    """Return True if E||x_t||^2 stays bounded under ZO-GD.

    Criterion: the trajectory is 'unstable' if the mean second moment grows by
    more than `grow_factor` x its starting value by horizon T (a multiplicative
    blow-up, characteristic of the unstable side of the boundary).  Below the
    mean-square critical step size the second moment decays (possibly slowly), so
    it never reaches this factor.
    """
    rng = rng or np.random.default_rng(0)
    sq = simulate_zogd(H, eta, T, n_seeds, rng, x0_scale=3.0)
    mean_ms = sq.mean(axis=0)
    start = mean_ms[0]
    return bool(mean_ms[-1] < grow_factor * start and np.isfinite(mean_ms[-1]))


def find_critical_eta_mc(H: np.ndarray, rng: np.random.Generator,
                         T: int = 400, n_seeds: int = 60,
                         lo: float | None = None, hi: float | None = None,
                         iters: int = 20) -> float:
    """Binary search the empirical critical eta for ZO-GD.

    Uses a log-growth criterion: stable iff mean E||x_t||^2 does not grow
    super-threshold by horizon T.  Returns the boundary eta.
    """
    lam_max = np.linalg.eigvalsh(H).max()
    if hi is None:
        hi = 2.0 / lam_max * 4.0  # search wide of the FO threshold
    if lo is None:
        lo = 1e-5 / lam_max
    # ensure lo stable, hi unstable (shrink/grow to make so)
    for _ in range(20):
        if ms_stable_mc(H, lo, T, n_seeds, rng):
            break
        lo *= 0.5
    for _ in range(20):
        if not ms_stable_mc(H, hi, T, n_seeds, rng):
            break
        hi *= 1.5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ms_stable_mc(H, mid, T, n_seeds, rng):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
