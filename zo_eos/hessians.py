"""Test Hessians for the theory verification.

Crucially NON-toy: dense (non-diagonal) matrices with realistic spectra, plus a
Hessian built from real CIFAR-10 design data.  The theorems hold for any PSD H,
so we test across a spread of spectra / structures rather than a single toy one.
"""

from __future__ import annotations

import numpy as np


def _orthonormal(d: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q


def random_dense_psd(d: int, spectrum: str, seed: int, cond: float | None = None) -> np.ndarray:
    """A dense PSD matrix with a chosen eigenvalue profile, random eigenbasis."""
    rng = np.random.default_rng(seed)
    if spectrum == "decay":
        lam = 1.0 / np.arange(1, d + 1) ** 1.2
    elif spectrum == "powerlaw":
        lam = d ** (-np.arange(1, d + 1) / 3.0)
    elif spectrum == "uniform":
        lam = np.ones(d)
    elif spectrum == "two_group":
        lam = np.r_[np.full(d // 2, 1.0), np.full(d - d // 2, 0.05)]
    elif spectrum == "linear":
        lam = np.linspace(1.0, 1e-2, d)
    else:
        raise ValueError(spectrum)
    lam = lam / lam.max()
    if cond is not None:  # rescale to a target condition number
        lam = lam * (cond - 1.0) / (lam.max() / lam.min() - 1.0 + 1e-12)
        lam = lam + (1.0)
    Q = _orthonormal(d, rng)
    H = (Q * lam) @ Q.T
    return 0.5 * (H + H.T)


def diagonal_psd(eigs: np.ndarray) -> np.ndarray:
    return np.diag(np.asarray(eigs, dtype=float))


def real_data_hessian(X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Hessian of 1/2 ||X w||^2 (least squares) = X^T X / n  (+ ridge).  Dense,
    non-diagonal, real-data spectrum.  X is the (n, d) design matrix."""
    n = X.shape[0]
    H = (X.T @ X) / n + ridge * np.eye(X.shape[1])
    return 0.5 * (H + H.T)
