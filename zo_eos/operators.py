"""Exact mean-square second-moment operators for zeroth-order methods on a quadratic.

We study the quadratic f(x) = 1/2 x^T H x (x* = 0), H PSD, H != 0, with the
standard two-point Gaussian estimator

    grad_hat(x) = (f(x+mu u) - f(x-mu u)) / (2 mu) * u .

On a *quadratic* this is exact for any mu: f(x+mu u)-f(x-mu u) = 2 mu (u^T H x),
so grad_hat(x) = (u^T H x) u, u ~ N(0, I).  The estimator is unbiased:
E_u[grad_hat] = H x.

Mean-square stability of the resulting (linear, stochastic) dynamics is governed
by the spectral radius of the *second-moment linear operator* that maps
S_t = E[x_t x_t^T] -> S_{t+1}.  Below we derive that operator directly from the
update rule and Gaussian fourth moments (Isserlis' theorem) -- independently of
the paper's cone / Krein-Rutman argument -- so that agreement with the paper's
closed-form critical step size is a genuine verification, not a tautology.

General frozen-preconditioned ZO momentum family
-------------------------------------------------
    m' = beta m + s * B(u) x          (B(u) = u u^T H ;  u ~ N(0, I))
    x' = x - eta D m'                 (D = P^{-1} preconditioner)
with the specializations
    ZO-GD          : beta = 0, s = 1, D = I        (state collapses to x only)
    ZO-GDM         : beta = beta, s = 1, D = I
    Frozen ZO-Adam : beta = beta1, s = (1-beta1), D = P^{-1}
"""

from __future__ import annotations

import numpy as np


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


# ---------------------------------------------------------------------------
# Gaussian fourth-moment identities used throughout (Isserlis / Wick).
# For u ~ N(0, I), B(u) = u u^T H:
#   E[B]              = H
#   E[B X B^T]        = Tr(H X H) I + 2 H X H            (X symmetric)
# We call the latter the "R" term.
# ---------------------------------------------------------------------------


def _R_term(H: np.ndarray, X: np.ndarray) -> np.ndarray:
    """E_u[ B(u) X B(u)^T ] = Tr(H X H) I + 2 H X H,  B(u)=u u^T H, u~N(0,I)."""
    HXH = H @ X @ H
    return np.trace(HXH) * np.eye(H.shape[0]) + 2.0 * HXH


# ---------------------------------------------------------------------------
# General second-moment apply for the frozen-preconditioned ZO momentum family.
# State: X = E[x x^T] (sym), C = E[x m^T] (general), M = E[m m^T] (sym).
# Returns the next (X', C', M').  Derived analytically from the update rule.
# ---------------------------------------------------------------------------


def apply_fpm(
    X: np.ndarray,
    C: np.ndarray,
    M: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    eta: float,
    beta: float,
    s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One step of the exact second-moment recursion for the FPM family.

    Update:  m' = beta m + s B(u) x ;  x' = x - eta D m' ,  B(u)=u u^T H.
    """
    d = H.shape[0]
    I = np.eye(d)
    DH = D @ H
    HD = H @ D.T  # = (D H)^T only if H sym; H is sym so H D^T = (D H)^T
    R = _R_term(H, X)  # Tr(HXH) I + 2 H X H
    DRD = D @ R @ D.T

    # X' = E[x' x'^T]
    Xp = (
        X
        - eta * s * (DH @ X + X @ HD)
        + eta ** 2 * s ** 2 * DRD
        - eta * beta * ((I - eta * s * DH) @ C @ D.T + D @ C.T @ (I - eta * s * HD))
        + eta ** 2 * beta ** 2 * (D @ M @ D.T)
    )

    # C' = E[x' m'^T].  Direct derivation (x'=x-eta beta D m - eta s D B x;
    #   m'=beta m + s B x, B=uu^T H):
    #   C' = beta C + s X H - eta beta^2 D M - eta beta s D C^T H
    #        - eta s beta D H C - eta s^2 D R
    #   (the xx^T B^T and mx^T B^T contractions end in E_u[B^T]=H, with no D^T.)
    Cp = (
        s * X @ H
        + beta * C
        - eta * s * beta * (D @ H @ C)
        - eta * beta * s * (D @ C.T @ H)
        - eta * beta ** 2 * (D @ M)
        - eta * s ** 2 * (D @ R)
    )

    # M' = E[m' m'^T]
    Mp = s ** 2 * R + s * beta * (H @ C + C.T @ H) + beta ** 2 * M

    return _sym(Xp), Cp, _sym(Mp)


def apply_gd(X: np.ndarray, H: np.ndarray, eta: float) -> np.ndarray:
    """ZO-GD second-moment step:  S' = S - eta(HS+SH) + eta^2 (Tr(H^2 S) I + 2 HSH)."""
    HSH = H @ X @ H
    return X - eta * (H @ X + X @ H) + eta ** 2 * (np.trace(HSH) * np.eye(H.shape[0]) + 2.0 * HSH)


# ---------------------------------------------------------------------------
# Spectral radius of the second-moment operator.
#
# Two exact, independent computations:
#   (1) ZO-GD closed form in H's eigenbasis (scales to huge d):
#         - off-diagonal entries of S decouple with multiplier
#             m_ij = 1 - eta(lam_i+lam_j) + 2 eta^2 lam_i lam_j ;
#         - the diagonal vector s = (s_ii) evolves under a d x d matrix
#             Mdiag_ij = [i==j](1-2eta lam_i + 2eta^2 lam_i^2) + eta^2 lam_j^2 .
#         rho = max( max_{i<j}|m_ij|, rho(Mdiag) ).
#   (2) FPM (GDM / frozen ZO-Adam): the operator on vec(X,C,M) is linear; we use
#       scipy.sparse.linalg.eigs (Arnoldi, matrix-free) on the analytic apply to
#       get the largest-magnitude eigenvalue = spectral radius.  Validated against
#       dense eig on small d (see dense_operator_fpm below).
# ---------------------------------------------------------------------------


def spectral_radius_gd(eigs: np.ndarray, eta: float) -> float:
    """Exact spectral radius of the ZO-GD second-moment operator from H's spectrum."""
    lam = np.asarray(eigs, dtype=float)
    d = len(lam)
    # off-diagonal multipliers
    if d > 1:
        i, j = np.meshgrid(lam, lam, indexing="ij")
        m = 1.0 - eta * (i + j) + 2.0 * eta ** 2 * i * j
        off = float(np.max(np.abs(m[np.triu_indices(d, 1)])))
    else:
        off = 0.0
    # diagonal block
    Mdiag = np.diag(1.0 - 2.0 * eta * lam + 2.0 * eta ** 2 * lam ** 2)
    Mdiag = Mdiag + eta ** 2 * np.outer(np.ones(d), lam ** 2)  # rank-1 trace coupling
    rho_diag = float(np.max(np.abs(np.linalg.eigvals(Mdiag))))
    return max(off, rho_diag)


def _fpm_apply_flat(v: np.ndarray, d: int, H, D, eta, beta, s) -> np.ndarray:
    X = v[: d * d].reshape(d, d)
    C = v[d * d : 2 * d * d].reshape(d, d)
    M = v[2 * d * d :].reshape(d, d)
    Xp, Cp, Mp = apply_fpm(_sym(X), C, _sym(M), H, D, eta, beta, s)
    return np.concatenate([Xp.reshape(-1), Cp.reshape(-1), Mp.reshape(-1)])


def spectral_radius_fpm(
    H: np.ndarray, D: np.ndarray, eta: float, beta: float, s: float
) -> float:
    """Spectral radius of the FPM second-moment operator via matrix-free Arnoldi."""
    from scipy.sparse.linalg import LinearOperator, eigs

    d = H.shape[0]
    n = 3 * d * d
    op = LinearOperator((n, n), matvec=lambda v: _fpm_apply_flat(v, d, H, D, eta, beta, s))
    # largest-magnitude eigenvalue.  k must be < n; use a few for robustness.
    k = min(8, n - 1)
    try:
        ev = eigs(op, k=k, which="LM", tol=1e-9, maxiter=2000, return_eigenvectors=False)
    except Exception:
        ev = eigs(op, k=min(3, n - 1), which="LM", tol=1e-6, maxiter=5000, return_eigenvectors=False)
    return float(np.max(np.abs(ev)))


# ---------------------------------------------------------------------------
# Dense Kronecker operator for ZO-GD (validation of the power-iteration route).
# M(eta) = I - eta(I⊗H + H⊗I) + eta^2 [ 2 (H⊗H) + vec(I) vec(H^2)^T ].
# Acts on vec(S).  Spectral radius via dense eig -- ground truth for small d.
# ---------------------------------------------------------------------------


def dense_operator_gd(H: np.ndarray, eta: float) -> np.ndarray:
    d = H.shape[0]
    I = np.eye(d)
    H2 = H @ H
    kII = np.eye(d * d)
    kHH = np.kron(H, H)
    base = kII - eta * (np.kron(I, H) + np.kron(H, I)) + eta ** 2 * (2.0 * kHH)
    vi = I.reshape(-1)
    vh2 = H2.reshape(-1)
    base = base + eta ** 2 * np.outer(vi, vh2)  # rank-1 trace coupling
    return base


def dense_spectral_radius_gd(H: np.ndarray, eta: float) -> float:
    M = dense_operator_gd(H, eta)
    ev = np.linalg.eigvals(M)
    return float(np.max(np.abs(ev)))


def dense_spectral_radius_fpm(
    H: np.ndarray, D: np.ndarray, eta: float, beta: float, s: float
) -> float:
    """Ground-truth spectral radius of the FPM operator via a dense 3d^2 x 3d^2
    matrix built by finite-differencing the (exact) analytic apply.  Small d only.
    """
    d = H.shape[0]
    n = 3 * d * d
    Id = np.eye(n)
    cols = np.empty((n, n))
    for j in range(n):
        cols[:, j] = _fpm_apply_flat(Id[:, j], d, H, D, eta, beta, s)
    ev = np.linalg.eigvals(cols)
    return float(np.max(np.abs(ev)))
