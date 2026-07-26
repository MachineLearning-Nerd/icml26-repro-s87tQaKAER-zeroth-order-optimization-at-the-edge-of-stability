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
    """Spectral radius of the FPM second-moment operator.

    Fast exact route for simultaneously-diagonalizable (H, D): the operator
    decouples in the shared eigenbasis into (i) a per-pair 4x4 block for every
    off-diagonal covariance entry and (ii) a 3d x 3d 'diagonal' block coupled by
    the trace term Tr(H X H).  rho = max over these.  Falls back to matrix-free
    Arnoldi if H, D do not commute.
    """
    d = H.shape[0]
    # commute test (D diagonal in H's eigenbasis <=> D,H simultaneously diagonalizable)
    comm = np.linalg.norm(H @ D - D @ H) / (np.linalg.norm(H @ D) + 1e-30)
    if comm < 1e-8:
        # rotate to H's eigenbasis; D becomes diagonal there too
        lam, Q = np.linalg.eigh(H)
        delta = np.diag(Q.T @ D @ Q)
        return spectral_radius_fpm_structured(lam, delta, eta, beta, s)
    # non-commuting fallback (not used by our commuting-P test problems)
    from scipy.sparse.linalg import LinearOperator, eigs

    n = 3 * d * d
    op = LinearOperator((n, n), matvec=lambda v: _fpm_apply_flat(v, d, H, D, eta, beta, s))
    k = min(6, n - 1)
    try:
        ev = eigs(op, k=k, which="LM", tol=1e-7, maxiter=400, return_eigenvectors=False)
    except Exception:
        ev = eigs(op, k=min(3, n - 1), which="LM", tol=1e-5, maxiter=800, return_eigenvectors=False)
    return float(np.max(np.abs(ev)))


def spectral_radius_fpm_structured(
    lam: np.ndarray, delta: np.ndarray, eta: float, beta: float, s: float
) -> float:
    """Exact spectral radius in the shared eigenbasis of H (eigs `lam`) and D=diag(`delta`).

    Uses the entrywise second-moment recursion (general frozen-preconditioned ZO
    momentum family; ZO-GD = beta 0, ZO-GDM = delta 1 / s 1, Frozen ZO-Adam =
    delta = diag(P)^{-1}, s = 1-beta1).
    """
    lam = np.asarray(lam, float)
    delta = np.asarray(delta, float)
    d = len(lam)
    # ---- off-diagonal pairs: 4x4 systems on (x=X_ij, c=C_ij, d=C_ji, m=M_ij) ----
    rho_off = 0.0
    for i in range(d):
        ai, di = lam[i], delta[i]
        for j in range(i + 1, d):
            aj, dj = lam[j], delta[j]
            M44 = np.zeros((4, 4))
            # x'
            M44[0] = [1 - eta*s*(di*ai+aj*dj) + 2*eta**2*s**2*di*dj*ai*aj,
                      -eta*beta*dj*(1-eta*s*di*ai), -eta*beta*di*(1-eta*s*aj*dj), eta**2*beta**2*di*dj]
            # c' = C'_ij  (self coef has factor 1: only DHC contributes)
            M44[1] = [s*aj - 2*eta*s**2*di*ai*aj, beta - eta*s*beta*di*ai, -eta*beta*s*di*aj, -eta*beta**2*di]
            # d' = C'_ji  (swap i<->j)
            M44[2] = [s*ai - 2*eta*s**2*dj*ai*aj, -eta*beta*s*dj*ai, beta - eta*s*beta*dj*aj, -eta*beta**2*dj]
            # m' = M'_ij  (HC, C^T H carry no D)
            M44[3] = [2*s**2*ai*aj, s*beta*ai, s*beta*aj, beta**2]
            rho_off = max(rho_off, float(np.max(np.abs(np.linalg.eigvals(M44)))))
    # ---- diagonal block: 3d x 3d on (x_i, c_i, m_i), coupled by T = sum_k lam_k^2 x_k ----
    # ordering: [x(0..d-1), c(d..2d-1), m(2d..3d-1)]
    M3 = np.zeros((3 * d, 3 * d))
    lam2 = lam ** 2
    for i in range(d):
        ai, di, li = lam[i], delta[i], lam2[i]
        # x'_i
        M3[i, i] = 1 - 2*eta*s*di*ai + 2*eta**2*s**2*di**2*ai**2
        M3[i, d + i] = -2*eta*beta*di*(1 - eta*s*di*ai)
        M3[i, 2*d + i] = eta**2*beta**2*di**2
        # the trace term eta^2 s^2 di^2 T contributes eta^2 s^2 di^2 * (lam_k^2) to column x_k
        for k in range(d):
            M3[i, k] += eta**2*s**2*di**2*lam2[k]
        # c'_i
        M3[d + i, i] = s*ai - 2*eta*s**2*di*ai**2
        M3[d + i, d + i] = beta - 2*eta*s*beta*di*ai
        M3[d + i, 2*d + i] = -eta*beta**2*di
        for k in range(d):
            M3[d + i, k] += -eta*s**2*di*lam2[k]
        # m'_i  (HC carries no D)
        M3[2*d + i, i] = 2*s**2*ai**2
        M3[2*d + i, d + i] = 2*s*beta*ai
        M3[2*d + i, 2*d + i] = beta**2
        for k in range(d):
            M3[2*d + i, k] += s**2*lam2[k]
    rho_diag = float(np.max(np.abs(np.linalg.eigvals(M3))))
    return max(rho_off, rho_diag)


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
