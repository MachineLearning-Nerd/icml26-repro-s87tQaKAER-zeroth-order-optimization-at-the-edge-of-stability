"""Paper's closed-form mean-square critical step sizes and bounds (Theorems 1-3).

Source: "Zeroth-Order Optimization at the Edge of Stability", arXiv:2604.14669.

  Theorem 1 (ZO-GD), Eq. 16-17:
      eta_ms* is the unique eta>0 with  eta*lam_max < 1  and
          sum_i  eta*lam_i / (2 (1 - eta*lam_i))  =  1
      bounds:  2/(Tr(H)+2 lam_max)  <= eta_ms* <= 2/Tr(H).

  Theorem 2 (ZO-GDM), Eq. 18-19:
      eta*lam_max < 1-beta^2  and
          sum_i  eta*lam_i / (2 (1-beta) (1 - eta*lam_i/(1-beta^2)))  =  1
      bounds:  2(1-beta)/(Tr(H)+2 lam_max/(1+beta)) <= eta_ms* <= 2(1-beta)/Tr(H).

  Theorem 3 (Frozen ZO-Adam), Eq. 20-21, with lam~_i the eigenvalues of P^{-1}H:
      eta*lam_max(P^{-1}H) < 1+beta1  and
          sum_i  eta*lam~_i / (2 (1 - eta*lam~_i/(1+beta1)))  =  1
      bounds:  2/(Tr(P^{-1}H) + 2 lam_max(P^{-1}H)/(1+beta1))
               <= eta_ms* <= 2 / Tr(P^{-1}H).

Each critical step size is found by a bracketed scalar root solve.  All inputs
are the eigenvalues of the relevant curvature matrix (H for GD/GDM, P^{-1}H for
Adam); we never use the formula itself to *select* the spectrum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Exact critical step size from the fixed-point equation (root of F(eta)=1).
# ---------------------------------------------------------------------------


def critical_eta_zogd(eigs: np.ndarray) -> float:
    lam = np.asarray(eigs, dtype=float)
    lam_max = lam.max()
    # F(eta) = sum_i eta lam_i / (2(1-eta lam_i)); domain eta in (0, 1/lam_max).
    def F(eta: float) -> float:
        return np.sum(eta * lam / (2.0 * (1.0 - eta * lam)))

    target = 1.0
    upper = 1.0 / lam_max
    # monotone increasing in eta on (0,1/lam_max), F->0 at 0, F->+inf near upper.
    lo, hi = 1e-12, (1.0 - 1e-9) / lam_max
    f_lo, f_hi = F(lo), F(hi)
    if f_lo > target or f_hi < target:  # pragma: no cover - sanity
        return float("nan")
    return brentq(lambda e: F(e) - target, lo, hi, xtol=1e-14, rtol=1e-14)


def critical_eta_zogdm(eigs: np.ndarray, beta: float) -> float:
    lam = np.asarray(eigs, dtype=float)
    lam_max = lam.max()
    b2 = beta * beta

    def F(eta: float) -> float:
        return np.sum(eta * lam / (2.0 * (1.0 - beta) * (1.0 - eta * lam / (1.0 - b2))))

    upper = (1.0 - b2) / lam_max
    lo, hi = 1e-12, (1.0 - 1e-9) * upper
    return brentq(lambda e: F(e) - 1.0, lo, hi, xtol=1e-14, rtol=1e-14)


def critical_eta_zoadam(eigs_pinvH: np.ndarray, beta1: float) -> float:
    lam = np.asarray(eigs_pinvH, dtype=float)
    lam_max = lam.max()

    def F(eta: float) -> float:
        return np.sum(eta * lam / (2.0 * (1.0 - eta * lam / (1.0 + beta1))))

    upper = (1.0 + beta1) / lam_max
    lo, hi = 1e-12, (1.0 - 1e-9) * upper
    return brentq(lambda e: F(e) - 1.0, lo, hi, xtol=1e-14, rtol=1e-14)


# ---------------------------------------------------------------------------
# Bounds (Eqs. 17, 19, 21).
# ---------------------------------------------------------------------------


def bounds_zogd(eigs: np.ndarray) -> tuple[float, float]:
    tr = float(np.sum(eigs))
    lmax = float(np.max(eigs))
    return 2.0 / (tr + 2.0 * lmax), 2.0 / tr


def bounds_zogdm(eigs: np.ndarray, beta: float) -> tuple[float, float]:
    tr = float(np.sum(eigs))
    lmax = float(np.max(eigs))
    lo = 2.0 * (1.0 - beta) / (tr + 2.0 * lmax / (1.0 + beta))
    hi = 2.0 * (1.0 - beta) / tr
    return lo, hi


def bounds_zoadam(eigs_pinvH: np.ndarray, beta1: float) -> tuple[float, float]:
    tr = float(np.sum(eigs_pinvH))
    lmax = float(np.max(eigs_pinvH))
    lo = 2.0 / (tr + 2.0 * lmax / (1.0 + beta1))
    hi = 2.0 / tr
    return lo, hi


# ---------------------------------------------------------------------------
# First-order counterparts (Propositions 1-3) -- used as the negative control.
# FO critical step size depends ONLY on the top eigenvalue of the (precond.) H.
# ---------------------------------------------------------------------------


def critical_eta_gd(eigs: np.ndarray) -> float:
    return 2.0 / float(np.max(eigs))


def critical_eta_gdm(eigs: np.ndarray, beta: float) -> float:
    return 2.0 * (1.0 + beta) / float(np.max(eigs))


def critical_eta_frozen_adam(eigs_pinvH: np.ndarray, beta1: float) -> float:
    return 2.0 * (1.0 + beta1) / ((1.0 - beta1) * float(np.max(eigs_pinvH)))
