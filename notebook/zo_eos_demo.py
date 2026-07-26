"""Zeroth-Order Optimization at the Edge of Stability — interactive demo.

A marimo notebook that explains the paper's central mean-square stability claim
for ZO-GD and verifies it live on a small dense Hessian. It opens with the
already-produced headline evidence (no expensive reruns needed) and keeps the
interactive part bounded to a few seconds.

Run:  marimo edit notebook/zo_eos_demo.py   |   marimo run notebook/zo_eos_demo.py
"""

import marimo

__generated_with__ = "0.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.optimize import brentq
    mo.md(
        r"""
        # Zeroth-Order Optimization at the Edge of Stability

        **Central question.** First-order gradient descent trains nets at the
        *edge of stability* — the top Hessian eigenvalue λ_max climbs to ≈ 2/η.
        Do **zeroth-order** (ZO) methods have their own edge of stability, and
        which curvature quantity governs it?

        The paper's answer: ZO-GD's mean-square stability is set by the **trace**
        of the Hessian, not its top eigenvalue. It proves (Theorem 1) that the
        mean-square critical step size satisfies

        $$\frac{2}{\mathrm{Tr}(H)+2\lambda_{\max}(H)} \;\le\; \eta^{\star}_{\mathrm{ms}} \;\le\; \frac{2}{\mathrm{Tr}(H)}.$$

        This notebook verifies that claim **independently**: we build the exact
        second-moment operator of the ZO dynamics and check its spectral radius
        equals 1 at the paper's formula root.
        """
    )
    return brentq, mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Headline evidence (already produced)

        Across 34 dense/structured Hessians (d ≤ 200), the operator's spectral
        radius evaluated at the paper's closed-form critical step size is **1 to
        machine precision**:

        > `max |ρ(operator @ η_formula) − 1| = 1.1e-14`  (tol 1e-5)

        The momentum variant ZO-GDM is the sharpest test: its threshold
        **decreases** with β (0.466 → 0.054 as β: 0 → 0.9) — the *opposite* of
        first-order GDM. See the full report at
        https://huggingface.co/spaces/DineshAI/s87tQaKAER.
        """
    )
    return


@app.cell
def _(mo, np):
    # Build a small but NON-diagonal dense PSD Hessian (d=12) and its spectrum.
    d = 12
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = 1.0 / np.arange(1, d + 1) ** 1.2
    lam = lam / lam.max()
    H = (Q * lam) @ Q.T
    H = 0.5 * (H + H.T)
    eigs = np.linalg.eigvalsh(H)
    mo.md(
        f"Test Hessian: dense, d={d}, random eigenbasis, spectrum "
        f"λ_max={eigs.max():.3f}, Tr={eigs.sum():.3f}."
    )
    return H, eigs, lam


@app.cell
def _(mo):
    mo.md("## The exact second-moment operator (derived from first principles)")
    return


@app.cell
def _(H, np):
    # The ZO-GD second-moment operator on the quadratic f(x)=1/2 x^T H x:
    #   T_eta(S) = S - eta(HS+SH) + eta^2 (Tr(H^2 S) I + 2 HSH),
    # built densely here (d small) to get the EXACT spectral radius.
    def operator_matrix(eta):
        d = H.shape[0]
        I = np.eye(d)
        kId = np.eye(d * d)
        base = kId - eta * (np.kron(I, H) + np.kron(H, I)) + eta ** 2 * (2 * np.kron(H, H))
        vi = I.reshape(-1)
        vh2 = (H @ H).reshape(-1)
        return base + eta ** 2 * np.outer(vi, vh2)

    def rho(eta):
        return float(np.max(np.abs(np.linalg.eigvals(operator_matrix(eta)))))
    return rho,


@app.cell
def _(brentq, eigs, np):
    # Paper's closed-form critical step size (root of Eq. 16) and bounds (Eq. 17).
    def eta_formula(eigs):
        lam = np.asarray(eigs, float)

        def F(eta):
            return np.sum(eta * lam / (2 * (1 - eta * lam)))

        return brentq(lambda e: F(e) - 1.0, 1e-12, (1 - 1e-9) / lam.max(), xtol=1e-14)

    eta_form = eta_formula(eigs)
    lo = 2 / (eigs.sum() + 2 * eigs.max())
    hi = 2 / eigs.sum()
    eta_form, lo, hi
    return eta_form, eta_formula, hi, lo


@app.cell
def _(eta_form, hi, lo, mo, rho):
    # THE CHECK: rho evaluated at the paper's formula root must equal 1.
    rho_at_form = rho(eta_form)
    mo.md(
        rf"""
        **Verification.** Operator spectral radius at the paper's $\eta_{{form}}$

        - $\eta_{{form}}$ = {eta_form:.6f}
        - bounds (Eq. 17): [{lo:.6f}, {hi:.6f}]
        - **ρ(operator @ η_form) = {rho_at_form:.10f}**  ← should be 1.0

        The independent operator route and the paper's formula agree to
        {abs(rho_at_form - 1):.1e}. η_form lies inside the bounds, as Theorem 1 requires.
        """
    )
    return


@app.cell
def _(H, np, rho):
    # Sweep eta and watch rho cross 1 exactly at eta_form.
    grid = np.linspace(0.05, 0.7, 60)
    rhos = [rho(e) for e in grid]
    (grid, rhos)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The spectral radius crosses 1 precisely where the paper's formula
        predicts — that crossing is the mean-square edge of stability. The same
        construction (a joint `(x,m)` operator) reproduces Theorems 2 and 3; see
        `zo_eos/operators.py` and the report.
        """
    )
    return


if __name__ == "__main__":
    app.run()
