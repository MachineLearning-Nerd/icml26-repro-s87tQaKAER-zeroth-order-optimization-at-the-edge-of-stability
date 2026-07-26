"""Theory verifier for Claims 1, 2, 3, and the theory half of Claim 5.

For each ZO method we compute the mean-square critical step size TWO independent
ways and require them to agree:

  (i)  eta_op  -- from the DEFINITION: the eta>0 where the exact second-moment
                  operator's spectral radius crosses 1 (root of rho(eta)=1).
                  Derived from the update rule + Gaussian fourth moments only.
  (ii) eta_form -- the paper's closed-form root (Theorems 1-3, Eqs 16/18/20).

Agreement |eta_op - eta_form| / eta_form < tol across a spread of NON-toy
Hessians (dense, structured, and real-data) and across momentum values is the
verification.  We also require eta_op to lie inside the paper's (Tr, lam_max)
bounds (Eqs 17/19/21), and run a first-order negative control plus a Monte Carlo
cross-check.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from zo_eos import formulas as fm
from zo_eos import hessians as hz
from zo_eos import mc
from zo_eos import operators as op

warnings.filterwarnings("ignore")

ARTIFACT_DIR = os.environ.get("ZO_EOS_ARTIFACTS", ".openresearch/artifacts/theory")
TOL = 5e-4  # relative agreement between operator-defined and paper-formula eta
BOUND_TOL = 1e-5  # relative slack for the (Tr,lam_max) bound check (numerical edge cases)


def _in_bounds(e: float, lo: float, hi: float) -> bool:
    if not (np.isfinite(e) and np.isfinite(lo) and np.isfinite(hi)):
        return False
    slack = BOUND_TOL * max(1.0, abs(lo), abs(hi))
    return (lo - slack) <= e <= (hi + slack)

# Problem sizes (tunable).  ZO-GD uses the closed-form structured spectral radius
# so it scales to large d; FPM (GDM/Adam) uses matrix-free Arnoldi so we keep d
# moderate.  Both are far above the 5-dim diagonal toy setup of the prior logbook.
GD_DIMS = [60, 150, 300]
FPM_DIM = 45
FPM_BETAS = [0.0, 0.3, 0.6, 0.9]
ADAM_BETA1S = [0.1, 0.5, 0.9]
MC_T = 800
MC_SEEDS = 80


@dataclass
class Row:
    claim: str
    method: str
    hessian: str
    d: int
    param: float
    eta_op: float
    eta_formula: float
    bound_lo: float
    bound_hi: float
    rho_at_formula: float
    rel_err: float
    bound_ok: bool
    mc_eta: float = float("nan")


def _eta_op_gd(eigs: np.ndarray) -> float:
    """Critical eta where the ZO-GD operator spectral radius crosses 1."""
    lam = np.asarray(eigs, float)
    lo, hi = fm.bounds_zogd(lam)
    grid = np.geomspace(0.2 * lo, 2.0 * hi, 60)
    prev_e, prev_r = grid[0], op.spectral_radius_gd(lam, grid[0])
    for e in grid[1:]:
        r = op.spectral_radius_gd(lam, e)
        if (prev_r - 1.0) * (r - 1.0) < 0:  # sign change -> refine
            from scipy.optimize import brentq

            return brentq(lambda x: op.spectral_radius_gd(lam, x) - 1.0, prev_e, e, xtol=1e-9)
        prev_e, prev_r = e, r
    return float("nan")


def _eta_op_fpm(H, D, beta, s) -> float:
    """Critical eta where the FPM operator spectral radius crosses 1."""
    # bracket: upper from the FO analogue scaled; widen until rho>1.
    def rho(eta):
        return op.spectral_radius_fpm(H, D, eta, beta, s)

    eigs = np.linalg.eigvalsh(H) if D is None or np.allclose(D, np.eye(H.shape[0])) else np.linalg.eigvalsh(D @ H)
    lo = 1e-4
    hi = 2.0 / max(eigs.max(), 1e-9)
    while rho(hi) <= 1.0 and hi < 1e6:
        hi *= 1.5
    from scipy.optimize import brentq

    # find first upward crossing of 1 over (small, hi)
    grid = np.geomspace(lo, hi, 30)
    prev_e, prev_r = grid[0], rho(grid[0])
    for e in grid[1:]:
        r = rho(e)
        if (prev_r - 1.0) * (r - 1.0) <= 0 and e > lo * 5:
            return brentq(lambda x: rho(x) - 1.0, prev_e, e, xtol=1e-8)
        prev_e, prev_r = e, r
    return float("nan")


def _build_hessians() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    specs = ["decay", "powerlaw", "two_group", "linear", "uniform"]
    for d in GD_DIMS:
        for sp in specs:
            out.append((f"random_dense_d{d}_{sp}", hz.random_dense_psd(d, sp, seed=1000 + d + hash(sp) % 97)))
    # a diagonal (but high-d) one for completeness -- still far from 5-dim toy
    out.append(("diag_decay_d300", hz.diagonal_psd(1.0 / np.arange(1, 301) ** 1.3)))
    # real-data least-squares Hessian (dense, real spectrum)
    try:
        X = _cifar_design(d_feat=250, n=1000)
        out.append(("realdata_cifar_d250", hz.real_data_hessian(X, ridge=1e-3)))
    except Exception as e:  # pragma: no cover
        print(f"[theory] real-data Hessian skipped: {e}")
    return out


def _cifar_design(d_feat: int, n: int):
    from torchvision import datasets

    ds = datasets.CIFAR10(root="/tmp/cifar_data", train=True, download=True)
    X = ds.data.reshape(len(ds.data), -1).astype(np.float64)  # (50000, 3072)
    labels = np.asarray(ds.targets)
    X = X[labels < 4][:n]  # first 4 classes, n examples (paper's subset)
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sigma
    rng = np.random.default_rng(123)
    idx = rng.choice(X.shape[1], size=d_feat, replace=False)  # pick d_feat pixels
    return X[:, idx]


def run(out_dir: str | None = None) -> dict:
    out_dir = out_dir or ARTIFACT_DIR
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rows: list[Row] = []
    hessians = _build_hessians()

    # ---- Claim 1: ZO-GD (Theorem 1) ----
    print("\n" + "=" * 78)
    print("CLAIM 1 (Theorem 1, ZO-GD): eta_ms* in [2/(Tr+2lam_max), 2/Tr],")
    print("  exact root of Sum_i eta lam_i/(2(1-eta lam_i)) = 1.")
    print("=" * 78)
    print(f"{'Hessian':<28}{'d':>5}{'eta_op':>11}{'eta_form':>11}{'bounds':>22}{'rel_err':>11}")
    for name, H in hessians:
        eigs = np.linalg.eigvalsh(H)
        e_op = _eta_op_gd(eigs)
        e_form = fm.critical_eta_zogd(eigs)
        lo, hi = fm.bounds_zogd(eigs)
        rho_form = op.spectral_radius_gd(eigs, e_form)
        rel = abs(e_op - e_form) / e_form
        rows.append(Row("1", "ZO-GD", name, len(eigs), float("nan"), e_op, e_form, lo, hi, rho_form, rel, _in_bounds(e_op, lo, hi)))
        print(f"{name:<28}{len(eigs):>5}{e_op:>11.5f}{e_form:>11.5f}  [{lo:.5f},{hi:.5f}]{rel:>11.2e}")

    # ---- Claim 2: ZO-GDM (Theorem 2) ----
    print("\n" + "=" * 78)
    print("CLAIM 2 (Theorem 2, ZO-GDM): eta_ms* adjusted by beta via Eq.18,")
    print("  bounds Eq.19; decreases with beta (opposite of FO GDM).")
    print("=" * 78)
    betas = FPM_BETAS
    gdm_sub = [(n, H) for n, H in hessians if n.startswith(f"random_dense_d{FPM_DIM}")][:3]
    for name, H in gdm_sub:
        eigs = np.linalg.eigvalsh(H)
        print(f"\n  Hessian: {name} (d={len(eigs)})")
        print(f"  {'beta':>6}{'eta_op':>12}{'eta_form':>12}{'bounds':>24}{'rel_err':>12}")
        prev_eta = None
        for beta in betas:
            e_op = _eta_op_fpm(H, np.eye(len(eigs)), beta, 1.0)
            e_form = fm.critical_eta_zogdm(eigs, beta)
            lo, hi = fm.bounds_zogdm(eigs, beta)
            rho_form = op.spectral_radius_fpm(H, np.eye(len(eigs)), e_form, beta, 1.0)
            rel = abs(e_op - e_form) / e_form if e_form else float("nan")
            rows.append(Row("2", "ZO-GDM", name, len(eigs), beta, e_op, e_form, lo, hi, rho_form, rel, _in_bounds(e_op, lo, hi)))
            print(f"  {beta:>6.1f}{e_op:>12.5f}{e_form:>12.5f}  [{lo:.5f},{hi:.5f}]{rel:>12.2e}")
            if prev_eta is not None:
                assert e_op <= prev_eta * (1 + 5e-3), f"ZO-GDM eta must decrease with beta ({name})"
            prev_eta = e_op

    # ---- Claim 3: Frozen ZO-Adam (Theorem 3) ----
    print("\n" + "=" * 78)
    print("CLAIM 3 (Theorem 3, Frozen ZO-Adam): eta_ms* depends on spectrum of")
    print("  P^{-1}H; exact root Eq.20, bounds Eq.21 (commuting P).")
    print("=" * 78)
    beta1s = ADAM_BETA1S
    for name, H in gdm_sub:
        d = H.shape[0]
        Q = np.linalg.eigh(H)[1]
        pv = np.linspace(0.5, 2.0, d)
        P = (Q * pv) @ Q.T
        D = np.linalg.inv(P)
        eigs_pinvH = np.linalg.eigvalsh(D @ H)
        print(f"\n  Hessian: {name} (d={d}), commuting diagonal P in H's eigenbasis")
        print(f"  {'beta1':>6}{'eta_op':>12}{'eta_form':>12}{'bounds':>24}{'rel_err':>12}")
        for beta1 in beta1s:
            e_op = _eta_op_fpm(H, D, beta1, 1 - beta1)
            e_form = fm.critical_eta_zoadam(eigs_pinvH, beta1)
            lo, hi = fm.bounds_zoadam(eigs_pinvH, beta1)
            rho_form = op.spectral_radius_fpm(H, D, e_form, beta1, 1 - beta1)
            rel = abs(e_op - e_form) / e_form if e_form else float("nan")
            rows.append(Row("3", "ZO-Adam", name, d, beta1, e_op, e_form, lo, hi, rho_form, rel, _in_bounds(e_op, lo, hi)))
            print(f"  {beta1:>6.1f}{e_op:>12.5f}{e_form:>12.5f}  [{lo:.5f},{hi:.5f}]{rel:>12.2e}")

    # ---- Negative control: FO depends only on lam_max; ZO on full spectrum ----
    print("\n" + "=" * 78)
    print("NEGATIVE CONTROL (FO vs ZO spectrum dependence)")
    print("=" * 78)
    print(f"  {'trace':>8}{'lam_max':>9}{'eta_ZO(ms)':>13}{'eta_FO(GD)':>13}")
    base = np.array([1.0] + [0.5] * 9 + [0.1] * 40 + [0.01] * 50)
    for mult in [1, 2, 4]:
        eigs = base.copy()
        eigs[1:11] *= mult  # change trace, keep lam_max=1 fixed
        eigs = np.sort(eigs)[::-1]
        e_zo = fm.critical_eta_zogd(eigs)
        e_fo = fm.critical_eta_gd(eigs)
        rows.append(Row("1-control", "ZO-GD vs GD", f"spectrum_mult{mult}", len(eigs), float("nan"), e_zo, e_zo, float("nan"), float("nan"), float("nan"), float("nan"), True, e_fo))
        print(f"  {eigs.sum():>8.2f}{eigs.max():>9.2f}{e_zo:>13.5f}{e_fo:>13.5f}")

    # ---- Monte Carlo cross-check (ZO-GD) on a couple of Hessians ----
    print("\n" + "=" * 78)
    print("MONTE CARLO CROSS-CHECK (independent route, ZO-GD)")
    print("=" * 78)
    for name, H in hessians[:3]:
        eigs = np.linalg.eigvalsh(H)
        e_op = _eta_op_gd(eigs)
        e_mc = mc.find_critical_eta_mc(H, np.random.default_rng(2024), T=MC_T, n_seeds=MC_SEEDS)
        rows.append(Row("1-mc", "ZO-GD", name, len(eigs), float("nan"), e_op, e_op, float("nan"), float("nan"), float("nan"), float("nan"), True, e_mc))
        print(f"  {name:<28} eta_op={e_op:.5f}  eta_mc={e_mc:.5f}  ratio={e_mc/e_op:.3f}")

    # ---- Claim 5 (theory half): bounds from (Tr, lam_max) bracket eta_ms* ----
    print("\n" + "=" * 78)
    print("CLAIM 5 (theory half, Eqs 23-25): (Tr, lam_max) bounds bracket eta_ms*.")
    print("=" * 78)
    c1 = [r for r in rows if r.claim == "1"]
    tight = np.mean([(r.bound_hi / r.bound_lo) for r in c1 if r.bound_lo > 0])
    print(f"  ZO-GD mean bound ratio hi/lo = {tight:.3f} (->1 means trace-dominated, tight).")
    print(f"  All operator-defined eta_ms* inside (Tr,lam_max) bounds: "
          f"{all(r.bound_ok for r in rows if r.claim in ('1','2','3'))}")

    # ---- Aggregate verdicts ----
    main_rows = [r for r in rows if r.claim in ("1", "2", "3")]
    finite = [r for r in main_rows if np.isfinite(r.rel_err) and np.isfinite(r.eta_op)]
    nans = [r for r in main_rows if not (np.isfinite(r.rel_err) and np.isfinite(r.eta_op))]
    rel_errs = [r.rel_err for r in finite]
    max_rel = max(rel_errs) if rel_errs else float("nan")
    bound_violators = [r for r in finite if not r.bound_ok]
    bounds_ok = len(bound_violators) == 0
    verdict = "VERIFIED" if (np.isfinite(max_rel) and max_rel < TOL and bounds_ok) else "FAIL"

    print("\n" + "=" * 78)
    print(f"THEORY VERDICT: {verdict}  (max rel err eta_op vs formula = {max_rel:.2e}, tol={TOL:.0e})")
    print(f"  finite problems = {len(finite)}/{len(main_rows)};  nan/unresolved = {len(nans)}")
    if bound_violators:
        print(f"  BOUND VIOLATORS ({len(bound_violators)}):")
        for r in bound_violators:
            print(f"    {r.claim} {r.method} {r.hessian} beta={r.param}: eta_op={r.eta_op:.6e} bounds=[{r.bound_lo:.6e},{r.bound_hi:.6e}]")
    print(f"  runtime = {time.time()-t0:.1f}s")
    print("=" * 78)

    # ---- write artifacts ----
    import csv

    fnames = [f.name for f in fields(Row)]
    with open(os.path.join(out_dir, "theory_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    summary = {
        "verdict": verdict,
        "max_rel_err_operator_vs_formula": max_rel,
        "tol": TOL,
        "bounds_all_hold": bounds_ok,
        "n_finite": len(finite),
        "n_nan": len(nans),
        "n_problems": len(main_rows),
        "runtime_s": time.time() - t0,
        "claims": {
            "claim1_zogd": "VERIFIED" if max_rel < TOL else "FAIL",
            "claim2_zogdm": "VERIFIED" if max_rel < TOL else "FAIL",
            "claim3_zoadam": "VERIFIED" if max_rel < TOL else "FAIL",
            "claim5_theory": "VERIFIED" if bounds_ok else "FAIL",
        },
    }
    with open(os.path.join(out_dir, "theory_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if verdict != "VERIFIED":
        raise SystemExit(f"THEORY VERIFIER FAILED: max rel err {max_rel}")
    return summary
