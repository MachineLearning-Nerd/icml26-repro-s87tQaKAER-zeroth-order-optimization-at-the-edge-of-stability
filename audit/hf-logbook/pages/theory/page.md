# Theory evidence — Claims 1, 2, 3, 5 (Theorems 1–3, Eqs. 16–25)

## Exact claim contracts (source: arXiv:2604.14669, retrieved 2026-07-26 from ar5iv)

Let `H ⪰ 0, H ≠ 0` with eigenvalues `λ1 ≥ … ≥ λd`. For ZO-GD/ZO-GDM/Frozen-ZO-Adam
under the **linearized dynamics** (quadratic `f_quad`, two-point Gaussian
estimator `u∼N(0,I)`), the **mean-square** critical step size `η*_ms` is:

- **Claim 1 / Thm 1 (ZO-GD), Eq. 16–17:** `ηλ_max < 1` and
  `Σ_i ηλ_i / (2(1−ηλ_i)) = 1`; with bounds `2/(Tr+2λ_max) ≤ η*_ms ≤ 2/Tr`.
- **Claim 2 / Thm 2 (ZO-GDM), Eq. 18–19:** `ηλ_max < 1−β²` and
  `Σ_i ηλ_i / (2(1−β)(1−ηλ_i/(1−β²))) = 1`; bounds
  `2(1−β)/(Tr+2λ_max/(1+β)) ≤ η*_ms ≤ 2(1−β)/Tr`. (η*_ms **decreases** with β.)
- **Claim 3 / Thm 3 (Frozen ZO-Adam, commuting `PH=HP`), Eq. 20–21:** with
  `λ̃_i` = eigs of `P⁻¹H`: `Σ_i ηλ̃_i / (2(1−ηλ̃_i/(1+β1))) = 1`; bounds
  `2/(Tr(P⁻¹H)+2λ_max(P⁻¹H)/(1+β1)) ≤ η*_ms ≤ 2/Tr(P⁻¹H)`.
- **Claim 5 (Eqs. 23–25):** the above (Tr, λ_max) bounds are the practical
  tracking quantities — they depend only on the trace and top eigenvalue.

## Method (independent of the paper's derivation)

The paper proves these via a cone-preserving covariance operator + Krein–Rutman.
We instead verify by **direct construction of the exact second-moment linear
operator** from the update rule and Gaussian fourth moments (Isserlis/Wick):

`S_{t+1} = T_η(S_t)`, where for ZO-GD
`T_η(S) = S − η(HS+SH) + η²(Tr(H²S)·I + 2HSH)` (and the analogous joint
`(x,m)` operator for momentum/preconditioned variants — `zo_eos/operators.py`).
Mean-square stability ⟺ `ρ(T_η) < 1`. We compute `ρ(T_η)` **exactly**: in H's
eigenbasis the operator decouples into per-pair 4×4 blocks + a 3d×3d diagonal
block (`spectral_radius_fpm_structured`). The analytic apply is validated against
both a single-step Monte-Carlo expectation and dense Kronecker eig (small d).

**Non-circularity:** `ρ(T_η)` is derived from the update rule only; the paper's
closed-form root is a *separate* quantity. Requiring `ρ(T_η) @ η_formula = 1` is
a genuine check that the formula root IS the operator's stability boundary.

## Primary check (independent checker)

For every test Hessian, evaluate the operator's spectral radius **at the paper's
formula root** `η_formula`. It must equal 1.

> **max |ρ(operator @ η_formula) − 1| = 8.4e-15** (tol 1e-5) — HF run `331dadb7`
> (Claim 1) and `2.4e-14` over all 34 problems incl. Claims 2/3 (local
> reproduction; HF child run `0d4f03fd`, same code/seeds).
> **Secondary:** operator's own root matches the formula,
> `max|η_op − η_formula|/η_formula = 5.3e-11`.

## Claim 1 (ZO-GD) — raw results (HF run 331dadb7)

`η_op` = operator-defined critical η; `η_form` = paper Eq.16 root; bounds = Eq.17.

| Hessian (dense, random eigenbasis) | d | η_op | η_formula | bounds [lo,hi] | rel.err |
| --- | --- | --- | --- | --- | --- |
| decay | 40 | 0.46639 | 0.46639 | [0.38412, 0.62370] | 3.6e-14 |
| two_group | 40 | 0.08727 | 0.08727 | [0.08696, 0.09524] | 4.1e-13 |
| linear | 40 | 0.09276 | 0.09276 | [0.09009, 0.09901] | 1.1e-13 |
| uniform | 40 | 0.04762 | 0.04762 | [0.04762, 0.05000] | 1.5e-16 |
| decay | 100 | 0.43727 | 0.43727 | [0.35695, 0.55509] | 2.3e-14 |
| two_group | 100 | 0.03676 | 0.03676 | [0.03670, 0.03810] | 3.7e-14 |
| linear | 100 | 0.03858 | 0.03858 | [0.03810, 0.03960] | 8.3e-13 |
| uniform | 100 | 0.01961 | 0.01961 | [0.01961, 0.02000] | 3.0e-12 |
| decay | 200 | 0.41953 | 0.41953 | [0.34132, 0.51819] | 4.0e-13 |
| two_group | 200 | 0.01871 | 0.01871 | [0.01869, 0.01905] | 3.4e-13 |
| linear | 200 | 0.01954 | 0.01954 | [0.01942, 0.01980] | 2.0e-13 |
| uniform | 200 | 0.00990 | 0.00990 | [0.00990, 0.01000] | 0 |
| diag_decay | 200 | 0.46609 | 0.46609 | [0.38078, 0.61494] | 1.4e-13 |

`ρ(@η_formula) = 1` to ≤ 8.4e-15 for every row. All rows inside the bounds.

## Claim 2 (ZO-GDM) — η*_ms decreases with β (opposite of FO GDM)

| Hessian | β | η_op | η_formula | bounds | rel.err |
| --- | --- | --- | --- | --- | --- |
| decay | 0.0 | 0.46639 | 0.46639 | [0.38412, 0.62370] | 1.1e-14 |
| decay | 0.3 | 0.35016 | 0.35016 | [0.29504, 0.43659] | 4.1e-12 |
| decay | 0.6 | 0.20892 | 0.20892 | [0.17951, 0.24948] | 2.6e-13 |
| decay | 0.9 | 0.05378 | 0.05378 | [0.04696, 0.06237] | 1.9e-14 |
| two_group | 0.0/0.3/0.6/0.9 | 0.0873/0.0623/0.0360/0.0091 | (matches) | (in bounds) | ≤6e-12 |
| linear | 0.0/0.3/0.6/0.9 | 0.0928/0.0659/0.0380/0.0096 | (matches) | (in bounds) | ≤7e-14 |

Full 12-row table in [theory_rows.csv](./artifacts/theory/theory_rows.csv). η*_ms
strictly decreases as β ↑ (e.g. 0.466 → 0.054), the **opposite** of first-order
GDM whose threshold `2(1+β)/λ_max` increases with β — confirming the paper's
"momentum affects ZO and FO stability in opposite ways".

## Claim 3 (Frozen ZO-Adam) — depends on the spectrum of P⁻¹H

| Hessian | β1 | η_op | η_formula | bounds | rel.err |
| --- | --- | --- | --- | --- | --- |
| decay | 0.1/0.5/0.9 | 0.844/0.891/0.918 | (matches) | [0.697,1.020] | ≤6e-14 |
| two_group | 0.1/0.5/0.9 | 0.135/0.138/0.140 | (matches) | [0.132,0.146] | ≤2e-11 |
| linear | 0.1/0.5/0.9 | 0.132/0.133/0.134 | (matches) | [0.130,0.138] | ≤4e-13 |

The threshold is governed by `P⁻¹H`'s spectrum (not `H`'s); ρ(@η_formula)=1 to ≤6e-14.

## Negative control (FO vs ZO spectrum dependence)

| trace(H) | λ_max(H) | η*_ms (ZO-GD) | η_FO (GD) |
| --- | --- | --- | --- |
| 10.0 | 1.0 | 0.18578 | 2.0 |
| 14.6 | 1.0 | 0.12441 | 2.0 |
| 23.8 | 2.0 | 0.07395 | 1.0 |

Same λ_max, different trace → **different** ZO threshold but the FO threshold
`2/λ_max` is unchanged. This is the paper's central contrast: FO stability is set
by λ_max alone; ZO mean-square stability depends on the whole spectrum.

## Monte-Carlo cross-check (third independent route, ZO-GD)

`η_mc` from simulating the actual stochastic ZO-GD iterates (no operator, no
formula): `η_mc/η_op = 1.039, 1.033, 1.046` (decay/two_group/linear, d=40,
T=500, 48 seeds) — within the bounds and within a few % of the exact value.

## Claim 5 (theory half) — (Tr, λ_max) bounds are sufficient

Across all 34 theory problems the (Tr, λ_max) bounds bracket η*_ms (operator
root). Mean bound ratio `hi/lo = 1.21` (→ 1 when trace-dominated, i.e. tight).
Practical tracking therefore needs only Tr and λ_max, exactly as the paper claims.

## Reproducibility

- **Code:** `zo_eos/operators.py` (operator), `zo_eos/formulas.py` (paper
  formulas), `zo_eos/verify_theory.py` (verifier), `zo_eos/mc.py`,
  `zo_eos/hessians.py`.
- **Command:** `uv run --frozen --extra nn python -m zo_eos` (selects the
  `theory` stage via `zo_eos/config.py` on the baseline branch).
- **Env:** uv, Python 3.12, numpy/scipy (CPU). No GPU.
- **Seeds:** `np.random.default_rng` keyed per problem; deterministic.
- **Runtime:** 116 s on HF `cpu-upgrade` (Claim-1 run); ~3 s theory compute, the
  rest deps install. Local 1-core reproduction: 3 s.
- **Git SHAs:** baseline `d9665aa`; child (Claims 2/3 fix) `a8bc2d2`.
- **Raw data:** [theory_rows.csv](./artifacts/theory/theory_rows.csv),
  [theory_summary.json](./artifacts/theory/theory_summary.json).

## Limitations / deviations

- Theorems are **universally quantified** over PSD `H`; our 34 finite problems
  (dense, structured, d ≤ 200) are **scoped corroboration**, not a proof. They
  cover a spread of spectra (decay, two-group, linear, uniform) and a
  non-commuting-vs-commuting `P`. A machine-checked proof certificate is out of
  scope; the verification is the independent operator-ρ route + MC cross-check.
- Real-data CIFAR Hessian is wired but reserved for the empirical stage (no
  download in the fast theory stage).
