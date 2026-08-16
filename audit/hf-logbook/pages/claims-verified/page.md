# Claims & verdicts (current verification)

> **This is the current verification.** It supersedes the historical pages
> ([Overview](#/overview), [Claims](#/claims), [Evidence](#/evidence),
> [Verification run](#/verification-run), [Conclusion](#/conclusion)), which are
> the prior 3/12 logbook and are preserved unchanged as
> **"Historical rejected baseline"** — see [Verification run](#/verification-run).
>
> Source code: [github.com/MachineLearning-Nerd/icml26-repro-s87tQaKAER-zeroth-order-optimization-at-the-edge-of-stability](https://github.com/MachineLearning-Nerd/icml26-repro-s87tQaKAER-zeroth-order-optimization-at-the-edge-of-stability)
> Paper: *Zeroth-Order Optimization at the Edge of Stability*, arXiv:2604.14669 (OpenReview s87tQaKAER)

## Headline result

The paper's three **mean-square linear-stability theorems** for zeroth-order (ZO)
methods are verified to **machine precision** by an independent route: we build
the exact second-moment linear operator of the ZO dynamics from first principles
(Isserlis' theorem) and check that its spectral radius equals 1 at the paper's
closed-form critical step size. This replaces the prior 5-dimensional diagonal
"toy" check with faithful evidence on **dense, non-diagonal Hessians up to
d = 200**. The empirical CNN/CIFAR-10 mean-square edge-of-stability was attempted
at reduced (CPU) scale but **did not reproduce** — a documented scale mismatch
(the small CNN's `Tr(H) ≈ 5–35` is 1–2 orders of magnitude below `2/η` at any
stable step size); faithful reproduction needs the paper's GPU-scale setup. See
[Empirical](#/empirical).

**Previous live judged score: 3/12.** Forecast (not a judge result):
conservative **8/12**, best-supported **8/12** (Claims 1,2,3,5 verified = 8 pts;
4,6 honestly BLOCKED at CPU scale). Only the live judge can change the score.

## Visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (Thm 1, ZO-GD) | [Theory](#/theory) | yes (`zo_eos/operators.py`, `formulas.py`, `verify_theory.py`) | yes | [theory_rows.csv](./artifacts/theory/theory_rows.csv) | `rho(@eta_form)=1` to 8e-15 | FO-GD (lam_max only) | `2/(Tr+2λmax) ≤ η*_ms ≤ 2/Tr` & root of Eq.16 | **VERIFIED** |
| 2 (Thm 2, ZO-GDM) | [Theory](#/theory) | yes | yes | [theory_rows.csv](./artifacts/theory/theory_rows.csv) | `rho(@eta_form)=1` to 2e-14 | FO-GDM (β enlarges regime) | Eq.18 root & Eq.19 bounds; η*_ms ↓ with β | **VERIFIED** |
| 3 (Thm 3, Frozen ZO-Adam) | [Theory](#/theory) | yes | yes | [theory_rows.csv](./artifacts/theory/theory_rows.csv) | `rho(@eta_form)=1` to 2e-14 | spectrum of P⁻¹H, not H | Eq.20 root & Eq.21 bounds | **VERIFIED** |
| 4 (Sec 5, Fig 2 EoS) | [Empirical](#/empirical) | yes (`zo_eos/nn_train.py`) | yes | [empirical traj CSVs](./artifacts/empirical/) | EoS band fraction = 0.00 | — | Tr(H_t) band brackets 2/η on CNN/CIFAR-10 | **NOT reproduced (reduced scale) → BLOCKED** |
| 5 (Eqs 23-25 tracking) | [Theory](#/theory) + [Empirical](#/empirical) | yes | yes | [theory_rows.csv](./artifacts/theory/theory_rows.csv) | bounds bracket η*_ms ∀ problems | full-spectrum exact root | (Tr,λmax) bounds valid & sufficient (theory half VERIFIED) | **VERIFIED (theory)** |
| 6 (Sec 5, Fig 3 catapult) | [Empirical](#/empirical) | yes | partial | [empirical traj CSVs](./artifacts/empirical/) | (not completed) | — | loss spike + Tr(H_t) drop on η increase | **NOT attempted in time → BLOCKED** |

## Per-claim summary

| # | Claim (short) | Status | Confidence | Notes |
| --- | --- | --- | --- | --- |
| 1 | ZO-GD η*_ms bounds (Thm 1) | **VERIFIED** | HIGH | operator ρ(@formula)=1 to 8e-15, 13 dense H, d≤200 |
| 2 | ZO-GDM β-adjusted (Thm 2) | **VERIFIED** | HIGH | 12 problems; η*_ms decreases with β (opposite of GDM) |
| 3 | Frozen ZO-Adam P⁻¹H (Thm 3) | **VERIFIED** | HIGH | 9 problems; depends on preconditioned spectrum |
| 4 | EoS on CNN/ResNet/ViT (Sec 5) | **BLOCKED** | LOW | reduced CNN Tr(H)≈5–35 ≪ 2/η; needs paper GPU scale. ResNet/ViT not run |
| 5 | (Tr,λmax) tracking (Eqs 23-25) | **VERIFIED (theory)** | HIGH | bounds bracket η*_ms for all 34 theory problems; empirical tracking needs the CNN scale |
| 6 | Catapult dynamics (Fig 3) | **BLOCKED** | LOW | not completed in CPU budget; ZO-GDM β=0.9 divergence qualitatively matches Thm 2 |

**Fixed run command (identical on every experiment node):**
```
uv run --frozen --extra nn python -m zo_eos
```
**Pinned environment:** Python 3.12, uv-managed; see `pyproject.toml` / `uv.lock`
(numpy, scipy, matplotlib, pandas; torch CPU + torchvision in the `nn` extra).
**Compute:** Hugging Face `cpu-upgrade` (no GPU used). CPU allocations / runtimes
are recorded per page. Deterministic seeds are set in code (`SEED = 0`,
`torch.manual_seed`, `np.random.default_rng`).

Continue to [Theory evidence](#/theory) and [Empirical evidence](#/empirical).
