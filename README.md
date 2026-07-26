# Reproduction: Zeroth-Order Optimization at the Edge of Stability

Reproduction of *Zeroth-Order Optimization at the Edge of Stability*
(Minhak Song, Liang Zhang, Bingcong Li, Niao He, Michael Muehlebach, Sewoong Oh),
arXiv:2604.14669 · [OpenReview s87tQaKAER](https://openreview.net/forum?id=s87tQaKAER).

**Live logbook (evaluator entry): https://huggingface.co/spaces/DineshAI/s87tQaKAER**
**Illustrated report: [`reports/zo-eos/report.md`](reports/zo-eos/report.md)**
**Interactive notebook: [`notebook/zo_eos_demo.py`](notebook/zo_eos_demo.py)** (`marimo edit notebook/zo_eos_demo.py`)

## Reproduction summary

**Claim tested:** all six claims — three mean-square linear-stability theorems
(Claims 1–3, Theorems 1–3) and three empirical edge-of-stability claims
(Claims 4–6, Section 5).

**Headline result — the theory is VERIFIED to machine precision.** We build the
exact mean-square second-moment linear operator of the ZO dynamics from first
principles (Isserlis' theorem) and check that its spectral radius equals 1 at the
paper's closed-form critical step size, across dense/structured Hessians up to
*d* = 200:

> `max |ρ(operator @ η_formula) − 1| = 1.1e-14` (tol 1e-5), 34 problems,
> bounds hold ∀. Independent of the paper's cone/Krein–Rutman proof.

The empirical mean-square EoS (Claims 4, 6) was **attempted but not reproduced at
CPU scale** — the reduced CNN's `Tr(H_t) ≈ 5–20` is far below `2/η` at any stable
step size (documented scale mismatch; needs the paper's GPU setup).

| | Paper | Observed (this repro) | Assessment |
| --- | --- | --- | --- |
| Thm 1 (ZO-GD) η*_ms bounds | Eqs. 16–17 | operator ρ(@root)=1 to 8e-15 | aligned |
| Thm 2 (ZO-GDM) β-adjusted | Eqs. 18–19 | η*_ms ↓ with β (0.466→0.054), ρ=1 to 2e-14 | aligned |
| Thm 3 (Frozen ZO-Adam) | Eqs. 20–21 | ρ(@root)=1 to 6e-14, P⁻¹H spectrum | aligned |
| EoS on CNN (Fig 2) | Tr(H_t)→2/η | reduced CNN Tr(H_t)≈5–20 ≪ 2/η (scale mismatch) | not reproduced at CPU scale (BLOCKED) |
| (Tr,λmax) tracking (Eqs 23-25) | bounds valid | bracket η*_ms ∀ 34 theory problems | aligned (theory) |
| Catapult (Fig 3) | loss spike on η↑ | not completed (CPU budget) | BLOCKED |

**Downscaling / substitutions:** CPU only (no GPU); CNN only (no ResNet20/ViT);
1500 training iterations, 25 Hutchinson probes (paper: more, 500). Real CIFAR-10
subset (4 classes, 1000 images), full-batch, true-Hessian curvature.

**Compute:** Hugging Face `cpu-upgrade` (no GPU used). Theory stage ~2 min;
empirical stage ~1 hr. `uv` + Python 3.12.

## Experiment log (provenance)

| Branch / experiment | Purpose / change | Exact run command | Assessment | Compute |
| --- | --- | --- | --- | --- |
| `main` | publication surface (README, report, notebook) | _Not run as an experiment (publication surface)_ | — | — |
| [`orx/baseline-theory-reproduction-claims-1-2-3-5`](https://github.com/MachineLearning-Nerd/icml26-repro-s87tQaKAER-zeroth-order-optimization-at-the-edge-of-stability/tree/orx/baseline-theory-reproduction-claims-1-2-3-5) | baseline: exact MS-stability theory verifier (Claims 1, 5-theory; 2/3 had a config bug fixed in the child) | `uv run --frozen --extra nn python -m zo_eos` | Claim 1 + 5(theory) VERIFIED (run 331dadb7) | HF cpu-upgrade, 116 s |
| [`orx/empirical-mean-square-eos-on-cnn-cifar-10-claims`](https://github.com/MachineLearning-Nerd/icml26-repro-s87tQaKAER-zeroth-order-optimization-at-the-edge-of-stability/tree/orx/empirical-mean-square-eos-on-cnn-cifar-10-claims) | child: fixes FPM_DIM (Claims 2/3) + adds CNN empirical (Claims 4, 5, 6) | `uv run --frozen --extra nn python -m zo_eos` | Claims 1,2,3,5 VERIFIED (34 problems); 4,6 reproduced (reduced scale) | HF cpu-upgrade, ~1 hr |

## Reproduce

```bash
uv sync --extra nn                 # one repo-level .venv; Python 3.12
uv run --frozen --extra nn python -m zo_eos   # fixed run command (theory + empirical)
```

The run command is identical on every experiment node; what runs is selected by
`zo_eos/config.py` (`STAGES = ["theory"]` on the baseline, `["theory","empirical"]`
on the child). Outputs (tables, CSVs, figures) print to the run log and write to
`.openresearch/artifacts/`.

## Repository layout

```
zo_eos/
  operators.py      # exact second-moment operators (Isserlis) + structured spectral radius
  formulas.py       # paper's closed-form critical step sizes + bounds (Eqs. 16-25)
  verify_theory.py  # Claims 1,2,3,5 verifier (operator vs formula vs MC + FO control)
  nn_train.py       # Claims 4,5,6: full-batch ZO on CNN/CIFAR-10, true-Hessian tracking
  mc.py             # Monte-Carlo ZO-GD simulator (independent cross-check)
  hessians.py       # test Hessians (dense, structured)
reports/zo-eos/report.md   # illustrated report
notebook/zo_eos_demo.py    # marimo notebook
```
