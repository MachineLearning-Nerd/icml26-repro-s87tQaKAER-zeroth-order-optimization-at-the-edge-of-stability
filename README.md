# ICML 2026 — Zeroth-Order Optimization at the Edge of Stability

This repository is a paper-first reproduction and claim audit for:

> Minhak Song, Liang Zhang, Bingcong Li, Niao He, Michael Muehlebach, and
> Sewoong Oh, “Zeroth-Order Optimization at the Edge of Stability,” ICML 2026,
> arXiv:2604.14669.

Paper links: [current arXiv record](https://arxiv.org/abs/2604.14669),
[OpenReview s87tQaKAER](https://openreview.net/forum?id=s87tQaKAER), and the
[historical Hugging Face logbook](https://huggingface.co/spaces/DineshAI/s87tQaKAER).

The final repository name is
`icml26-zeroth-order-edge-of-stability-independent-audit`. It is intentionally
distinct from the sibling
[icml26-zeroth-order-edge-of-stability](https://github.com/MachineLearning-Nerd/icml26-zeroth-order-edge-of-stability):
both are associated with this paper, but this repository preserves the
construction-led second-moment operator verifier, the full theory branch
history, and the later CNN experiment branch. Neither repository is affiliated
with or endorsed by the paper authors.

## Result at a glance

The audit labels the paper’s three theory theorems as C1–C3, the practical
tracking result as C5, and the two Section-5 empirical results as C4 and C6.
These labels make the code-to-evidence paths easy to follow; they are not
additional claims made by the paper.

| Audit claim | Reproduction result | What the result means |
| --- | --- | --- |
| C1 — ZO-GD mean-square stability (Theorem 1) | **VERIFIED, high confidence** | The exact operator root agrees with the paper’s formula across 13 dense/structured Hessians, up to *d* = 200 |
| C2 — ZO-GDM momentum dependence (Theorem 2) | **VERIFIED, high confidence** | 12 operator checks agree, and the critical step size decreases as β increases |
| C3 — Frozen ZO-Adam (Theorem 3) | **VERIFIED, high confidence** | 9 commuting-preconditioner checks agree and depend on the spectrum of `P⁻¹H` |
| C4 — neural-network mean-square EoS (Section 5, Figure 2) | **BLOCKED at reduced CPU scale** | The width-16/500-example CNN stayed far below the paper’s curvature scale; ResNet and ViT were not run |
| C5 — trace/top-eigenvalue tracking (Eqs. 23–25) | **VERIFIED for the theory half; empirical half blocked** | The bounds bracket the exact operator root for all 34 theory problems; neural-network tracking needs paper-scale compute |
| C6 — catapult dynamics (Section 5, Figure 3) | **BLOCKED** | The experiment did not finish within the CPU budget |

The local theory snapshot reports:

- `max|rho(operator @ eta_formula) - 1| = 2.53e-14` (tolerance `1e-5`);
- operator-root/formula relative error at most `5.35e-11`;
- all 34 theory problems finite, with every `(Tr(H), lambda_max(H))` bound holding.

This is scoped reproduction evidence, not a proof of the universally
quantified theorems and not an ICML review score.

## How each claim is produced

The theory path constructs the exact covariance operator from the two-point
Gaussian estimator and Isserlis/Wick fourth moments. It then compares two
independent quantities:

1. the spectral-radius boundary of the operator constructed by this
   repository; and
2. the closed-form critical step size stated by the paper.

The empirical path trains a full-batch CNN with the same two-point estimator,
measures the true Hessian with autograd Hessian-vector products, and checks
whether the paper’s trace/eigenvalue interval contains the fixed threshold.

| Audit claim | Production path | Evidence path |
| --- | --- | --- |
| C1 | `zo_eos/operators.py` → `zo_eos/formulas.py` → `zo_eos/verify_theory.py` | `audit/faithful-theory/` and `audit/hf-logbook/pages/theory/` |
| C2 | `zo_eos/operators.py` joint `(x,m)` operator → `zo_eos/formulas.py` → `verify_theory.py` | `audit/faithful-theory/` and `audit/hf-logbook/pages/theory/` |
| C3 | structured preconditioned operator in `operators.py` → `formulas.py` → `verify_theory.py` | `audit/faithful-theory/` and `audit/hf-logbook/pages/theory/` |
| C4 | `zo_eos/nn_train.py::train_zogd`, `train_zogdm`, `train_zoadam` → `eos_fraction` | `audit/hf-logbook/pages/empirical/` and the preserved trajectory CSVs |
| C5 | `verify_theory.py` bound checks; empirical `curvature` and `curvature_precond` in `nn_train.py` | theory summary plus the empirical logbook page |
| C6 | `zo_eos/nn_train.py::train_catapult` → `_catapult_spikes` | empirical page records that the run was not completed |

For the exact claim wording, controls, observed values, and limitations, read
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md).

## Reproduce

The experiment command is fixed across the two historical experiment branches;
the committed `zo_eos/config.py` selects the stages.

### Theory branch

```text
git clone https://github.com/MachineLearning-Nerd/icml26-zeroth-order-edge-of-stability-independent-audit.git
cd icml26-zeroth-order-edge-of-stability-independent-audit
git switch baseline/theory-reproduction
uv sync --extra nn
uv run --frozen --extra nn python -m zo_eos
```

This branch runs the fast theory stage only. It does not download CIFAR-10.

### CNN experiment branch

```text
git switch experiment/cnn-eos
uv run --frozen --extra nn python -m zo_eos
```

This branch enables theory plus the reduced CPU CNN experiment. It downloads
CIFAR-10 to `/tmp/cifar_data` if needed and can take about an hour or more on
CPU. Its run is intentionally recorded as partial when the paper-scale
phenomenon is not reached.

Generated outputs belong under `.openresearch/artifacts/`, which is ignored.
The reviewed theory output and the selected external logbook artifacts are
frozen under `audit/` so the documented result does not depend on a mutable
runtime directory.

## Branches

| Final branch | Purpose |
| --- | --- |
| `main` | Canonical paper-first README, audit documents, source, paper artifacts, and frozen evidence |
| `baseline/theory-reproduction` | Historical theory-only verifier and its fixed environment |
| `experiment/cnn-eos` | Historical child that fixes the theory dimension mismatch and enables the reduced CNN experiment |

The old `orx/*` names are retired. There is no `master` branch. The exact
pre-cleanup tips, final branch policy, and history backup are recorded in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md).

## Repository map

```text
zo_eos/                         theory and empirical implementation
notebook/zo_eos_demo.py        bounded interactive theory demonstration
reports/zo-eos/                original illustrated report and figures
audit/faithful-theory/         local theory run log and raw outputs
audit/hf-logbook/              selected immutable mirror of the HF evidence
source/arxiv/                  archived arXiv source packages
paper_2604.14669v1.pdf         original paper version
paper_2604.14669v2.pdf         current paper version
CLAIM_EVIDENCE.md              claim-to-code-to-evidence audit
STATUS.md                      current verdicts and limitations
SOURCE_AUDIT.md                paper and repository provenance
ENVIRONMENT.md                 pinned environment and compute boundary
BRANCH_AUDIT.md                branch and history policy
CITATION.cff                   citation metadata
AUTHOR_THANK_YOU.md            thank-you note to the paper authors
claims.json                    machine-readable claim map
EVIDENCE_MANIFEST.json         hashes for the archived evidence
verify_final.py                lightweight final-state verifier
```

## Citation

If this audit or its implementation is useful, please cite the paper and this
repository. The paper citation is also recorded in [CITATION.cff](CITATION.cff):

```bibtex
@inproceedings{song2026zeroth,
  title     = {Zeroth-Order Optimization at the Edge of Stability},
  author    = {Song, Minhak and Zhang, Liang and Li, Bingcong and He, Niao
               and Muehlebach, Michael and Oh, Sewoong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.14669},
  doi       = {10.48550/arXiv.2604.14669}
}
```

See [AUTHOR_THANK_YOU.md](AUTHOR_THANK_YOU.md) for the note to Minhak Song,
Liang Zhang, Bingcong Li, Niao He, Michael Muehlebach, and Sewoong Oh.

## Scope and limitations

The universal theorems are checked on a finite, deliberately varied set of
positive-semidefinite Hessians. The empirical run uses one width-16 CNN, 500
CIFAR-10 examples, 800 iterations, and 12 curvature probes on CPU; the paper
uses a width-32 CNN, 1,000 examples, longer runs, and also ResNet20, ViT, LSTM,
and Mamba experiments. Those differences are why C4 and C6 remain blocked
instead of being reported as reproduced.

See [STATUS.md](STATUS.md) for the current decision and
[SOURCE_AUDIT.md](SOURCE_AUDIT.md) for paper-version and provenance details.
