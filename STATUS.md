# Reproduction status

Audit date: 2026-08-16

Paper contract: arXiv 2604.14669v2

Current paper record: arXiv 2604.14669v2, ICML 2026

Overall status: **MIXED_RESULTS / PARTIAL**

## Decision

The theory path independently verifies the three mean-square stability formulas
and the trace/top-eigenvalue bounds on 34 finite problems. The neural-network
path is only a reduced CPU attempt: the width-16 CNN remained in a
low-curvature regime, ZO-GDM diverged at one tested setting, and the
paper-scale EoS/catapult experiments were not completed. C4 and C6 therefore
remain blocked rather than being counted as reproduced.

## Observed evidence

### Local theory snapshot

Command:

    ZO_EOS_ARTIFACTS=audit/faithful-theory uv run --frozen --extra nn python -c 'from zo_eos import verify_theory; verify_theory.run()'

Environment: CPython 3.12.11 on macOS arm64, CPU only; exact package versions
are in [ENVIRONMENT.md](ENVIRONMENT.md).

The run completed with exit code 0 in 1.9 seconds:

- primary `rho(operator @ eta_formula)` error: `2.53e-14` (tolerance `1e-5`);
- secondary operator-root/formula relative error: `5.34e-11`;
- finite theory problems: `34/34`;
- `(Tr(H), lambda_max(H))` bounds: all hold;
- theory verdict: `VERIFIED`.

The complete log, CSV, and JSON summary are under
`audit/faithful-theory/`.

### Historical empirical child snapshot

The selected Hugging Face logbook snapshot records child run `76b95d3f` at
source commit `b9a3840`, using the `experiment/cnn-eos` code path on
Hugging Face `cpu-upgrade`:

- CIFAR-10: first four classes, 500 examples;
- CNN width: 16, 7,476 parameters;
- 800 iterations, checkpoint every 200 iterations;
- two-point smoothing `mu = 1e-3`;
- true-Hessian curvature: 12 Hutchinson probes and 12 power iterations.

The three ZO-GD sweeps had EoS band fraction `0.00`; the threshold
`2/eta = 133–667` remained well above the observed curvature band. ZO-GDM
with `beta = 0.9` and `eta = 5e-3` diverged. ZO-Adam and the catapult run did
not complete within the CPU budget. The selected raw trajectories and page
text are preserved under `audit/hf-logbook/`.

## Claim outcomes

| Claim | Verdict | Confidence | Evidence summary |
| --- | --- | --- | --- |
| C1 — ZO-GD (Theorem 1) | **VERIFIED** | High | 13 operator/formula checks; root and bounds agree |
| C2 — ZO-GDM (Theorem 2) | **VERIFIED** | High | 12 checks; threshold decreases with beta |
| C3 — Frozen ZO-Adam (Theorem 3) | **VERIFIED** | High | 9 commuting-preconditioner checks; `P⁻¹H` spectrum is used |
| C4 — neural-network EoS (Figure 2) | **BLOCKED** | Low | reduced CNN did not reach the paper’s curvature scale; ResNet/ViT absent |
| C5 — trace/top-eigenvalue tracking (Eqs. 23–25) | **VERIFIED (theory)** | High for theory, low for empirical | bounds bracket all 34 exact theory roots; empirical scale is insufficient |
| C6 — catapult (Figure 3) | **BLOCKED** | Low | no completed catapult evidence |

## What remains open

- GPU-scale CNN, ResNet20, ViT, LSTM, and Mamba runs matching the paper’s
  Section-5 settings;
- a completed catapult run with loss and trace re-equilibration;
- empirical verification of C5 on the paper-scale trajectories;
- any claim that turns this finite audit into a proof of the universal
  theorems.
