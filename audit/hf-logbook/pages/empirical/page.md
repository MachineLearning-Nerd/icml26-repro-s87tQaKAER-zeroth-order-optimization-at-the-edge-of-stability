# Empirical evidence — Claims 4, 5 (empirical), 6 (Section 5, Figures 2–3)

> **Honest outcome: NOT reproduced at the reduced CPU scale.** The full-batch ZO
> sweep on a width-16 CNN (CIFAR-10) trains in a **low-curvature regime** with
> `Tr(H_t) ≈ 5–20`, far below the mean-square threshold `2/η ≈ 133–666` at any
> stable step size. The mean-square EoS band-tracking phenomenon (Claim 4) and
> the catapult (Claim 6) therefore **did not appear** at this scale. A faithful
> reproduction needs the paper's width-32 CNN + ResNet20 + ViT on a GPU (the paper
> uses an H100), which exceeds the `cpu-upgrade` budget (each width-32 run was
> still incomplete after ~5 h of wall-clock). This is a **documented scale
> mismatch, not a falsification** — the paper's claim is at its own (larger) scale.

## Exact claim contracts (source: arXiv:2604.14669 §5)

- **Claim 4 (Fig 2):** full-batch ZO-GD/ZO-GDM/ZO-Adam on a CNN stabilize near the
  predicted mean-square EoS threshold (`Tr(H_t) ≤ 2/η ≤ Tr(H_t)+2λ_max(H_t)`,
  Eq. 23, and analogues).
- **Claim 6 (Fig 3):** increasing η midway triggers catapult dynamics (loss spike,
  `Tr(H_t)` drops and re-equilibrates).

## Setup (Appendix C; CPU reductions stated)

CIFAR-10, first 4 classes, **500 examples** (paper: 1000), standardized, squared
loss. CNN width **16**, d = 7,476 params (paper: width 32, d ≈ 28,772). Two-point
estimator, μ = 1e-3, full-batch. Curvature every 200 iters via the **true
Hessian** (power iteration 12 iters + Hutchinson 12 probes, autograd HVPs).
HF `cpu-upgrade` (no GPU).

## Raw results (HF child run `76b95d3f`, commit b9a3840)

ZO-GD, three step sizes (EoS band fraction = post-burn-in checkpoints with
`2/η ∈ [Tr(H_t), Tr(H_t)+2λ_max(H_t)]`):

| η | 2/η (threshold) | Tr(H_t) range | λ_max(H_t) | band fraction | final loss |
| --- | --- | --- | --- | --- | --- |
| 3e-3 | 666.7 | 2.3 → 4.3 | ~1.25 | **0.00** | 0.376 |
| 8e-3 | 250.0 | 2.3 → 4.8 | ~1.28 | **0.00** | 0.370 |
| 1.5e-2 | 133.3 | 2.3 → 20.3 | ~1.4–7.2 | **0.00** | 0.337 |

The threshold `2/η` (133–667) sits **one-to-two orders of magnitude above** the
upper band edge `Tr(H_t)+2λ_max ≈ 5–35`. Training is stable and low-curvature; the
"sharpening toward 2/η" that defines EoS does not occur at this scale.

![ZO-GD curvature vs threshold on the width-16 CNN](./artifacts/empirical/images/eos_cnn.png)

ZO-GDM, β = 0.9, η = 5e-3 (threshold 40): **diverged** (loss → NaN by iter 200),
while all β = 0 ZO-GD runs at comparable η were stable. This is qualitatively
consistent with **Theorem 2** (Claim 2): increasing β *shrinks* the ZO-GDM
stable regime, so a step size stable for ZO-GD can be unstable for ZO-GDM at high
β — an empirical connection from the NN run to the verified theory.

ZO-Adam and the catapult experiment **did not complete** within the CPU budget
(run terminated at the wall-clock limit).

## Reproducibility

- **Code:** `zo_eos/nn_train.py`, `zo_eos/verify_empirical.py`; enabled by
  `STAGES = ["theory","empirical"]` on the empirical child branch.
- **Command:** `uv run --frozen --extra nn python -m zo_eos`.  Env: uv, Python
  3.12, torch 2.13 CPU.  Seed `SEED = 0`.
- **Raw trajectories:** [`traj_ZO-GD_eta0.003_betax.csv`](./artifacts/empirical/traj_ZO-GD_eta0.003_betax.csv),
  [`...eta0.008...`](./artifacts/empirical/traj_ZO-GD_eta0.008_betax.csv),
  [`...eta0.015...`](./artifacts/empirical/traj_ZO-GD_eta0.015_betax.csv),
  [`traj_ZO-GDM_eta0.005_beta0.9.csv`](./artifacts/empirical/traj_ZO-GDM_eta0.005_beta0.9.csv).

## Limitations / why this is BLOCKED, not VERIFIED

- **Scale mismatch (primary):** the paper's EoS requires `Tr(H_t)` to grow to
  `~2/η`. On the width-16 / 500-image CNN the Hessian trace stays ~5–35, so no
  stable η aligns `2/η` with the curvature. Reaching the paper's regime needs the
  larger width-32 CNN (≈4× params, ≈2× data) trained for many more iterations,
  which is infeasible on `cpu-upgrade` (a single width-32 run did not finish in
  ~5 h).
- **Architecture coverage:** ResNet20 and ViT not attempted (CPU budget).
- **Catapult:** not completed (Claim 6 has no direct evidence here).

**Verdict (Claims 4, 6): NOT reproduced at reduced scale — BLOCKED for the
faithful (paper-scale, GPU) reproduction.** The theoretical claims (1, 2, 3, 5),
verified independently to machine precision, remain the load-bearing result.
