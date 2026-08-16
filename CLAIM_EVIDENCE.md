# Claim-to-evidence audit

This document records the claim contract, the code that produces each result,
the observed evidence, and what that evidence does not prove. The
machine-readable version is [claims.json](claims.json); hashes for the
archived files are in [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json).

## Production pipelines

Theory:

    zo_eos/operators.py
      -> exact second-moment covariance operator
    zo_eos/formulas.py
      -> paper's closed-form critical step and bounds
    zo_eos/verify_theory.py
      -> operator root, formula root, bound check, Monte Carlo control
    audit/faithful-theory/
      -> run.log, theory_rows.csv, theory_summary.json

Empirical:

    zo_eos/nn_train.py
      -> full-batch two-point ZO training
      -> true-Hessian HVP curvature estimates
      -> band and catapult checks
    audit/hf-logbook/
      -> selected external run pages and raw trajectories

The theory verifier is intentionally independent of the paper's
cone/Krein–Rutman proof: it derives the covariance operator from the update
rule and Gaussian fourth moments, then evaluates the operator at the paper's
separately implemented formula root.

## C1 — ZO-GD mean-square stability

Paper contract: Theorem 1 in arXiv 2604.14669v2.

Claim tested: for a positive-semidefinite quadratic Hessian H, the
mean-square critical step size is the unique root of

    sum_i eta * lambda_i / (2 * (1 - eta * lambda_i)) = 1

and lies between 2 / (Tr(H) + 2 lambda_max(H)) and 2 / Tr(H).

Production path:

- zo_eos/operators.py implements the exact ZO-GD second-moment update;
- zo_eos/formulas.py implements the paper root and bounds;
- zo_eos/verify_theory.py evaluates the operator spectral radius at both
  roots and runs a Monte Carlo cross-check;
- the tested Hessians include dense non-diagonal random-eigenbasis matrices,
  structured spectra, and a 200-dimensional diagonal control.

Observed evidence:

- 13 finite C1 problems;
- rho(operator @ eta_formula) differs from one by at most 2.53e-14;
- the operator-defined root and formula root differ by at most 5.34e-11
  relative error across the combined theory run;
- every tested operator root lies inside the trace/top-eigenvalue bounds;
- the Monte Carlo cross-check is within a few percent on three 40-dimensional
  dense Hessians.

Evidence files:

- audit/faithful-theory/theory_summary.json
- audit/faithful-theory/theory_rows.csv
- audit/faithful-theory/run.log
- audit/hf-logbook/pages/theory/page.md

Interpretation: the independent finite operator checks strongly corroborate
Theorem 1. They do not replace the theorem's universal proof.

## C2 — ZO-GDM momentum dependence

Paper contract: Theorem 2 in arXiv 2604.14669v2.

Claim tested: the ZO-GDM mean-square critical step size is the unique root of
the paper's beta-adjusted spectral equation and satisfies the corresponding
trace/top-eigenvalue bounds. Unlike first-order GDM, the ZO threshold
decreases as beta increases.

Production path:

- zo_eos/operators.py builds the joint covariance operator for (x, m);
- zo_eos/formulas.py implements the beta-adjusted root and bounds;
- zo_eos/verify_theory.py sweeps beta in {0.0, 0.3, 0.6, 0.9} on three
  dense 40-dimensional Hessians and asserts monotonic decrease.

Observed evidence:

- 12 finite operator/formula comparisons;
- all roots and bounds agree;
- on the decay spectrum, the threshold decreases from about 0.466 at
  beta 0.0 to about 0.054 at beta 0.9;
- the first-order GDM control has the opposite beta dependence;
- the reduced CNN experiment's beta 0.9 run diverges at eta = 5e-3,
  which is qualitative context, not a theory proof.

Evidence files:

- audit/faithful-theory/theory_summary.json
- audit/faithful-theory/theory_rows.csv
- audit/faithful-theory/run.log
- audit/hf-logbook/pages/theory/page.md
- audit/hf-logbook/pages/empirical/page.md
- audit/hf-logbook/artifacts/empirical/traj_ZO-GDM_eta0.005_beta0.9.csv

Interpretation: the operator check verifies the finite theory instances and
the monotonicity assertion used by this audit. It does not claim to reproduce
the paper's neural-network EoS experiment.

## C3 — Frozen ZO-Adam

Paper contract: Theorem 3 in arXiv 2604.14669v2, under its commutativity
assumption PH = HP.

Claim tested: the critical root and bounds are governed by the eigenvalues of
the preconditioned Hessian P-inverse-H, not by the raw Hessian spectrum alone.

Production path:

- zo_eos/operators.py implements the structured preconditioned joint
  operator;
- zo_eos/formulas.py evaluates the root and bounds using the spectrum of
  P-inverse-H;
- zo_eos/verify_theory.py uses a diagonal preconditioner in H's eigenbasis
  and sweeps beta1 in {0.1, 0.5, 0.9} across three dense Hessians.

Observed evidence:

- 9 finite operator/formula comparisons;
- rho(operator @ eta_formula) agrees with one to the displayed
  machine-precision scale;
- all roots lie inside the preconditioned trace/top-eigenvalue bounds;
- the reported root changes with the preconditioned spectrum as expected.

Evidence files:

- audit/faithful-theory/theory_summary.json
- audit/faithful-theory/theory_rows.csv
- audit/faithful-theory/run.log
- audit/hf-logbook/pages/theory/page.md

Interpretation: this is an audit of the frozen, commuting preconditioner
model in Theorem 3. It is not a verification of arbitrary non-commuting
adaptive Adam dynamics.

## C4 — neural-network mean-square EoS

Paper contract: Section 5, Figure 2 in v2. The earlier source-era logbook
also calls this the empirical Claim 4.

Claim tested: during full-batch ZO-GD, ZO-GDM, and ZO-Adam training, the
threshold remains within or close to the paper's curvature interval. For
ZO-GD the interval is:

    Tr(H_t) <= 2 / eta <= Tr(H_t) + 2 * lambda_max(H_t)

The momentum and Adam variants use their corresponding theoretical terms.

Production path:

- zo_eos/nn_train.py::train_zogd, train_zogdm, and train_zoadam perform
  full-batch two-point updates;
- curvature and curvature_precond estimate true-Hessian trace and top
  eigenvalue using autograd HVPs;
- eos_fraction measures post-burn-in band membership.

Observed evidence:

- the selected child run used a width-16 CNN with 7,476 parameters, 500
  CIFAR-10 examples, 800 iterations, and CPU-only curvature estimates;
- all three ZO-GD sweeps have EoS band fraction 0.00;
- observed trace values remain approximately 5–35, while stable tested
  thresholds are approximately 133–667;
- ResNet20 and ViT were not attempted;
- the paper-scale GPU regime was not reached.

Evidence files:

- audit/hf-logbook/pages/empirical/page.md
- audit/hf-logbook/artifacts/empirical/traj_ZO-GD_eta0.003_betax.csv
- audit/hf-logbook/artifacts/empirical/traj_ZO-GD_eta0.008_betax.csv
- audit/hf-logbook/artifacts/empirical/traj_ZO-GD_eta0.015_betax.csv
- audit/hf-logbook/artifacts/empirical/images/eos_cnn.png

Interpretation: C4 is BLOCKED at reduced CPU scale, not falsified. The
experiment is too small to test the paper's reported regime.

## C5 — trace/top-eigenvalue tracking

Paper contract: the practical tracking result in Section 5 (equations
numbered (5)–(7) in v2; the source-era logbook refers to the same family as
Eqs. 23–25).

Claim tested: the exact full-spectrum stability root can be bracketed during
training using only the trace and top eigenvalue of the relevant Hessian (or
preconditioned Hessian).

Production path:

- zo_eos/verify_theory.py checks that the (Tr(H), lambda_max(H)) bounds
  bracket the exact operator root for every C1–C3 problem;
- zo_eos/nn_train.py::curvature and curvature_precond implement the
  empirical trace/top-eigenvalue measurement path.

Observed evidence:

- all 34 theory problems satisfy the bounds;
- the mean upper/lower bound ratio for the ZO-GD rows is about 1.21;
- the empirical measurement code is present, but the reduced CNN does not
  reach the paper's EoS scale.

Evidence files:

- audit/faithful-theory/theory_summary.json
- audit/faithful-theory/theory_rows.csv
- audit/faithful-theory/run.log
- audit/hf-logbook/pages/theory/page.md
- audit/hf-logbook/pages/empirical/page.md

Interpretation: C5 is VERIFIED for its finite theory half and BLOCKED for
the paper-scale empirical half.

## C6 — catapult dynamics

Paper contract: Section 5, Figure 3 in v2. The earlier logbook calls this
empirical Claim 6.

Claim tested: increasing the ZO-GD step size during training produces a loss
spike and a transient drop/re-equilibration of the Hessian trace near the new
threshold.

Production path:

- zo_eos/nn_train.py::train_catapult increases the step size over committed
  segments;
- _catapult_spikes compares loss immediately before and after each increase;
- the figure writer records the loss and trace trajectory when a run
  completes.

Observed evidence:

- the code path is implemented;
- the selected CPU run did not complete the catapult experiment within its
  wall-clock budget;
- no completed loss/trace trajectory is treated as evidence.

Evidence file:

- audit/hf-logbook/pages/empirical/page.md

Interpretation: C6 is BLOCKED, with no claim of reproduction.
