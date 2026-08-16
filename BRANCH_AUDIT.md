# Branch audit

## Original remote state

The repository was imported with one publication branch and two experiment
branches:

| Original branch | Tip before cleanup | Purpose |
| --- | --- | --- |
| main | 04fa5756688202eb6d459ace9d4e215ef25880ae | publication README, report, notebook, and integrated source |
| orx/baseline-theory-reproduction-claims-1-2-3-5 | d9665aa5a092286319623e1594f26bd37a8ecaee | theory-only verifier |
| orx/empirical-mean-square-eos-on-cnn-cifar-10-claims | b9a384038ca44dc6395ce25344f0fcfb59858aca | child with the corrected FPM dimension and reduced CNN experiment |

The two old experiment refs are useful provenance, but their names describe
the automation system rather than the scientific role.

## Final branch policy

| Final branch | Purpose | Allowed contents |
| --- | --- | --- |
| main | canonical paper-first repository surface | integrated source, audit documents, frozen evidence, citation, paper artifacts, and report |
| baseline/theory-reproduction | theory reproduction entry point | theory verifier, pinned environment, and the fixed theory-stage configuration |
| experiment/cnn-eos | empirical child entry point | theory verifier plus the reduced CNN training and curvature path |

The old orx/* refs are retired. No master ref is retained. The descriptive
branch names make the experiment lineage readable without needing knowledge of
the OpenResearch automation.

## Branch-to-claim map

- C1: exact ZO-GD covariance operator and Theorem-1 formula.
- C2: joint ZO-GDM covariance operator, beta sweep, and Theorem-2 formula.
- C3: structured Frozen ZO-Adam operator, commuting preconditioner, and
  Theorem-3 formula.
- C4: CNN ZO-GD/GDM/Adam band tracking; reduced run is blocked.
- C5: exact theory bounds plus the empirical curvature measurement path.
- C6: committed catapult training path; no completed result.

The branch-independent claim map is in claims.json. Captured outputs are under
audit/faithful-theory/ and audit/hf-logbook/.

## Verification requirements

The final GitHub repository must have exactly these three public branches:

    main
    baseline/theory-reproduction
    experiment/cnn-eos

Its default branch must be main, with no master or orx/* refs. Every reachable
commit must use:

    MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>

No Co-Authored-By trailer is allowed. verify_final.py checks the local
materialized refs and commit identities; the final GitHub branch list is
verified separately after publication.

## Published final state

Verified against GitHub on 2026-08-16 after the repository rename and
force-with-lease publication:

- final URL: https://github.com/MachineLearning-Nerd/icml26-zeroth-order-edge-of-stability-independent-audit;
- default branch: main;
- public branches: main, baseline/theory-reproduction, and experiment/cnn-eos;
- old orx/* branches: deleted;
- master branch: absent;
- repository owner: MachineLearning-Nerd;
- description and homepage: the paper-first audit description and the arXiv
  record, respectively.

The final branch tip hashes and the read-back verification output are recorded
in the ICML2026_REPOSITORIES.md tracker. The local pre-history-rewrite bundle
is retained at:

    /tmp/icml-zo-eos-before-history.1UIAv2/icml-zo-eos-before-history.bundle

SHA-256:

    010ecff3e74599878b42d837a654f153dc9fc235357e78d36c4376c8e1fc6ad0
