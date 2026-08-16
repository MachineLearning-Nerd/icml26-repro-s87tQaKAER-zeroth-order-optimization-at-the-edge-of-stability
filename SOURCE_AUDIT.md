# Source and provenance audit

## Paper identity

- Title: Zeroth-Order Optimization at the Edge of Stability
- Authors: Minhak Song, Liang Zhang, Bingcong Li, Niao He, Michael Muehlebach,
  and Sewoong Oh
- arXiv identifier: 2604.14669
- Current paper version used for this audit: v2
- v2 revision date: 2026-07-01
- Venue note: ICML 2026
- OpenReview identifier: s87tQaKAER
- Paper record: https://arxiv.org/abs/2604.14669
- Review record: https://openreview.net/forum?id=s87tQaKAER

The paper studies mean-square linear stability for two-point Gaussian
zeroth-order methods. It gives exact spectrum-dependent formulas for ZO-GD,
ZO-GDM, and Frozen ZO-Adam, then derives trace/top-eigenvalue bounds and tests
them on full-batch neural-network training.

## Archived paper artifacts

| Artifact | Role | SHA-256 |
| --- | --- | --- |
| paper_2604.14669v1.pdf | original preprint retained for version comparison | 4aede09df7bc713437fdc6c8743c72c66735226e943a2010f0e3961e89fb9942 |
| paper_2604.14669v2.pdf | current paper and citation source | 1a37bea38e4f3518fa6ec93e149cbe7886bba0bb46719b3a0066ca465f6cdc39 |
| source/arxiv/2604.14669v1.tar | v1 TeX/source archive | bd532c759faad1ba1d7a74574b75237874968b8e7974ae653cf16d09a3fc3b41 |
| source/arxiv/2604.14669v2.tar | v2 TeX/source archive | 5acb3d2d3fc99e7a9810a0dbfedd351dbba86822c30d0c67cec2232095f84f84 |

The v1 PDF hash above is intentionally checked again before publication; the
source archive and PDF are separate files even when a downloaded source
package has a similar provenance role.

## Version boundary

The implementation targets the stable Theorem 1–3 and Section-5 contracts
present in v2. The source-era Hugging Face pages use older equation labels for
the same stability formulas; this repository records both the current theorem
numbers and the v2 Section-5 equation numbering where it matters.

This audit does not silently treat the paper's full CNN/ResNet/ViT/LSTM/Mamba
results as reproduced. The local code implements only a reduced CNN attempt,
and the two GPU-scale architecture families and sequence-model experiments
remain outside the observed evidence.

## Repository provenance

Original repository:

https://github.com/MachineLearning-Nerd/icml26-repro-s87tQaKAER-zeroth-order-optimization-at-the-edge-of-stability

Final repository:

https://github.com/MachineLearning-Nerd/icml26-zeroth-order-edge-of-stability-independent-audit

This is a distinct reproduction of the same paper as:

https://github.com/MachineLearning-Nerd/icml26-zeroth-order-edge-of-stability

The sibling repository is a separate finite-proxy audit. This repository
retains the exact covariance-operator route, its theory-only baseline, and its
CNN experiment child rather than merging the two projects.

## Evidence provenance

### Local theory snapshot

The archived local theory result under audit/faithful-theory was generated on
2026-08-16 from the current source tree with a direct theory-stage invocation.
It is a fresh execution of the operator verifier, not a copied summary:

    ZO_EOS_ARTIFACTS=audit/faithful-theory uv run --frozen --extra nn python -c 'from zo_eos import verify_theory; verify_theory.run()'

### Hugging Face snapshot

The selected external evidence was downloaded on 2026-08-16 from:

https://huggingface.co/spaces/DineshAI/s87tQaKAER

The Space metadata reported an update at 2026-07-26. The preserved files under
audit/hf-logbook/ include the current claims, theory, and empirical pages,
the theory summary and rows, the four available empirical trajectories, and
the CNN figure. They are a bounded mirror of the evidence used here; the
original Space may contain additional UI state or later revisions.

The empirical page identifies child run 76b95d3f and source commit b9a3840.
No completed ZO-Adam or catapult result is inferred from their absence.
