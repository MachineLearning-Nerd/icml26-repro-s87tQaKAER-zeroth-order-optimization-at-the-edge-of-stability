# Reproduction environment

## Pinned project

The project requires Python 3.11–3.12 and uses uv.lock. The optional nn extra
adds PyTorch and torchvision; the theory formulas also use NumPy and SciPy.

Install:

    uv sync --extra nn

The fixed experiment command on both historical experiment branches is:

    uv run --frozen --extra nn python -m zo_eos

The committed zo_eos/config.py selects the stages:

- baseline/theory-reproduction: theory only;
- experiment/cnn-eos: theory plus empirical CNN stages;
- main: integrated source from the empirical child and the paper-first audit
  surface.

## Captured local environment

Run date: 2026-08-16

| Component | Observed value |
| --- | --- |
| Operating system | macOS-26.5.2-arm64-arm-64bit |
| Python | CPython 3.12.11 |
| uv | 0.8.15 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.1 |
| pandas | 3.0.5 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| CUDA | unavailable |

The local theory snapshot completed in 1.9 seconds on CPU. Its exact command
and output hashes are recorded in EVIDENCE_MANIFEST.json.

## Compute boundary

The theory verifier uses dense/structured positive-semidefinite Hessians up to
dimension 200, with a matrix-free structured operator for the momentum and
preconditioned cases. It does not download CIFAR-10 in the theory-only
configuration.

The empirical child is a deliberately reduced CPU attempt:

- CIFAR-10, first four classes, 500 examples rather than the paper's 1,000;
- a four-layer CNN of width 16 rather than width 32;
- 800 iterations and checkpoints every 200 iterations;
- 12 Hutchinson probes and 12 power iterations;
- one CNN only; ResNet20, ViT, LSTM, and Mamba were not run.

The paper's empirical regime uses longer full-batch runs and additional
architectures on GPU hardware. The scale difference is the reason C4 and C6
are marked BLOCKED.

## Re-run interpretation

Theory results use explicit per-spectrum seed offsets and should be
deterministic under the pinned environment. Empirical timings and trajectories
vary by hardware. Add a future GPU result as a new evidence snapshot with its
own command, environment, and hashes; do not overwrite the CPU result.
