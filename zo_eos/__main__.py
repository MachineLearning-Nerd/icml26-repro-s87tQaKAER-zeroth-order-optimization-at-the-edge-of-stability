"""Entrypoint: run the configured verification stages and print all evidence.

Run command (identical on every experiment node):

    uv run --frozen --extra nn python -m zo_eos

Stages are selected by `zo_eos.config.STAGES` (committed code, not env/flags).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

from zo_eos import config


def _header():
    import numpy as np

    print("=" * 78)
    print("Zeroth-Order Optimization at the Edge of Stability -- reproduction")
    print("Paper: arXiv:2604.14669 (OpenReview s87tQaKAER)")
    print("=" * 78)
    print(f"python   : {sys.version.split()[0]}")
    print(f"platform : {platform.platform()}")
    try:
        import torch

        print(f"torch    : {torch.__version__}  (cuda available={torch.cuda.is_available()})")
    except Exception:
        print("torch    : not installed (theory-only stage)")
    print(f"numpy    : {np.__version__}")
    print(f"stages   : {config.STAGES}")
    print(f"artifacts: {os.environ.get('ZO_EOS_ARTIFACTS', '.openresearch/artifacts/<stage>')}")
    print("=" * 78)


def main():
    _header()
    t0 = time.time()
    results = {}
    for stage in config.STAGES:
        print(f"\n>>> STAGE: {stage}\n")
        if stage == "theory":
            from zo_eos import verify_theory

            results["theory"] = verify_theory.run()
        elif stage == "empirical":
            from zo_eos import verify_empirical

            results["empirical"] = verify_empirical.run()
        else:
            raise ValueError(f"unknown stage {stage!r}")
    print(f"\n{'=' * 78}\nTOTAL RUNTIME: {time.time()-t0:.1f}s\n{'=' * 78}")
    print("RESULT SUMMARY:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
