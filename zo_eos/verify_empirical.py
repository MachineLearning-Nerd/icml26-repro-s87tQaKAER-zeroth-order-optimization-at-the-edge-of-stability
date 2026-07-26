"""Stage wrapper for the empirical NN-training verification (Claims 4, 5, 6)."""

from __future__ import annotations

import os

from zo_eos import nn_train


def run(out_dir: str | None = None) -> dict:
    out_dir = out_dir or os.environ.get(
        "ZO_EOS_ARTIFACTS", ".openresearch/artifacts/empirical"
    )
    return nn_train.run(out_dir=out_dir)
