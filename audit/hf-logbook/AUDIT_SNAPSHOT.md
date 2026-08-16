# Hugging Face evidence snapshot

This directory is a bounded local mirror of selected files from
https://huggingface.co/spaces/DineshAI/s87tQaKAER, downloaded on 2026-08-16.

The mirror preserves:

- the current claims, theory, and empirical pages;
- the historical baseline pages referenced by the logbook;
- the theory summary and raw theory rows;
- the four available empirical trajectory CSVs; and
- the empirical CNN figure.

The Space is a mutable external logbook. This snapshot is the evidence
surface used by this repository after download; its hashes are recorded in
the root EVIDENCE_MANIFEST.json. Missing trajectories are not treated as
successful runs: the empirical page explicitly records that ZO-Adam and the
catapult experiment did not complete within the CPU budget.
