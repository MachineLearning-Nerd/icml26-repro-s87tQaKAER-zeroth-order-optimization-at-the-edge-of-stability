"""Which verification stages the entrypoint runs.

The run command is fixed (`uv run --frozen --extra nn python -m zo_eos`); what
runs is selected by this list committed on the branch, so different experiment
nodes vary *code* (this list) rather than the command.  Every node reruns the
theory stages; this empirical child additionally enables the NN training stages
(Claims 4, 5, 6).
"""

STAGES = [
    "theory",
    "empirical",
]
