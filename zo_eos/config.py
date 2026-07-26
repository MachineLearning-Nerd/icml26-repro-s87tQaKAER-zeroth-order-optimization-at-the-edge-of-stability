"""Which verification stages the entrypoint runs.

The run command is fixed (`uv run --frozen --extra nn python -m zo_eos`); what
runs is selected by this list committed on the branch, so different experiment
nodes vary *code* (this list) rather than the command.  Every node reruns the
theory stages; the empirical child additionally enables the NN training stages.
"""

# Theory stages (claims 1, 2, 3, theory-half of 5) -- fast, CPU-light.
STAGES = [
    "theory",
]
